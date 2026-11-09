"""Collect HCCL bucket readiness and device traces from actual DDP training."""

import argparse
import contextlib
import json
import os
import random
from pathlib import Path
import time

import torch
import torch.distributed as dist
import torch_npu
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
from transformers import Qwen3Config, Qwen3ForCausalLM


class BucketTrace:
    def __init__(self, names):
        self.names = names
        self.records = []
        self.parameter_records = []
        self.origin = None

    def begin(self):
        self.records = []
        self.parameter_records = []
        self.origin = torch.npu.Event(enable_timing=True)
        self.origin.record()


def trace_hook(state, bucket):
    ready = torch.npu.Event(enable_timing=True)
    ready.record()
    state.records.append((ready, bucket.index(), bucket.buffer().numel()*bucket.buffer().element_size(),
                          [state.names[id(p)] for p in bucket.parameters()]))
    return default_hooks.allreduce_hook(None, bucket)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket-mb', type=float, default=25)
    parser.add_argument('--sequence', type=int, default=128)
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--layers', type=int, default=6)
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--microbench', action='store_true')
    parser.add_argument('--concurrent',choices=['none','cube'],default='none')
    parser.add_argument('--parameter-events', action='store_true')
    parser.add_argument('--model-config', type=Path)
    parser.add_argument('--seed', type=int, default=321)
    parser.add_argument('--data',type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    rank = int(os.environ['LOCAL_RANK'])
    torch.npu.set_device(rank)
    dist.init_process_group('hccl')
    args.output.mkdir(parents=True, exist_ok=True)
    if args.microbench:
        rows = []
        busy_stream = None
        if args.concurrent == 'cube':
            busy_stream = torch.npu.Stream()
            left = torch.randn(4096,4096,device=f'npu:{rank}',dtype=torch.bfloat16)
            right = torch.randn_like(left)
            product = torch.empty_like(left)
            with torch.npu.stream(busy_stream):
                torch.mm(left,right,out=product)
            torch.npu.synchronize()
        for size in [4096, 16384, 65536, 262144, 1048576, 4194304, 16777216, 67108864,
                     134217728, 268435456, 536870912]:
            tensor = torch.ones(size//2, device=f'npu:{rank}', dtype=torch.bfloat16)
            for _ in range(3):
                dist.all_reduce(tensor)
            for repeat in range(3):
                tensor.fill_(rank+1)
                dist.barrier()
                torch.npu.synchronize()
                busy_done = None
                if busy_stream is not None:
                    with torch.npu.stream(busy_stream):
                        for _ in range(32):
                            torch.mm(left,right,out=product)
                        busy_done = torch.npu.Event()
                        busy_done.record()
                device_start = torch.npu.Event(enable_timing=True)
                device_end = torch.npu.Event(enable_timing=True)
                device_start.record()
                compute_pending = not busy_done.query() if busy_done is not None else False
                start = time.perf_counter()
                dist.all_reduce(tensor)
                device_end.record()
                device_end.synchronize()
                elapsed = (time.perf_counter()-start)*1000
                device_ms = device_start.elapsed_time(device_end)
                torch.npu.synchronize()
                expected = dist.get_world_size()*(dist.get_world_size()+1)/2
                torch.testing.assert_close(tensor, torch.full_like(tensor, expected))
                rows.append({'bytes': size, 'repeat': repeat, 'ms': elapsed,
                             'device_ms':device_ms,'condition':args.concurrent,
                             'compute_pending_at_start':compute_pending,
                             'world_size': dist.get_world_size(), 'rank': rank, 'dtype': 'bfloat16'})
        (args.output/f'collectives_rank{rank}.json').write_text(json.dumps(rows,indent=2)+'\n')
        dist.destroy_process_group()
        return
    torch.manual_seed(args.seed)
    config = Qwen3Config(hidden_size=512, intermediate_size=1536,
                         num_hidden_layers=args.layers, num_attention_heads=8,
                         num_key_value_heads=2, head_dim=64, vocab_size=4096,
                         use_cache=False, attention_dropout=0.)
    if args.model_config:
        config = Qwen3Config.from_json_file(str(args.model_config))
        config.use_cache = False
    config._attn_implementation = 'eager'
    model = Qwen3ForCausalLM(config).to(device=f'npu:{rank}', dtype=torch.bfloat16)
    trace = BucketTrace({id(p): n for n,p in model.named_parameters()})
    if args.parameter_events:
        for name, parameter in model.named_parameters():
            def ready_hook(p, name=name):
                event = torch.npu.Event(enable_timing=True)
                event.record()
                trace.parameter_records.append((event, name, p.numel()*p.element_size()))
            parameter.register_post_accumulate_grad_hook(ready_hook)
    ddp = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], bucket_cap_mb=args.bucket_mb)
    ddp.register_comm_hook(trace, trace_hook)
    optimizer = torch.optim.AdamW(ddp.parameters(), lr=1e-4)
    torch.manual_seed(1234+rank)
    if args.data:
        dataset = json.loads(args.data.read_text())
        documents = [d['token_ids'] for d in dataset['documents'] if len(d['token_ids'])>=args.sequence]
        rng = random.Random(args.seed+rank)
        rows = []
        for _ in range(args.batch):
            document = rng.choice(documents)
            start = rng.randrange(len(document)-args.sequence+1)
            rows.append(document[start:start+args.sequence])
        tokens = torch.tensor(rows,device=f'npu:{rank}')
    else:
        tokens = torch.randint(0, config.vocab_size, (args.batch,args.sequence), device=f'npu:{rank}')
    rows = []
    profiler = (torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(args.output/'profiler')),
        schedule=torch_npu.profiler.schedule(wait=0,warmup=2,active=args.steps,repeat=1),
        record_shapes=True) if args.profile else contextlib.nullcontext())
    with profiler:
        for step in range(args.steps+2):
            optimizer.zero_grad(set_to_none=True)
            dist.barrier()
            torch.npu.synchronize()
            trace.begin()
            start = time.perf_counter()
            loss = ddp(input_ids=tokens, labels=tokens).loss
            loss.backward()
            backward_end = torch.npu.Event(enable_timing=True)
            backward_end.record()
            optimizer.step()
            torch.npu.synchronize()
            elapsed = (time.perf_counter()-start)*1000
            rows.append({'step': step, 'warmup': step<2, 'step_ms': elapsed,
                         'backward_complete_ms': trace.origin.elapsed_time(backward_end),
                         'loss': loss.item(),
                         'parameters': [{'name': name, 'bytes': size,
                                         'ready_ms': trace.origin.elapsed_time(event)}
                                        for event,name,size in trace.parameter_records],
                         'buckets': [{'ready_ms': trace.origin.elapsed_time(event), 'index': index,
                                      'bytes': size, 'parameters': names}
                                     for event,index,size,names in trace.records]})
            if args.profile:
                profiler.step()
    result = {'config': {key: str(value) if isinstance(value, Path) else value
                         for key,value in vars(args).items()}, 'rank': rank,
              'world_size': dist.get_world_size(), 'initialization': 'random',
              'training_inputs': {'source':dataset['source'],'partition':dataset['partition']}
                                 if args.data else 'seeded synthetic token IDs',
              'parameter_count': sum(p.numel() for p in model.parameters()), 'steps': rows}
    (args.output/f'train_rank{rank}.json').write_text(json.dumps(result,indent=2)+'\n')
    print(f'rank={rank} saved {args.output}',flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
