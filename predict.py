"""Predict DDP bucket schedules from measured gradient readiness and HCCL costs."""

import argparse
from bisect import bisect_right
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import statistics


@dataclass
class Node:
    name: str
    kind: str
    duration_ms: float
    dependencies: list[str]
    bytes: int = 0


def critical_path(nodes):
    finish = {}
    predecessor = {}
    for node in nodes:
        parent = max(node.dependencies, key=finish.__getitem__, default=None)
        start = finish[parent] if parent else 0.
        finish[node.name] = start+node.duration_ms
        predecessor[node.name] = parent
    tail = max(finish, key=finish.__getitem__, default=None)
    path = []
    while tail is not None:
        path.append(tail)
        tail = predecessor[tail]
    return max(finish.values(), default=0.), list(reversed(path)), finish


class CollectiveCosts:
    def __init__(self, rows, world_size, field='ms', condition='none'):
        grouped = {}
        for row in rows:
            if row['world_size'] == world_size and row.get('condition','none') == condition:
                grouped.setdefault((row['bytes'],row.get('run',0),row['repeat']), []).append(row[field])
        by_size = {}
        for (size,_,_), times in grouped.items():
            by_size.setdefault(size,[]).append(max(times))
        self.sizes = sorted(by_size)
        self.times = [statistics.median(by_size[size]) for size in self.sizes]
        if len(self.sizes) < 2:
            raise ValueError('At least two independently measured message sizes are required')

    def __call__(self, size):
        if size < self.sizes[0] or size > self.sizes[-1]:
            raise ValueError(f'{size} bytes is outside the calibrated message range')
        i = min(len(self.sizes)-2, max(0,bisect_right(self.sizes,size)-1))
        fraction = (size-self.sizes[i])/(self.sizes[i+1]-self.sizes[i])
        return self.times[i] + fraction*(self.times[i+1]-self.times[i])


def build_graph(parameters, bucket_bytes, costs, optimizer_ms=0.):
    """Use measured autograd order, whole-parameter buckets, one HCCL communicator."""
    by_rank = {}
    for parameter in parameters:
        by_rank.setdefault(parameter.get('rank', 0), []).append(parameter)
    for rank in by_rank:
        by_rank[rank].sort(key=lambda p:p['ready_ms'])
    parameters = by_rank[min(by_rank)]
    expected_names = {p['name'] for p in parameters}
    if any({p['name'] for p in rows} != expected_names for rows in by_rank.values()):
        raise ValueError('Each measured rank must contain the same parameters')
    compute, comm = [], []
    bucket, size = [], 0

    def flush():
        dependencies = list(bucket)
        if comm:
            dependencies.append(comm[-1].name)
        comm.append(Node(f'allreduce:{len(comm)}', 'communication', costs(size), dependencies, size))

    last_compute = []
    for rank, rows in by_rank.items():
        previous_compute = None
        previous_ready = 0.
        for parameter in rows:
            name = f'rank{rank}:gradient:{parameter["name"]}'
            compute.append(Node(name, 'compute', parameter['ready_ms']-previous_ready,
                                [previous_compute] if previous_compute else []))
            previous_compute, previous_ready = name, parameter['ready_ms']
        if previous_compute:
            last_compute.append(previous_compute)
    for parameter in parameters:
        bucket.extend(f'rank{rank}:gradient:{parameter["name"]}' for rank in by_rank)
        size += parameter['bytes']
        if size >= bucket_bytes:
            flush()
            bucket, size = [], 0
    if bucket:
        flush()
    nodes = compute+comm
    dependencies = last_compute + ([comm[-1].name] if comm else [])
    nodes.append(Node('optimizer', 'optimizer', optimizer_ms, dependencies))
    return nodes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace', type=Path, nargs='+', required=True)
    parser.add_argument('--collectives', type=Path, nargs='+', required=True)
    parser.add_argument('--world-size', type=int, required=True)
    parser.add_argument('--buckets-mb', type=float, nargs='+', default=[1,4,8,16,25,32])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cost-field',choices=['ms','device_ms'],default='ms')
    parser.add_argument('--condition',choices=['none','cube'],default='none')
    args = parser.parse_args()
    traces = [json.loads(path.read_text()) for path in args.trace]
    per_parameter = {}
    optimizer_times = []
    for trace in traces:
        steps = [s for s in trace['steps'] if not s['warmup']]
        if not steps or not steps[0]['parameters']:
            raise ValueError('Collect a training trace with --parameter-events')
        optimizer_times.append(statistics.median(max(0.,s['step_ms']-s['backward_complete_ms']) for s in steps))
        for step in steps:
            for parameter in step['parameters']:
                per_parameter.setdefault((trace['rank'],parameter['name']), []).append(parameter)
    parameters = [{'rank': rank, 'name': name, 'bytes': rows[0]['bytes'],
                   'ready_ms': statistics.median(r['ready_ms'] for r in rows)}
                  for (rank,name),rows in per_parameter.items()]
    rows = [r | {'run':str(path.parent)} for path in args.collectives
            for r in json.loads(path.read_text())]
    costs = CollectiveCosts(rows,args.world_size,args.cost_field,args.condition)
    optimizer_ms = max(optimizer_times)
    candidates = []
    for mb in args.buckets_mb:
        graph = build_graph(parameters,int(mb*1024**2),costs,optimizer_ms)
        predicted, path, finish = critical_path(graph)
        rank_compute = {}
        for node in graph:
            if node.kind == 'compute':
                rank = node.name.split(':',1)[0]
                rank_compute[rank] = rank_compute.get(rank,0.)+node.duration_ms
        additive = max(rank_compute.values())+sum(
            n.duration_ms for n in graph if n.kind != 'compute')
        candidates.append({'bucket_mb': mb, 'predicted_ms': predicted,
                           'additive_ms': additive,
                           'critical_path': path,
                           'nodes': [asdict(n) | {'finish_ms': finish[n.name]} for n in graph]})
    result = {'selected_bucket_mb': min(candidates,key=lambda c:c['predicted_ms'])['bucket_mb'],
              'world_size': args.world_size, 'source_traces': [str(p) for p in args.trace],
              'cost_field':args.cost_field,'condition':args.condition,
              'candidates': candidates}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'selected_bucket_mb': result['selected_bucket_mb'],
                      'predictions': [(c['bucket_mb'],c['predicted_ms']) for c in candidates]}))


if __name__ == '__main__':
    main()
