"""Extract Ascend operators, stream order and measured communication overlap."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def merge(intervals):
    merged = []
    for start,end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1],end)
        else:
            merged.append([start,end])
    return merged


def duration(intervals):
    return sum(end-start for start,end in merge(intervals))


def intersection(left,right):
    left,right = merge(left),merge(right)
    i = j = 0
    total = 0.
    while i<len(left) and j<len(right):
        total += max(0.,min(left[i][1],right[j][1])-max(left[i][0],right[j][0]))
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return total


def analyze(path):
    data = json.loads(path.read_text())
    events = data if isinstance(data,list) else data['traceEvents']
    processes = {e['pid']:e['args']['name'] for e in events if e.get('name')=='process_name'}
    tasks = [e for e in events if e.get('ph')=='X' and
             processes.get(e['pid'])=='Ascend Hardware']
    collectives = [e for e in events if e.get('ph')=='X' and
                   processes.get(e['pid'])=='Communication' and 'allReduce' in e['name']]
    compute = [e for e in tasks if e.get('args',{}).get('Task Type','').startswith('KERNEL_')]
    def intervals(rows):
        return [(float(e['ts']),float(e['ts'])+float(e['dur'])) for e in rows]
    compute_intervals,comm_intervals = intervals(compute),intervals(collectives)
    overlap = intersection(compute_intervals,comm_intervals)
    streams = defaultdict(list)
    names = Counter()
    for index,event in enumerate(sorted(tasks,key=lambda e:float(e['ts']))):
        stream = streams[str(event['tid'])]
        stream.append({'id':index,'name':event['name'],'start_us':float(event['ts']),
                       'duration_us':float(event['dur']),
                       'task_type':event.get('args',{}).get('Task Type'),
                       'dependencies':[stream[-1]['id']] if stream else [],
                       'connection_id':event.get('args',{}).get('connection_id')})
        names[event['name']] += float(event['dur'])
    flows = defaultdict(dict)
    for event in events:
        if event.get('ph') in ('s','f'):
            key = (event.get('cat'),event.get('name'),event.get('id'))
            flows[key][event['ph']] = {k:event.get(k) for k in ('pid','tid','ts')}
    return {
        'source':str(path),'operator_count':len(tasks),'allreduce_count':len(collectives),
        'compute_busy_ms':duration(compute_intervals)/1000,
        'allreduce_busy_ms':duration(comm_intervals)/1000,
        'compute_allreduce_overlap_ms':overlap/1000,
        'allreduce_without_compute_ms':(duration(comm_intervals)-overlap)/1000,
        'operators_by_total_ms':[{ 'name':n,'total_ms':t/1000 } for n,t in names.most_common()],
        'streams':dict(streams),
        'explicit_flows':[{'category':k[0],'name':k[1],'id':k[2],**v}
                          for k,v in flows.items() if 's' in v and 'f' in v],
        'dependency_scope':'Per-stream order and explicit profiler flows; gradient-to-collective dependencies are recorded by collect.py.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('trace',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    result = analyze(args.trace)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in (
        'streams','explicit_flows','operators_by_total_ms')}))


if __name__ == '__main__':
    main()
