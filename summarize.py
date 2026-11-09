"""Summarize rank-local validation, contention ablations, and critical paths."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from predict import CollectiveCosts


def path_statistics(candidate):
    nodes = {node['name']: node for node in candidate['nodes']}
    compute_end = max(node['finish_ms'] for node in nodes.values() if node['kind']=='compute')
    communication_end = max(node['finish_ms'] for node in nodes.values() if node['kind']=='communication')
    return {
        'bucket_mb': candidate['bucket_mb'],
        'predicted_ms': candidate['predicted_ms'],
        'communication_nodes': sum(node['kind']=='communication' for node in nodes.values()),
        'critical_path_communication_ms': sum(nodes[name]['duration_ms'] for name in candidate['critical_path']
                                              if nodes[name]['kind']=='communication'),
        'compute_end_ms': compute_end,
        'collective_end_ms': communication_end,
        'exposed_communication_tail_ms': max(0.,communication_end-compute_end),
    }


def layout_statistics(directory,candidate):
    expected = [(node['bytes'],{name.split(':gradient:',1)[1]
                 for name in node['dependencies'] if name.startswith('rank0:gradient:')})
                for node in candidate['nodes'] if node['kind']=='communication']
    checked = matched = 0
    for path in directory.glob(f"eval_*_{candidate['bucket_mb']:g}/train_rank*.json"):
        for step in json.loads(path.read_text())['steps']:
            if step['warmup']:
                continue
            actual = [(bucket['bytes'],set(bucket['parameters']))
                      for bucket in sorted(step['buckets'],key=lambda b:b['index'])]
            checked += 1
            matched += actual == expected
    return {'bucket_mb':candidate['bucket_mb'],'predicted_bucket_count':len(expected),
            'checked_rank_steps':checked,'matching_rank_steps':matched}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directories',type=Path,nargs='+')
    parser.add_argument('--output',type=Path,default=Path('results/summary.json'))
    args = parser.parse_args()
    summaries = []
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42})
    figure,axes = plt.subplots(1,len(args.directories),figsize=(5*len(args.directories),3.8),
                               squeeze=False,layout='constrained')
    for axis,directory in zip(axes.flat,args.directories):
        evaluation = json.loads((directory/'evaluation.json').read_text())
        prediction = json.loads((directory/'prediction.json').read_text())
        calibration = json.loads(next((directory/'calibration').glob('train_rank*.json')).read_text())
        summary = {'directory':str(directory),'world_size':evaluation['world_size'],
                   'sequence':calibration['config']['sequence'],'batch_per_rank':calibration['config']['batch'],
                   'training_inputs':calibration.get('training_inputs','seeded synthetic token IDs'),
                   'evaluation':evaluation,'critical_paths':[path_statistics(c) for c in prediction['candidates']],
                   'bucket_layouts':[layout_statistics(directory,c) for c in prediction['candidates']]}
        rows = evaluation['rows']
        x = list(range(len(rows)))
        measured = [r['measured_ms'] for r in rows]
        axis.errorbar(x,measured,yerr=[[r['measured_ms']-r['min_ms'] for r in rows],
                                      [r['max_ms']-r['measured_ms'] for r in rows]],
                      fmt='o-',capsize=3,label='Measured (3-run range)')
        axis.plot(x,[r['predicted_ms'] for r in rows],'s--',label='Graph, '+evaluation.get('condition','none'))
        axis.plot(x,[r['additive_ms'] for r in rows],'^:',label='Additive, '+evaluation.get('condition','none'))
        idle_path = directory/'evaluation_none.json'
        if idle_path.exists():
            idle = json.loads(idle_path.read_text())
            summary['idle_evaluation'] = idle
            axis.plot(x,[r['predicted_ms'] for r in idle['rows']],'d--',label='Graph, idle costs')
            collective_rows = [r | {'run':str(p.parent)}
                               for p in directory.glob('cost_*/collectives_rank*.json')
                               for r in json.loads(p.read_text())]
            tables = {}
            for condition in ('none','cube'):
                costs = CollectiveCosts(collective_rows,evaluation['world_size'],'device_ms',condition)
                tables[condition] = [{'bytes':size,'device_ms':time}
                                     for size,time in zip(costs.sizes,costs.times)]
            summary['collective_costs'] = tables
            busy = [r for r in collective_rows if r['condition']=='cube']
            summary['concurrency_records'] = len(busy)
            summary['compute_pending_records'] = sum(r['compute_pending_at_start'] for r in busy)
        axis.set(xticks=x,xticklabels=[f"{r['bucket_mb']:g}" for r in rows],
                 xlabel='DDP bucket cap (MiB)',ylabel='Training step (ms)',
                 title=f"{evaluation['world_size']} NPUs, sequence {summary['sequence']}")
        axis.spines[['top','right']].set_visible(False)
        axis.legend(frameon=False,fontsize=8)
        summaries.append(summary)
    args.output.write_text(json.dumps(summaries,indent=2)+'\n')
    figure.savefig(args.output.with_suffix('.pdf'))
    figure.savefig(args.output.with_suffix('.png'),dpi=180)
    print(json.dumps([{k:s[k] for k in ('directory','world_size','sequence')} |
                      {k:s['evaluation'][k] for k in ('mean_absolute_relative_error',
                       'selected_bucket_mb','measured_best_bucket_mb','selection_gap')}
                      for s in summaries],indent=2))


if __name__=='__main__':
    main()
