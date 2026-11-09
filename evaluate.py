"""Compare fixed predictions with independent DDP bucket measurements."""

import argparse
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory',type=Path)
    parser.add_argument('--prediction',type=Path)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    prediction_path = args.prediction or args.directory/'prediction.json'
    prediction = json.loads(prediction_path.read_text())
    rows = []
    for candidate in prediction['candidates']:
        mb = candidate['bucket_mb']
        runs = []
        for directory in sorted(args.directory.glob(f'eval_*_{mb:g}')):
            ranks = [json.loads(p.read_text()) for p in directory.glob('train_rank*.json')]
            by_step = {}
            for rank in ranks:
                for step in rank['steps']:
                    if not step['warmup']:
                        by_step.setdefault(step['step'],[]).append(step['step_ms'])
            runs.append(statistics.median(max(times) for times in by_step.values()))
        if len(runs) != 3:
            raise ValueError(f'Expected three independent runs for bucket {mb}; found {len(runs)}')
        measured = statistics.median(runs)
        rows.append({'bucket_mb':mb,'measured_ms':measured,'independent_runs_ms':runs,
                     'min_ms':min(runs),'max_ms':max(runs),
                     'predicted_ms':candidate['predicted_ms'],
                     'additive_ms':candidate['additive_ms'],
                     'relative_error':(candidate['predicted_ms']-measured)/measured,
                     'additive_relative_error':(candidate['additive_ms']-measured)/measured})
    best = min(rows,key=lambda r:r['measured_ms'])
    selected = next(r for r in rows if r['bucket_mb']==prediction['selected_bucket_mb'])
    default = next(r for r in rows if r['bucket_mb']==25.)
    result = {'world_size':prediction['world_size'],'rows':rows,
              'prediction':str(prediction_path),
              'cost_field':prediction.get('cost_field','ms'),
              'condition':prediction.get('condition','none'),
              'mean_absolute_relative_error':statistics.mean(abs(r['relative_error']) for r in rows),
              'additive_mean_absolute_relative_error':statistics.mean(abs(r['additive_relative_error']) for r in rows),
              'selected_bucket_mb':selected['bucket_mb'],'measured_best_bucket_mb':best['bucket_mb'],
              'selection_gap':selected['measured_ms']/best['measured_ms']-1,
              'speedup_over_default':default['measured_ms']/selected['measured_ms']}
    (args.output or args.directory/'evaluation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
