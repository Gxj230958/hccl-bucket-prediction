"""Export held-out prediction and bucket-choice figures."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('evaluation',type=Path)
    args = parser.parse_args()
    data = json.loads(args.evaluation.read_text())
    rows = data['rows']
    x = range(len(rows))
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42})
    fig,ax = plt.subplots(figsize=(6.4,3.5),layout='constrained')
    measured = [r['measured_ms'] for r in rows]
    ax.errorbar(x,measured,yerr=[[r['measured_ms']-r['min_ms'] for r in rows],
                                [r['max_ms']-r['measured_ms'] for r in rows]],
                fmt='o-',capsize=3,label='Measured (range across 3 runs)')
    ax.plot(x,[r['predicted_ms'] for r in rows],'s--',label='Dependency graph')
    ax.plot(x,[r['additive_ms'] for r in rows],'^:',label='Simple sum')
    ax.set(xticks=list(x),xticklabels=[f"{r['bucket_mb']:g}" for r in rows],
           xlabel='DDP bucket cap (MiB)',ylabel='Training step (ms)',
           title=f"Qwen3-0.6B architecture, {data['world_size']} Ascend 910B NPUs")
    ax.spines[['top','right']].set_visible(False)
    ax.legend(frameon=False,fontsize=8)
    fig.savefig(args.evaluation.with_name('prediction.pdf'))
    fig.savefig(args.evaluation.with_name('prediction.png'),dpi=180)


if __name__ == '__main__':
    main()
