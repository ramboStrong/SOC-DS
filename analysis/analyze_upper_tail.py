from pathlib import Path
import argparse
import numpy as np
import pandas as pd

def percentile(values, q):
    try:
        return float(np.percentile(values, q, method='linear'))
    except TypeError:
        return float(np.percentile(values, q, interpolation='linear'))

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--a0_root', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--seeds', default='12345,23456,34567')
    p.add_argument('--expected_frames', type=int, default=4950)
    args = p.parse_args()
    root = Path(args.a0_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    seeds = [int(x) for x in args.seeds.split(',') if x.strip()]
    rows = []
    for seed in seeds:
        path = root / f'seed_{seed}' / 'best' / 'frame_results.csv'
        df = pd.read_csv(path)
        errors = pd.to_numeric(df['abs_error_total'], errors='raise').to_numpy(float)
        if len(errors) != args.expected_frames:
            raise ValueError(f'seed {seed}: expected {args.expected_frames} frames, found {len(errors)}')
        rows.append({'seed': seed, 'num_frames': len(errors), 'P95': percentile(errors, 95), 'P99': percentile(errors, 99), 'Maximum': float(np.max(errors))})
    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(out / 'a0_upper_tail_by_seed.csv', index=False)
    summary = []
    for metric in ['P95', 'P99', 'Maximum']:
        values = per_seed[metric].to_numpy(float)
        summary.append({'Statistic': metric, 'Mean': float(values.mean()), 'Sample_SD': float(values.std(ddof=1))})
    summary = pd.DataFrame(summary)
    summary.to_csv(out / 'a0_upper_tail_three_seed_summary.csv', index=False)
    print(summary.to_string(index=False))
if __name__ == '__main__':
    main()
