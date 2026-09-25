"""Compare like-for-like offline runs without averaging percentages across cases."""
import argparse
import json
import math
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, nargs='+', required=True)
    parser.add_argument('--after', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    before = [json.loads(path.read_text()) for path in args.before]
    after = [json.loads(path.read_text()) for path in args.after]
    first = before[0]
    for run in before + after:
        for field in ('python', 'platform', 'runner_sha256', 'dependencies', 'network'):
            if run[field] != first[field]:
                parser.error(f'comparison requires matching {field}')
        if run['network'] or run['cases'].keys() != first['cases'].keys():
            parser.error('comparison requires matching offline cases')
        for name, case in run['cases'].items():
            if any(case.get(field) != first['cases'][name].get(field) for field in ('source', 'scope')):
                parser.error(f'{name}: fixture or timing scope changed')
            samples = case['samples_ms']
            if (len(samples) != run['repeats'] or len(samples) < 3
                    or any(type(v) not in (float, int) or not math.isfinite(v) or v <= 0 for v in samples)):
                parser.error(f'{name}: expected at least three positive finite samples per run')
    for runs in (before, after):
        if len({run['source_sha256'] for run in runs}) != 1:
            parser.error('each variant must use one source snapshot')
    cases = {}
    for name in first['cases']:
        left = [v for run in before for v in run['cases'][name]['samples_ms']]
        right = [v for run in after for v in run['cases'][name]['samples_ms']]
        baseline, candidate = statistics.median(left), statistics.median(right)
        cases[name] = {
            'before_median_ms': baseline, 'after_median_ms': candidate,
            'time_reduction_percent': 100 * (1 - candidate / baseline),
            'speedup': baseline / candidate,
            'before_samples': len(left), 'after_samples': len(right),
            'scope': first['cases'][name]['scope'],
        }
    report = {
        'method': 'median of pooled raw samples per variant; no cross-case aggregate',
        'before_files': [str(path) for path in args.before],
        'after_files': [str(path) for path in args.after],
        'before_source_sha256': before[0]['source_sha256'],
        'after_source_sha256': after[0]['source_sha256'],
        'runner_sha256': first['runner_sha256'], 'cases': cases,
    }
    content = json.dumps(report, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content)
    else:
        print(content, end='')


if __name__ == '__main__':
    main()
