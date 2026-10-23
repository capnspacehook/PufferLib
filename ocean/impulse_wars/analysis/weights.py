"""Summarizes the first layer weights of a policy by input feature.

Usage: python weights.py weights.tsv [--top N] [--group name]

weights.tsv comes from iw_analyze --weights-out. Every input feature of the
projection (CNN output, one-hot type columns and scalars) and of the
projectile and floating wall record branches has its weight column's L2
norm (wnorm), and its mean and std over the policy's live steps. effect is
wnorm * std: how much the feature moves the layer's preactivations. A large
wnorm on a feature with a small std is the net compensating for a badly
scaled input; an effect near 0 means the feature is constant or ignored.
Groups sum effect in quadrature. Record branch effects are in their own
16 wide hidden layer, so only compare them within a branch.
"""
import argparse
import csv
import math
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('tsv')
    parser.add_argument('--top', type=int, default=25, help='features to list by effect, most and least')
    parser.add_argument('--group', help='list every feature of this group')
    args = parser.parse_args()
    with open(args.tsv) as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    for r in rows:
        for k in ('mean', 'std', 'wnorm', 'effect'):
            r[k] = float(r[k])

    groups = defaultdict(lambda: [0, 0.0, 0.0, 0])
    for r in rows:
        g = groups[(r['branch'], r['group'])]
        g[0] += 1
        g[1] += r['effect'] ** 2
        g[2] += r['wnorm'] ** 2
        g[3] += r['std'] < 1e-6
    print(f'{"branch":<14} {"group":<22} {"cols":>5} {"effect":>8} {"wnorm":>8} {"const":>6}')
    for (branch, group), (n, eff, wn, const) in sorted(groups.items(), key=lambda kv: (kv[0][0] != 'concat', -kv[1][1])):
        print(f'{branch:<14} {group:<22} {n:>5} {math.sqrt(eff):>8.3f} {math.sqrt(wn):>8.2f} {const:>6}')

    def show(title, rs):
        print(f'\n{title}')
        print(f'  {"branch":<14} {"feature":<34} {"mean":>8} {"std":>8} {"wnorm":>7} {"effect":>7}')
        for r in rs:
            print(f'  {r["branch"]:<14} {r["feature"]:<34} {r["mean"]:>8.4f} {r["std"]:>8.4f} {r["wnorm"]:>7.2f} {r["effect"]:>7.3f}')

    concat = [r for r in rows if r['branch'] == 'concat' and r['group'] != 'grid']
    concat.sort(key=lambda r: -r['effect'])
    show(f'top {args.top} non-grid concat features by effect', concat[:args.top])
    varying = [r for r in concat if r['std'] > 1e-6]
    show(f'bottom {args.top} varying non-grid concat features by effect', varying[-args.top:])
    scaled = sorted((r for r in varying if r['std'] < 0.2), key=lambda r: -r['wnorm'])
    show('largest weight norms on features with std < 0.2 (likely under-scaled)', scaled[:args.top])
    show('record branch features', [r for r in rows if r['branch'] != 'concat'])
    if args.group:
        show(f'group {args.group}', [r for r in rows if r['group'] == args.group])


if __name__ == '__main__':
    main()
