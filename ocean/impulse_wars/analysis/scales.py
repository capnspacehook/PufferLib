"""Summarizes iw_analyze --scales-out: the raw value distributions of the
scaled distance, position, speed and ray time obs, to choose their scales.

Usage: python scales.py scales.tsv [--knees 5,10,20,30,50,85]

Per feature (slot features like near_wall0..3 are also merged into one
row): the share of the policy's live steps it's present, the share clipped
(linear scaling, a component at +-1) or saturated (tanh, above 0.995) at the
current scale, raw value quantiles, the obs mean and std at the current
scale, the std a tanh squash with each candidate knee would give, and the
knee with the largest std, which spreads the common range the most.
"""
import argparse
import math
import re
from collections import OrderedDict


def load(path):
    rows = []
    with open(path) as f:
        header = f.readline().rstrip('\n').split('\t')
        edges = [float(x) for x in header[5:]]
        for line in f:
            p = line.rstrip('\n').split('\t')
            rows.append({'name': p[0], 'kind': p[1], 'scale': float(p[2]), 'n': float(p[3]), 'clipped': float(p[4]),
                         'hist': [float(x) for x in p[5:]]})
    return edges, rows


def mids(edges):
    # bin 0 is everything below the first edge
    out = [edges[0] / 2]
    for k in range(1, len(edges)):
        out.append(math.sqrt(edges[k - 1] * edges[k]))
    return out


def quantile(hist, edges, q):
    n = sum(hist)
    acc = 0
    for k, c in enumerate(hist):
        acc += c
        if acc >= q * n:
            return edges[k]
    return edges[-1]


def moments(hist, values, f):
    n = sum(hist)
    if n == 0:
        return float('nan'), float('nan')
    m = sum(c * f(v) for c, v in zip(hist, values)) / n
    var = sum(c * (f(v) - m) ** 2 for c, v in zip(hist, values)) / n
    return m, math.sqrt(max(var, 0.0))


def fmt(x):
    if x != x:
        return '-'
    if x >= 100:
        return f'{x:.0f}'
    if x >= 10:
        return f'{x:.1f}'
    return f'{x:.2f}'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('scales')
    parser.add_argument('--knees', default='0.25,0.5,1,2,5,8,10,15,20,25,30,40,50,60,80,100,150')
    args = parser.parse_args()
    knees = [float(k) for k in args.knees.split(',')]
    edges, rows = load(args.scales)
    values = mids(edges)
    steps = max(r['n'] for r in rows if r['name'].startswith('own_vel')) if any(r['name'] == 'own_vel' for r in rows) else 1

    # merged slot groups first, then each row
    groups = OrderedDict()
    for r in rows:
        g = re.sub(r'\d+$', '', r['name'])
        if g != r['name']:
            if g not in groups:
                groups[g] = dict(r, name=g + '*', n=0, clipped=0, hist=[0.0] * len(r['hist']))
            groups[g]['n'] += r['n']
            groups[g]['clipped'] += r['clipped']
            groups[g]['hist'] = [a + b for a, b in zip(groups[g]['hist'], r['hist'])]
    table = list(groups.values()) + [r for r in rows if r['name'].startswith(('near_wall0', 'pickup0', 'proj_pos0', 'own_proj_pos0', 'enemy_proj_pos0'))
                                     or not re.search(r'\d+$', r['name'])]
    table = [r for r in table if r['n'] > 0]

    print('raw values: units, or seconds for ray times; present = share of live steps (slot groups: slots per step)')
    print(f'{"feature":<20} {"scale":>11} {"present":>7} {"clip%":>6} {"p10":>6} {"p25":>6} {"p50":>6} {"p75":>6} {"p90":>6} {"p99":>6}'
          f' {"obs std":>7} {"best knee":>9} {"std then":>8}')
    for r in table:
        if r['kind'] == 'tanh':
            cur = lambda v, s=r['scale']: math.tanh(v / s)
        else:
            cur = lambda v, s=r['scale']: min(v / s, 1.0)
        _, std = moments(r['hist'], values, cur)
        best = max(knees, key=lambda k: moments(r['hist'], values, lambda v: math.tanh(v / k))[1])
        _, bstd = moments(r['hist'], values, lambda v: math.tanh(v / best))
        q = [quantile(r['hist'], edges, x) for x in (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)]
        scale = f'{r["kind"]} {fmt(r["scale"])}'
        print(f'{r["name"]:<20} {scale:>11} {r["n"] / steps:>7.2f} {100 * r["clipped"] / r["n"]:>6.1f} '
              + ' '.join(f'{fmt(v):>6}' for v in q) + f' {std:>7.3f} {fmt(best):>9} {bstd:>8.3f}')

    print('\nobs std of a tanh squash by knee (the largest per row is the best knee above)')
    print(f'{"feature":<20} ' + ' '.join(f'{fmt(k):>5}' for k in knees))
    for r in table:
        print(f'{r["name"]:<20} ' + ' '.join(f'{moments(r["hist"], values, lambda v: math.tanh(v / k))[1]:>5.2f}' for k in knees))


if __name__ == '__main__':
    main()
