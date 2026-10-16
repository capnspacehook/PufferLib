"""Summarizes and compares iw_analyze results.

Usage: python compare.py [--side p|b] [--sections a,b] [name=]games.jsonl ...

The first file is the baseline. Score tables show each run's mean score with
a 95% CI, and for later runs the paired difference to the baseline over the
games both played (same opponent, map, seed and seat). Other tables show per
game or per event rates side by side.

Opponents are bot personalities or, for runs made with --opponent, the
opponent policy's name. --side b summarizes the opponent instead of the
policy. Sections: opp, map, grid,
weapons, boost, deaths, behavior; all by default.
"""
import argparse
import json
import math
import os
from collections import defaultdict

MAPS = ['boring', 'prototypeArena', 'snipers', 'rooms', 'xArena', 'crossBounce', 'asteriskArena', 'foamPit',
        'siege', 'pinball', 'moat', 'hourglass', 'jaws', 'cavern', 'crossroads']
WEAPONS = ['standard', 'machinegun', 'sniper', 'shotgun', 'imploder', 'accelerator', 'flak', 'mine', 'blackhole', 'nuke']
SOURCES = ['invalid', 'movement', 'recoil', 'burst', 'projectile', 'explosion', 'drone']
OPPS = ['scripted', 'evader', 'camper', 'rusher', 'trapper', 'turtle', 'fundamentals']
SECTIONS = ['opp', 'map', 'grid', 'weapons', 'boost', 'deaths', 'behavior']


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def mean_ci(xs):
    n = len(xs)
    if n == 0:
        return float('nan'), float('nan')
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / max(n - 1, 1)
    return m, 1.96 * math.sqrt(var / n)


def ratio(num, den):
    return num / den if den else float('nan')


def total(games, side, key, idx=None):
    if idx is None:
        return sum(g[side][key] for g in games)
    return sum(g[side][key][idx] for g in games)


def key(g):
    return (g['opp'], g['map'], g['seed'], g.get('seat', 0))


def opponents(games):
    """Opponents in the games: bot personalities in their usual order, then policies."""
    present = {g['opp'] for g in games}
    return [o for o in OPPS if o in present] + sorted(present - set(OPPS))


def table(title, header, rows):
    widths = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
    print(f'\n{title}')
    for j, r in enumerate([header] + rows):
        print('  ' + '  '.join(str(c).ljust(widths[i]) if i == 0 else str(c).rjust(widths[i]) for i, c in enumerate(r)))
        if j == 0:
            print('  ' + '  '.join('-' * w for w in widths))


def fmt(x, digits=3):
    return 'nan' if x != x else f'{x:.{digits}f}'


def score_rows(runs, side, groups):
    """Rows of score +- CI for each group, with paired diffs to the baseline."""
    base = {key(g): g for g in runs[0][1]}
    rows = []
    for label, pred in groups:
        row = [label]
        for i, (_, games) in enumerate(runs):
            gs = [g for g in games if pred(g)]
            if not gs:
                row += ['-', '-'] if i > 0 else ['-']
                continue
            m, ci = mean_ci([g[side]['score'] for g in gs])
            row.append(f'{fmt(m)}±{fmt(ci)}')
            if i > 0:
                diffs = [g[side]['score'] - base[key(g)][side]['score'] for g in gs if key(g) in base]
                dm, dci = mean_ci(diffs)
                row.append(f'{dm:+.3f}±{fmt(dci)}' if diffs else '-')
        rows.append(row)
    return rows


def score_header(runs, first):
    header = [first]
    for i, (name, _) in enumerate(runs):
        header.append(name)
        if i > 0:
            header.append(f'Δ {name}')
    return header


def section_opp(runs, side):
    opps = opponents([g for _, gs in runs for g in gs])
    groups = [('all', lambda g: True)] + [(o, lambda g, o=o: g['opp'] == o) for o in opps]
    table('score by opponent', score_header(runs, 'opponent'), score_rows(runs, side, groups))
    rows = []
    for name, games in runs:
        for o in ['all'] + opps:
            gs = [g for g in games if o == 'all' or g['opp'] == o]
            n = len(gs)
            rows.append([f'{name} {o}', fmt(sum(g[side]['score'] == 1 for g in gs) / n),
                         fmt(sum(g[side]['score'] == 0.5 for g in gs) / n), fmt(sum(g[side]['score'] == 0 for g in gs) / n),
                         f"{sum(g['length'] for g in gs) / n:.0f}"])
    table('win / draw / loss', ['run opponent', 'win', 'draw', 'loss', 'length'], rows)


def section_map(runs, side):
    maps = sorted({g['map'] for _, gs in runs for g in gs})
    groups = [(MAPS[m], lambda g, m=m: g['map'] == m) for m in maps]
    table('score by map', score_header(runs, 'map'), score_rows(runs, side, groups))
    rows = []
    for m in maps:
        row = [MAPS[m]]
        for _, games in runs:
            gs = [g for g in games if g['map'] == m]
            n = len(gs)
            row += [fmt(sum(g[side]['score'] == 0.5 for g in gs) / n, 2),
                    fmt(total(gs, side, 'selfKills') / n, 2), fmt(total(gs, side, 'floatingWallSelfKills') / n, 2)]
        rows.append(row)
    header = ['map'] + [f'{name} {c}' for name, _ in runs for c in ('draw', 'self/g', 'floatSelf/g')]
    table('draws and self-kills by map', header, rows)


def section_grid(runs, side):
    for name, games in runs:
        maps = sorted({g['map'] for g in games})
        opps = opponents(games)
        rows = []
        for m in maps:
            row = [MAPS[m]]
            for o in opps:
                gs = [g for g in games if g['map'] == m and g['opp'] == o]
                row.append(fmt(sum(g[side]['score'] for g in gs) / len(gs), 2) if gs else '-')
            rows.append(row)
        table(f'{name}: score by map and opponent', ['map'] + [o[:6] for o in opps], rows)


def weapon_metrics(games, side):
    n = len(games)
    alive = total(games, side, 'alive')
    out = {}
    for w, name in enumerate(WEAPONS):
        fired = total(games, side, 'shotsFired', w)
        picked = total(games, side, 'weaponsPickedUp', w)
        kills = total(games, side, 'killsByWeapon', w)
        self_kills = total(games, side, 'selfKillsByWeapon', w)
        out[name] = {
            'held%': 100 * ratio(total(games, side, 'held', w), alive),
            'picked/g': picked / n,
            'fired/g': fired / n,
            'acc': ratio(total(games, side, 'shotsHit', w), fired),
            'kills/g': kills / n,
            'self/g': self_kills / n,
            'net/pickup': ratio(kills - self_kills, picked),
            'boosted%': 100 * ratio(total(games, side, 'boostedShotsFired', w), fired),
            'boostKill%': 100 * ratio(total(games, side, 'boostedKills', w), kills),
        }
    return out


def section_weapons(runs, side):
    metrics = [(name, weapon_metrics(games, side)) for name, games in runs]
    for m, digits in [('kills/g', 3), ('net/pickup', 2), ('picked/g', 2), ('held%', 1), ('acc', 3), ('self/g', 3)]:
        rows = [[w] + [fmt(wm[w][m], digits) for _, wm in metrics] for w in WEAPONS]
        table(f'weapons: {m}', ['weapon'] + [name for name, _ in runs], rows)
    # pickups by weapon held before, for the baseline
    name, games = runs[0]
    n = len(games)
    rows = []
    for a, held in enumerate(WEAPONS):
        rows.append([held] + [fmt(sum(g[side]['pickupsFrom'][a * 10 + b] for g in games) / n, 2) for b in range(10)])
    table(f'{name}: pickups per game by weapon held (rows) and picked up (columns)', ['held'] + [w[:5] for w in WEAPONS], rows)


def section_boost(runs, side):
    rows = []
    for name, games in runs:
        n = len(games)
        kills = sum(total(games, side, 'killsByWeapon', w) for w in range(10))
        boosted_kills = sum(total(games, side, 'boostedKills', w) for w in range(10))
        rows.append([name, fmt(total(games, side, 'totalBursts') / n, 2),
                     fmt(sum(total(games, side, 'boostedShotsFired', w) for w in range(10)) / n, 2),
                     fmt(100 * ratio(boosted_kills, kills), 1),
                     fmt(sum(total(games, side, 'boostedSelfKills', w) for w in range(10)) / n, 3)])
    table('burst boosting', ['run', 'bursts/g', 'boosted shots/g', 'boosted kill%', 'boosted self/g'], rows)
    for m in ['boosted%', 'boostKill%']:
        metrics = [(name, weapon_metrics(games, side)) for name, games in runs]
        rows = [[w] + [fmt(wm[w][m], 1) for _, wm in metrics] for w in WEAPONS]
        table(f'boost: {m} by weapon', ['weapon'] + [name for name, _ in runs], rows)
    rows = []
    for w, wname in enumerate(WEAPONS):
        row = [wname]
        for _, games in runs:
            fired = total(games, side, 'boostedShotsFired', w)
            row += [fmt(ratio(total(games, side, 'boostedShotsHit', w), fired)),
                    fmt(ratio(total(games, side, 'boostedSpeedAfter', w), total(games, side, 'boostedSpeedBefore', w)), 2)]
        rows.append(row)
    header = ['weapon'] + [f'{name} {c}' for name, _ in runs for c in ('boosted acc', 'speed x')]
    table('boost: hit rate of boosted shots and speed multiplier', header, rows)


def section_deaths(runs, side):
    rows = []
    for name, games in runs:
        n = len(games)
        rows.append([name, fmt(total(games, side, 'kills') / n, 2), fmt(total(games, side, 'deaths') / n, 2),
                     fmt(total(games, side, 'selfKills') / n, 2), fmt(total(games, side, 'floatingWallSelfKills') / n, 3),
                     fmt(total(games, side, 'floatingWallDeaths') / n, 3), fmt(total(games, side, 'unknownKills') / n, 3)])
    table('kills and deaths per game', ['run', 'kills', 'deaths', 'self', 'floatSelf', 'floatDeaths', 'unknown'], rows)
    rows = []
    for t, src in enumerate(SOURCES[1:], 1):
        row = [src]
        for _, games in runs:
            n = len(games)
            row += [fmt(total(games, side, 'deathsBySource', t) / n), fmt(total(games, side, 'selfKillsBySource', t) / n)]
        rows.append(row)
    header = ['source'] + [f'{name} {c}' for name, _ in runs for c in ('deaths/g', 'self/g')]
    table('deaths and self-kills by source', header, rows)


def behavior_metrics(games, side):
    s = lambda k: total(games, side, k)
    alive = s('alive')
    fire = s('fireSteps')
    first_los = [g[side]['firstLosStep'] for g in games if g[side]['firstLosStep'] >= 0]
    first_pick = [g[side]['firstPickupStep'] for g in games if g[side]['firstPickupStep'] >= 0]
    shots = sum(total(games, side, 'shotsFired', w) for w in range(10))
    hits = sum(total(games, side, 'shotsHit', w) for w in range(10))
    return [
        ('shots/g', shots / len(games), 1),
        ('accuracy', ratio(hits, shots), 3),
        ('trigger held', ratio(s('trigger'), alive), 3),
        ('trigger held, LoS', ratio(s('triggerLos'), s('losSteps')), 3),
        ('trigger held, no LoS', ratio(s('triggerNoLos'), s('noLosSteps')), 3),
        ('shots without LoS', ratio(s('fireNoLos'), fire), 3),
        ('shots aimed within 10°', ratio(s('fireAim10'), fire), 3),
        ('charging a weapon', ratio(s('charging'), alive), 3),
        ('holding a full charge', ratio(s('chargeHeldFull'), alive), 3),
        ('brake', ratio(s('brake'), alive), 3),
        ('burst charging', ratio(s('burstCharge'), alive), 3),
        ('burst hit rate', ratio(s('burstsHit'), s('totalBursts')), 3),
        ('energy', ratio(s('energy'), alive), 3),
        ('energy emptied/g', s('energyEmptied') / len(games), 3),
        ('speed', ratio(s('speed'), alive), 2),
        ('speed, LoS', ratio(s('speedLos'), s('losSteps')), 2),
        ('speed, no LoS', ratio(s('speedNoLos'), s('noLosSteps')), 2),
        ('LoS to enemy', ratio(s('losSteps'), s('sepSteps')), 3),
        ('separation', ratio(s('sepSum'), s('sepSteps')), 1),
        ('within 3 of a death wall', ratio(s('nearDeath3'), alive), 3),
        ('within 6 of a floating death wall', ratio(s('nearFloatingDeath6'), alive), 3),
        ('first LoS (steps)', sum(first_los) / max(len(first_los), 1), 1),
        ('first pickup (steps)', sum(first_pick) / max(len(first_pick), 1), 1),
        ('pickups/g', sum(total(games, side, 'weaponsPickedUp', w) for w in range(10)) / len(games), 2),
        ('shields broken/g', s('shieldsBroken') / len(games), 3),
        ('own shield broken/g', s('ownShieldBroken') / len(games), 3),
    ]


def aim_metrics(games):
    hist = [sum(g['aimHist'][i] for g in games) for i in range(len(games[0]['aimHist']))]
    tot = sum(hist)
    p = sorted((x / tot for x in hist), reverse=True)
    perplexity = math.exp(-sum(x * math.log(x) for x in p if x > 0))
    acc, used95 = 0.0, len(p)
    for i, x in enumerate(p):
        acc += x
        if acc >= 0.95:
            used95 = i + 1
            break
    return [('aim perplexity', perplexity, 1), ('aim dirs for 95%', used95, 0)]


def section_behavior(runs, side):
    cols = [(name, behavior_metrics(games, side) + (aim_metrics(games) if side == 'p' else [])) for name, games in runs]
    # the other side of the baseline for reference
    other = 'b' if side == 'p' else 'p'
    ref = behavior_metrics(runs[0][1], other)
    ref_name = 'policy' if other == 'p' else 'opponent'
    rows = []
    for i, (label, _, digits) in enumerate(cols[0][1]):
        row = [label] + [fmt(c[i][1], digits) for _, c in cols]
        row.append(fmt(ref[i][1], ref[i][2]) if i < len(ref) else '-')
        rows.append(row)
    table('behavior', ['metric'] + [name for name, _ in runs] + [f'{runs[0][0]} {ref_name}'], rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('runs', nargs='+', help='[name=]games.jsonl, the first is the baseline')
    parser.add_argument('--side', choices=['p', 'b'], default='p', help='p: the policy, b: the bot')
    parser.add_argument('--sections', default=','.join(SECTIONS), help=f'comma separated subset of {",".join(SECTIONS)}')
    args = parser.parse_args()
    runs = []
    for spec in args.runs:
        name, _, path = spec.rpartition('=')
        runs.append((name or os.path.splitext(os.path.basename(path))[0], load(path)))
    for section in args.sections.split(','):
        globals()[f'section_{section}'](runs, args.side)


if __name__ == '__main__':
    main()
