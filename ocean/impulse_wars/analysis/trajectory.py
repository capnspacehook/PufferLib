"""Shows how behavior changes over a run's checkpoints.

Usage: python trajectory.py [--table all|behavior|kills|held] [--csv] dir_or_files ...

Takes iw_analyze self-play results, one file per checkpoint, named by the
checkpoint's step count as analyze_run.sh writes them
(traj/<steps>.jsonl), or a directory of them, and prints a row per
checkpoint. Both drones are counted, as both are the same policy.

Tables: behavior (kills, self-kills, swaps, context use, charging,
holstered charges, setup kills, line of sight, firing without it, crates,
standoffs, assists), kills (kills per game by weapon) and held (% of live
steps with each weapon in hand). --csv prints every column of every table
as one CSV instead.
"""
import argparse
import glob
import os

from compare import WEAPON_SETS, load, ratio


def steps_of(path):
    name = os.path.splitext(os.path.basename(path))[0]
    try:
        return int(name)
    except ValueError:
        return None


def checkpoint_files(specs):
    files = []
    for spec in specs:
        if os.path.isdir(spec):
            files += glob.glob(os.path.join(spec, '*.jsonl'))
        else:
            files.append(spec)
    files = [f for f in files if steps_of(f) is not None]
    return sorted(files, key=steps_of)


def summarize(games):
    ds = [g[s] for g in games for s in ('p', 'b')]
    n = len(ds)
    t = lambda k: sum(d.get(k, 0) for d in ds)
    arr = lambda k, i: sum(d[k][i] for d in ds)
    nw = len(ds[0]['shotsFired'])
    weapons = WEAPON_SETS[nw]
    alive = t('alive')
    kills = t('kills')
    row = {
        'kills/g': kills / n,
        'self/g': t('selfKills') / n,
        'assists/g': t('assists') / n,
        'swaps/g': t('swaps') / n,
        'ctx held': ratio(sum(sum(d.get('contextPressed', [0])) for d in ds), alive),
        'ctx press/g': sum(sum(d['contextPresses']) for d in ds) / n if 'contextPresses' in ds[0] else float('nan'),
        'charging': ratio(t('charging'), alive),
        'hol chg%': 100 * ratio(t('holsterCharged'), alive),
        'setup%': 100 * ratio(t('setupKillsAny'), kills),
        'LoS': ratio(t('losSteps'), t('sepSteps')),
        'fire noLoS': ratio(t('fireNoLos'), t('fireSteps')),
        'crates/g': t('cratesPickedUp') / n,
    }
    if 'sitSteps' in ds[0]:
        both = sum(sum(d['sitSteps']) for d in ds)
        row['standoff'] = ratio(t('standoffSteps'), both)
    kill_row = {}
    held_row = {}
    for w, name in enumerate(weapons):
        if name == 'unset':
            continue
        kill_row[name[:6]] = arr('killsByWeapon', w) / n
        held_row[name[:6]] = 100 * ratio(arr('held', w), alive)
    return row, kill_row, held_row


def print_table(title, rows, digits):
    cols = list(rows[0][1].keys())
    print(f'\n{title}')
    header = ['steps'] + cols
    widths = [max(len(h), 6) for h in header]
    print('  ' + '  '.join(h.rjust(w) for h, w in zip(header, widths)))
    print('  ' + '  '.join('-' * w for w in widths))
    for steps, r in rows:
        cells = [f'{steps / 1e9:.2f}B'] + [f'{r[c]:.{digits}f}' if r[c] == r[c] else 'nan' for c in cols]
        print('  ' + '  '.join(c.rjust(w) for c, w in zip(cells, widths)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('files', nargs='+', help='per checkpoint results named <steps>.jsonl, or directories of them')
    parser.add_argument('--table', default='all', choices=['all', 'behavior', 'kills', 'held'])
    parser.add_argument('--csv', action='store_true', help='print every column as one CSV')
    args = parser.parse_args()

    files = checkpoint_files(args.files)
    if not files:
        parser.error('no files named <steps>.jsonl found')
    rows = [(steps_of(f), summarize(load(f))) for f in files]

    if args.csv:
        cols = [(i, k) for i, prefix in enumerate(['', 'kills ', 'held% ']) for k in rows[0][1][i]]
        print(','.join(['steps'] + [['', 'kills ', 'held% '][i] + k for i, k in cols]))
        for steps, parts in rows:
            print(','.join([str(steps)] + [f'{parts[i][k]:.4f}' for i, k in cols]))
        return
    if args.table in ('all', 'behavior'):
        print_table('behavior', [(s, p[0]) for s, p in rows], 3)
    if args.table in ('all', 'kills'):
        print_table('kills per game by weapon', [(s, p[1]) for s, p in rows], 3)
    if args.table in ('all', 'held'):
        print_table('% of live steps with each weapon in hand', [(s, p[2]) for s, p in rows], 1)


if __name__ == '__main__':
    main()
