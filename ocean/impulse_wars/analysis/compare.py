"""Summarizes and compares iw_analyze results.

Usage: python compare.py [--side p|b] [--sections a,b] [name=]games.jsonl ...

The first file is the baseline. Score tables show each run's mean score with
a 95% CI, and for later runs the paired difference to the baseline over the
games both played (same opponent, map, seed and seat). Other tables show per
game or per event rates side by side.

Opponents are bot personalities or, for runs made with --opponent, the
opponent policy's name. --side b summarizes the opponent instead of the
policy. Sections: opp, map, grid, weapons, boost, deaths, behavior, slots,
utilities, pace, mapweapons, standoff, shadow; all by default. Combo
tournament runs (iw_analyze --combos, weapon + utility loadouts from the
utility slot on) get a combos section instead of opp, map and grid;
--sections combos shows only it.

slots: weapon pickups and combo and setup kills, or the weapon slot stats
for results from before the utility slot. utilities: use, kills and the
grapple's and mines' own stats per utility.

pace: game length, draws, line of sight, firing without it and crate
refills, which together show passive play (e.g. combo drills farming
crates). mapweapons: kill share of each weapon by map, utilities as their
projectiles, with each map's featured weapon or utility marked. standoff: time in each situation (line of sight,
blocked by a wall close or far) by who's ahead on lives, and what happens
when a drone ends a standoff by peeking.
"""
import argparse
import json
import math
import os
from collections import defaultdict

MAPS = ['boring', 'prototypeArena', 'snipers', 'rooms', 'xArena', 'crossBounce', 'asteriskArena', 'foamPit',
        'siege', 'pinball', 'moat', 'hourglass', 'jaws', 'cavern', 'crossroads']
# weapon names by weapon count, as runs of older env builds have fewer
WEAPON_SETS = {
    10: ['standard', 'machinegun', 'sniper', 'shotgun', 'imploder', 'accelerator', 'flak', 'mine', 'blackhole', 'nuke'],
    11: ['standard', 'machinegun', 'sniper', 'shotgun', 'imploder', 'accelerator', 'flak', 'mine', 'blackhole', 'nuke',
         'grapple'],
    12: ['unset', 'standard', 'machinegun', 'sniper', 'shotgun', 'imploder', 'accelerator', 'flak', 'mine', 'blackhole',
         'nuke', 'grapple'],
    # from the utility slot on the grapple and mine launcher are utility
    # projectiles, ordered after the weapons
    'utilities': ['unset', 'standard', 'machinegun', 'sniper', 'shotgun', 'imploder', 'accelerator', 'flak',
                  'blackhole', 'nuke', 'grapple', 'mine'],
}
# weapons whose projectiles are only fired by a utility, as weaponIsUtility
# in settings.h
UTILITY_PROJECTILES = ('grapple', 'mine')
WEAPONS = WEAPON_SETS['utilities']


def weapon_names(drone):
    """Weapon names in the order a result's per weapon arrays use."""
    if 'utilitiesPickedUp' in drone:
        return WEAPON_SETS['utilities']
    return WEAPON_SETS[len(drone['shotsFired'])]
# action heads by how many there are; 8 is from before the utility slot
ATNS = {
    7: ['move', 'aim coarse', 'aim fine', 'trigger', 'utility', 'brake', 'burst'],
    8: ['move', 'aim coarse', 'aim fine', 'trigger', 'brake', 'burst', 'swap', 'context'],
}


def atn_names(n):
    return ATNS.get(n, [f'head {i}' for i in range(n)])
SOURCES = ['invalid', 'movement', 'recoil', 'burst', 'projectile', 'explosion', 'grapple', 'drone']
# from before grapples were their own physics source
SOURCES_NO_GRAPPLE = ['invalid', 'movement', 'recoil', 'burst', 'projectile', 'explosion', 'drone']
# utilities by type, and the weapons their projectiles are
UTILITIES = ['none', 'grapple', 'mines']
UTILITY_WEAPONS = {1: 'grapple', 2: 'mine'}
GRAPPLE_TARGETS = ['wall', 'floating wall', 'drone']
OPPS = ['scripted', 'evader', 'camper', 'rusher', 'trapper', 'turtle', 'fundamentals']
SECTIONS = ['opp', 'map', 'grid', 'weapons', 'boost', 'deaths', 'behavior', 'slots', 'utilities', 'pace', 'mapweapons',
            'standoff', 'projectiles', 'shadow']
# each map's featured weapon (its spawn weight override in map.h), marked in
# the mapweapons section; only applies to builds from fe03156a on. From the
# utility slot on siege and moat feature the grapple and mines utilities,
# which kill as the weapons their projectiles are
FEATURED = {'prototypeArena': 'machinegun', 'snipers': 'sniper', 'rooms': 'shotgun', 'xArena': 'nuke',
            'foamPit': 'imploder', 'siege': 'grapple', 'pinball': 'accelerator', 'moat': 'mine',
            'hourglass': 'blackhole', 'crossroads': 'flak'}
SITUATIONS = ['LoS', 'blocked close', 'blocked far']
LEADS = ['behind', 'even', 'ahead']


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
        if name == 'unset':
            continue
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
            'assists/g': sum(g[side].get('assistsByWeapon', [0] * (w + 1))[w] for g in games) / n,
            'boosted%': 100 * ratio(total(games, side, 'boostedShotsFired', w), fired),
            'boostKill%': 100 * ratio(total(games, side, 'boostedKills', w), kills),
        }
    return out


def section_weapons(runs, side):
    metrics = [(name, weapon_metrics(games, side)) for name, games in runs]
    for m, digits in [('kills/g', 3), ('assists/g', 3), ('net/pickup', 2), ('picked/g', 2), ('held%', 1), ('acc', 3), ('self/g', 3)]:
        rows = [[w] + [fmt(wm[w][m], digits) for _, wm in metrics] for w in WEAPONS if w in metrics[0][1]]
        table(f'weapons: {m}', ['weapon'] + [name for name, _ in runs], rows)
    # pickups by weapon held before, for the baseline
    name, games = runs[0]
    n = len(games)
    nw = len(WEAPONS)
    used = [w for w in range(nw) if WEAPONS[w] != 'unset']
    rows = []
    for a in used:
        rows.append([WEAPONS[a]] + [fmt(sum(g[side]['pickupsFrom'][a * nw + b] for g in games) / n, 2) for b in used])
    table(f'{name}: pickups per game by weapon in hand (rows) and picked up (columns)', ['held'] + [WEAPONS[w][:5] for w in used], rows)


def section_boost(runs, side):
    rows = []
    for name, games in runs:
        n = len(games)
        kills = sum(sum(g[side]['killsByWeapon']) for g in games)
        boosted_kills = sum(sum(g[side]['boostedKills']) for g in games)
        rows.append([name, fmt(total(games, side, 'totalBursts') / n, 2),
                     fmt(sum(sum(g[side]['boostedShotsFired']) for g in games) / n, 2),
                     fmt(100 * ratio(boosted_kills, kills), 1),
                     fmt(sum(sum(g[side]['boostedSelfKills']) for g in games) / n, 3)])
    table('burst boosting', ['run', 'bursts/g', 'boosted shots/g', 'boosted kill%', 'boosted self/g'], rows)
    for m in ['boosted%', 'boostKill%']:
        metrics = [(name, weapon_metrics(games, side)) for name, games in runs]
        rows = [[w] + [fmt(wm[w][m], 1) for _, wm in metrics] for w in WEAPONS if w in metrics[0][1]]
        table(f'boost: {m} by weapon', ['weapon'] + [name for name, _ in runs], rows)
    rows = []
    for w, wname in enumerate(WEAPONS):
        if wname == 'unset':
            continue
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
                     fmt(total(games, side, 'floatingWallDeaths') / n, 3), fmt(total(games, side, 'unknownKills') / n, 3),
                     fmt(sum(g[side].get('assists', 0) for g in games) / n, 3)])
    table('kills and deaths per game', ['run', 'kills', 'deaths', 'self', 'floatSelf', 'floatDeaths', 'unknown', 'assists'], rows)
    rows = []
    sources = SOURCES if len(games[0][side]['deathsBySource']) == len(SOURCES) else SOURCES_NO_GRAPPLE
    for t, src in enumerate(sources[1:], 1):
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
    shots = sum(sum(g[side]['shotsFired']) for g in games)
    hits = sum(sum(g[side]['shotsHit']) for g in games)
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
        ('pickups/g', sum(sum(g[side]['weaponsPickedUp']) for g in games) / len(games), 2),
        ('ammo crates/g', s('cratesPickedUp') / len(games) if 'cratesPickedUp' in games[0][side] else float('nan'), 2),
        ('ammo crates that refilled', ratio(s('cratesRefilled'), s('cratesPickedUp')) if 'cratesPickedUp' in games[0][side] else float('nan'), 3),
        ('ammo at crate pickup', ratio(s('crateAmmoFrac'), s('crateAmmoN')) if 'crateAmmoN' in games[0][side] else float('nan'), 3),
        ('ammo, all steps with a pickup weapon', ratio(s('ammoFrac'), s('ammoSteps')) if 'ammoSteps' in games[0][side] else float('nan'), 3),
        ('crates with no pickup weapon', ratio(s('crateNoWeapons'), s('cratesPickedUp')) if 'crateNoWeapons' in games[0][side] else float('nan'), 3),
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


def slot_metrics(games, side):
    n = len(games)
    s = lambda k: sum(g[side].get(k, 0) for g in games)
    alive = s('alive')
    kills = sum(sum(g[side]['killsByWeapon']) for g in games)
    pickups = s('pickupTopUp') + s('pickupToHand') + s('pickupToHolster') + s('pickupReplace')
    ctx = sum(sum(g[side].get('contextPressed', [])) for g in games)
    ctx_useful = sum(sum(g[side].get('contextUseful', [])) for g in games)
    presses = sum(sum(g[side].get('contextPresses', [])) for g in games)
    presses_useful = sum(sum(g[side].get('contextPressesUseful', [])) for g in games)
    return [
        ('swaps/g', s('swaps') / n, 2),
        ('swap pressed', ratio(s('swapPressed'), alive), 3),
        ('auto switches/g', s('autoSwitches') / n, 2),
        ('holding 2 pickup weapons', ratio(s('bothSlots'), alive), 3),
        ('only the standard gun', ratio(s('standardOnly'), alive), 3),
        ('empty weapon in hand', ratio(s('emptyHeld'), alive), 3),
        ('trigger suppressed', ratio(s('triggerSuppressed'), alive), 4),
        ('holstered charge', ratio(s('holsterCharged'), alive), 4),
        ('charged draws/g', s('chargedDraws') / n, 3),
        ('charged draws fired', ratio(s('chargedDrawFired'), s('chargedDraws')), 3),
        ('context held', ratio(ctx, alive), 3),
        ('context held steps useful', ratio(ctx_useful, ctx), 3),
        ('context presses/g', presses / n if 'contextPresses' in games[0][side] else float('nan'), 2),
        ('context presses useful (press semantics)', ratio(presses_useful, presses) if 'contextPresses' in games[0][side] else float('nan'), 3),
        ('pickups: top up', ratio(s('pickupTopUp'), pickups), 3),
        ('pickups: into hand', ratio(s('pickupToHand'), pickups), 3),
        ('pickups: into holster', ratio(s('pickupToHolster'), pickups), 3),
        ('pickups: replace', ratio(s('pickupReplace'), pickups), 3),
        ('replaced weapon had ammo', ratio(s('droppedWithAmmo'), s('pickupReplace')), 3),
        ('swapped before replacing', ratio(s('swapBeforeReplace'), s('pickupReplace')), 3),
        ('kills within 1s of a swap', ratio(s('killsAfterSwap'), kills), 3),
        ('kills with 2+ weapons in 3s', ratio(s('multiWeaponKills'), kills), 3),
        ('kills set up by another weapon hit in 2s', ratio(s('setupKillsAny'), kills), 3),
    ]


def pickup_metrics(games, side):
    n = len(games)
    s = lambda k: sum(g[side].get(k, 0) for g in games)
    alive = s('alive')
    kills = sum(sum(g[side]['killsByWeapon']) for g in games)
    pickups = s('pickupTopUp') + s('pickupToHand') + s('pickupReplace')
    return [
        ('auto switches/g', s('autoSwitches') / n, 2),
        ('only the standard gun', ratio(s('standardOnly'), alive), 3),
        ('pickups: top up', ratio(s('pickupTopUp'), pickups), 3),
        ('pickups: into hand', ratio(s('pickupToHand'), pickups), 3),
        ('pickups: replace', ratio(s('pickupReplace'), pickups), 3),
        ('kills with 2+ weapons in 3s', ratio(s('multiWeaponKills'), kills), 3),
        ('kills set up by another weapon hit in 2s', ratio(s('setupKillsAny'), kills), 3),
    ]


def combo_tables(runs, side, used):
    nw = len(WEAPONS)
    for name, games in runs:
        n = len(games)
        kills = [total(games, side, 'killsByWeapon', w) for w in range(nw)]
        combos = sorted(((sum(g[side]['comboKills'][k * nw + o] for g in games), k, o) for k in used for o in used if k != o),
                        reverse=True)
        rows = [[WEAPONS[k], WEAPONS[o], fmt(v / n, 3), fmt(100 * ratio(v, kills[k]), 1)] for v, k, o in combos[:20] if v > 0]
        table(f'{name}: kills with another weapon fired in the 3s before', ['killing weapon', 'also fired', 'per game',
                                                                            '% of its kills'], rows)
        if 'setupKills' not in games[0][side]:
            continue
        setups = sorted(((sum(g[side]['setupKills'][k * nw + o] for g in games), k, o) for k in used for o in used if k != o),
                        reverse=True)
        rows = [[WEAPONS[o], WEAPONS[k], fmt(v / n, 3), fmt(100 * ratio(v, kills[k]), 1)] for v, k, o in setups[:20] if v > 0]
        table(f'{name}: kills set up by another weapon hitting (or grappling) the enemy in the 2s before',
              ['setup weapon', 'killing weapon', 'per game', '% of its kills'], rows)


def section_slots(runs, side):
    if 'swaps' not in runs[0][1][0][side]:
        section_weapon_pickups(runs, side)
        return
    cols = [(name, slot_metrics(games, side)) for name, games in runs]
    other = 'b' if side == 'p' else 'p'
    ref = slot_metrics(runs[0][1], other)
    rows = []
    for i, (label, _, digits) in enumerate(cols[0][1]):
        rows.append([label] + [fmt(c[i][1], digits) for _, c in cols] + [fmt(ref[i][1], digits)])
    table('weapon slots', ['metric'] + [name for name, _ in runs] + [f'{runs[0][0]} {"opponent" if side == "p" else "policy"}'], rows)

    nw = len(WEAPONS)
    used = [w for w in range(nw) if WEAPONS[w] not in ('unset',)]
    for name, games in runs:
        n = len(games)
        alive = total(games, side, 'alive')
        rows = []
        for w in used:
            held = total(games, side, 'held', w)
            hol = total(games, side, 'heldHolster', w)
            press = total(games, side, 'contextPressed', w)
            rows.append([WEAPONS[w], fmt(100 * ratio(held, alive), 1), fmt(100 * ratio(hol, alive), 1),
                         fmt(100 * ratio(total(games, side, 'holsterChargedBy', w), alive), 2),
                         fmt(sum(g[side]['swapsFrom'][w * nw + b] for g in games for b in range(nw)) / n, 2),
                         fmt(sum(g[side]['swapsFrom'][a * nw + w] for g in games for a in range(nw)) / n, 2),
                         fmt(total(games, side, 'dropped', w) / n, 3),
                         fmt(total(games, side, 'diedHoldingInactive', w) / n, 3),
                         fmt(press / n, 2), fmt(ratio(total(games, side, 'contextUseful', w), press), 3)])
        table(f'{name}: slots by weapon', ['weapon', 'hand%', 'holster%', 'holster chg%', 'swapped out/g', 'swapped in/g',
                                           'dropped/g', 'died holstered/g', 'context/g', 'ctx useful'], rows)

        # swap pairs and combo kills, the most common first
        pairs = sorted(((sum(g[side]['swapsFrom'][a * nw + b] for g in games) / n, a, b) for a in used for b in used if a != b),
                       reverse=True)
        rows = [[f'{WEAPONS[a]} -> {WEAPONS[b]}', fmt(v, 3)] for v, a, b in pairs[:15] if v > 0]
        table(f'{name}: most common swaps (put away -> drawn), per game', ['swap', 'per game'], rows)
    combo_tables(runs, side, used)


# results from the utility slot on have one weapon slot
def section_weapon_pickups(runs, side):
    cols = [(name, pickup_metrics(games, side)) for name, games in runs]
    other = 'b' if side == 'p' else 'p'
    ref = pickup_metrics(runs[0][1], other)
    rows = []
    for i, (label, _, digits) in enumerate(cols[0][1]):
        rows.append([label] + [fmt(c[i][1], digits) for _, c in cols] + [fmt(ref[i][1], digits)])
    table('weapons and pickups', ['metric'] + [name for name, _ in runs] + [f'{runs[0][0]} {"opponent" if side == "p" else "policy"}'], rows)

    nw = len(WEAPONS)
    used = [w for w in range(nw) if WEAPONS[w] not in ('unset',)]
    for name, games in runs:
        n = len(games)
        alive = total(games, side, 'alive')
        rows = [[WEAPONS[w], fmt(100 * ratio(total(games, side, 'held', w), alive), 1),
                 fmt(total(games, side, 'dropped', w) / n, 3)] for w in used if WEAPONS[w] not in UTILITY_PROJECTILES]
        table(f'{name}: weapons in hand', ['weapon', 'hand%', 'dropped/g'], rows)
    combo_tables(runs, side, used)


def section_utilities(runs, side):
    if 'utilityPresses' not in runs[0][1][0][side]:
        return
    other = 'b' if side == 'p' else 'p'
    for name, games in runs:
        n = len(games)
        alive = total(games, side, 'alive')
        rows = []
        for u in range(1, len(UTILITIES)):
            w = WEAPONS.index(UTILITY_WEAPONS[u])
            held = total(games, side, 'utilityHeld', u)
            presses = total(games, side, 'utilityPresses', u)
            ammo = total(games, side, 'utilityAmmoSpent', u)
            kills = total(games, side, 'killsByWeapon', w)
            rows.append([UTILITIES[u], fmt(100 * ratio(held, alive), 1), fmt(100 * ratio(total(games, side, 'utilityReady', u), held), 1),
                         fmt(total(games, side, 'utilitiesPickedUp', u) / n, 2), fmt(presses / n, 2),
                         fmt(100 * ratio(total(games, side, 'utilityPressesUseful', u), presses), 1), fmt(ammo / n, 2),
                         fmt(total(games, side, 'utilityHits', u) / n, 2), fmt(kills / n, 3),
                         fmt(total(games, side, 'assistsByWeapon', w) / n, 3), fmt(total(games, side, 'selfKillsByWeapon', w) / n, 3),
                         fmt(ratio(kills, ammo), 3), fmt(total(games, other, 'killsByWeapon', w) / n, 3)])
        table(f'{name}: utilities (held% of live steps, ready% of held steps with ammo, useful% of presses that fired, '
              'reeled or detonated, hits on drones, kills/ammo; enemy k/g = kills on this side by it)',
              ['utility', 'held%', 'ready%', 'picked/g', 'press/g', 'useful%', 'ammo/g', 'hits/g', 'kills/g', 'assist/g',
               'self/g', 'kills/ammo', 'enemy k/g'], rows)

        grappleHeld = total(games, side, 'utilityHeld', 1)
        attached = total(games, side, 'grappleTethered') + total(games, side, 'grappleReeling')
        reeling = total(games, side, 'grappleReeling')
        rows = [[GRAPPLE_TARGETS[t], fmt(total(games, side, 'grappleAttaches', t) / n, 3)] for t in range(len(GRAPPLE_TARGETS))]
        rows += [['attached % of held', fmt(100 * ratio(attached, grappleHeld), 1)],
                 ['reeling % of attached', fmt(100 * ratio(reeling, attached), 1)],
                 ['reel speed / mean speed', fmt(ratio(ratio(total(games, side, 'grappleReelSpeed'), reeling),
                                                       ratio(total(games, side, 'speed'), alive)), 2)],
                 ['deaths attached/g', fmt(total(games, side, 'grappleAttachedDeaths') / n, 3)],
                 ['pulled into walls (kills)/g', fmt(total(games, side, 'killsByWeapon', WEAPONS.index('grapple')) / n, 3)]]
        table(f'{name}: grapple (attaches per game by target)', ['metric', 'value'], rows)
        mine = WEAPONS.index('mine')
        rows = [['detonations/g', fmt(total(games, side, 'minesDetonated') / n, 3)],
                ['kills/g', fmt(total(games, side, 'killsByWeapon', mine) / n, 3)],
                ['self kills/g', fmt(total(games, side, 'selfKillsByWeapon', mine) / n, 3)]]
        table(f'{name}: mines', ['metric', 'value'], rows)


def section_pace(runs, side):
    rows = []
    for name, games in runs:
        lengths = sorted(g['length'] for g in games)
        n = len(games)
        fire = total(games, side, 'fireSteps')
        crates = 'cratesRefilled' in games[0][side]
        rows.append([name, f'{lengths[n // 2]:.0f}', f'{lengths[int(n * 0.9)]:.0f}',
                     fmt(sum(g[side]['score'] == 0.5 for g in games) / n),
                     fmt(ratio(total(games, side, 'losSteps'), total(games, side, 'sepSteps'))),
                     fmt(ratio(total(games, side, 'fireNoLos'), fire)),
                     fmt(total(games, side, 'cratesRefilled') / n, 2) if crates else 'nan',
                     fmt(total(games, side, 'kills') / n, 2)])
    table('pace (high length, draws and firing without LoS with low LoS mean passive play)',
          ['run', 'len p50', 'len p90', 'draws', 'LoS', 'fire no LoS', 'crate refills/g', 'kills/g'], rows)


def section_mapweapons(runs, side):
    for name, games in runs:
        maps = sorted({g['map'] for g in games})
        used = [w for w in range(len(WEAPONS)) if WEAPONS[w] not in ('unset', 'standard')]
        rows = []
        for m in maps:
            gs = [g for g in games if g['map'] == m]
            kills = [total(gs, side, 'killsByWeapon', w) for w in range(len(WEAPONS))]
            t = sum(kills)
            featured = FEATURED.get(MAPS[m])
            cells = []
            for w in used:
                cell = fmt(100 * ratio(kills[w], t), 1)
                cells.append(cell + '*' if WEAPONS[w] == featured else cell)
            rows.append([MAPS[m], f"{sum(g['length'] for g in gs) / len(gs):.0f}"] + cells)
        table(f'{name}: % of kills by weapon on each map, utilities as their projectiles (grapple, mine); * = the map\'s '
              'featured weapon or utility',
              ['map', 'len'] + [WEAPONS[w][:6] for w in used], rows)


def section_standoff(runs, side):
    for name, games in runs:
        if 'sitSteps' not in games[0][side]:
            continue
        # both drones' views, so leads are symmetric
        ds = [g[s] for g in games for s in ('p', 'b')]
        steps = [sum(d['sitSteps'][k] for d in ds) for k in range(9)]
        both = sum(steps)
        rows = []
        for si, sit in enumerate(SITUATIONS):
            for li, lead in enumerate(LEADS):
                k = si * 3 + li
                rows.append([sit, lead, fmt(ratio(steps[k], both)),
                             fmt(ratio(sum(d['sitApproach'][k] for d in ds), steps[k]), 2),
                             fmt(ratio(sum(d['sitSpeed'][k] for d in ds), steps[k]), 2),
                             fmt(ratio(sum(d['sitTrigger'][k] for d in ds), steps[k]))])
        table(f'{name}: situations while both are alive by own lives vs the enemy\'s (approach = speed towards the enemy)',
              ['situation', 'lead', 'share', 'approach', 'speed', 'trigger'], rows)
        st = sum(d['standoffs'] for d in ds) / 2
        peeks = sum(d['peeks'] for d in ds)
        rows = [['standoffs/g (3s+ close with a wall between)', fmt(st / len(games), 2)],
                ['mean standoff length (s)', fmt(ratio(sum(d['standoffSteps'] for d in ds) / 2, st) / 10, 1)],
                ['share of both-alive time in standoffs', fmt(ratio(sum(d['standoffSteps'] for d in ds), both))],
                ['standoffs ended by a peek', fmt(ratio(peeks, st))],
                ['peeker kills within 3s', fmt(ratio(sum(d['peekKills'] for d in ds), peeks))],
                ['peeker dies within 3s', fmt(ratio(sum(d['peekDeaths'] for d in ds), peeks))]]
        table(f'{name}: standoffs', ['metric', 'value'], rows)


# bin widths of the projectile histograms, as iw_analyze writes them
PROJ_DIST_BIN = 5
PROJ_SPEED_BIN = 10
PROJ_K = [5, 8, 10, 15, 20, 30]
# weapons whose hits count as shots taken; explosions, grapples and black
# hole pulls don't
DIRECT_FIRE = ['standard', 'machinegun', 'sniper', 'shotgun', 'accelerator']


def hist_sum(games, side, key):
    h = None
    for g in games:
        v = g[side][key]
        h = list(v) if h is None else [a + b for a, b in zip(h, v)]
    return h


def hist_quantile(h, q):
    """the bin index the q quantile of a histogram falls in"""
    n = sum(h)
    acc = 0
    for i, c in enumerate(h):
        acc += c
        if acc >= q * n:
            return i
    return len(h) - 1


def hist_mean(h):
    return ratio(sum(i * c for i, c in enumerate(h)), sum(h))


def section_projectiles(runs, side):
    if 'projOwnHist' not in runs[0][1][0][side]:
        return
    rows = []
    for name, games in runs:
        for key, label in [('projOwnHist', 'own, all'), ('projOwnNearHist', 'own, within 30'),
                           ('projEnemyHist', 'enemy, all'), ('projEnemyNearHist', 'enemy, within 30'),
                           ('threatHist', 'enemy threats')]:
            h = hist_sum(games, side, key)
            last = len(h) - 1
            q = lambda x: (lambda b: f'{b}+' if b == last else str(b))(hist_quantile(h, x))
            rows.append([name, label, fmt(hist_mean(h), 2), q(0.5), q(0.9), q(0.95), q(0.99),
                         fmt(100 * ratio(h[0], sum(h)), 1)])
    table('live projectiles per step (counts; threats: enemy projectiles whose closest approach at the current relative '
          'velocity is within hitting or blast range within 2 s)',
          ['run', 'projectiles', 'mean', 'p50', 'p90', 'p95', 'p99', '% steps 0'], rows)

    for name, games in runs:
        rows = []
        for key, width, unit in [('projEnemyDist', PROJ_DIST_BIN, 'u'), ('threatDist', PROJ_DIST_BIN, 'u'),
                                 ('projEnemySpeed', PROJ_SPEED_BIN, 'u/s'), ('threatSpeed', PROJ_SPEED_BIN, 'u/s')]:
            h = hist_sum(games, side, key)
            last = len(h) - 1
            q = lambda x: (lambda b: f'{b * width}+' if b == last else f'<{(b + 1) * width}')(hist_quantile(h, x))
            rows.append([key, unit, q(0.1), q(0.25), q(0.5), q(0.75), q(0.9), q(0.99)])
        table(f'{name}: enemy projectiles and threats by distance and speed relative to the drone, quantiles',
              ['what', 'unit', 'p10', 'p25', 'p50', 'p75', 'p90', 'p99'], rows)
        h = hist_sum(games, side, 'threatTime')
        n = sum(h)
        cum = 0
        cells = []
        for k in range(len(h)):
            cum += h[k]
            if k + 1 in (1, 2, 3, 5, 10, 20):
                cells.append(fmt(100 * cum / n, 1) if n else '-')
        table(f'{name}: time until a threat\'s closest approach, % of threat steps within',
              ['1 step', '2', '3', '5', '10', '20 (2 s)'], [cells])

    rows = []
    for name, games in runs:
        n = sum(i * c for i, c in enumerate(hist_sum(games, side, 'threatHist')))
        out_all = hist_sum(games, side, 'threatOutAll')
        out_enemy = hist_sum(games, side, 'threatOutEnemy')
        rows.append([name, 'nearest of both drones'] + [fmt(100 * ratio(v, n), 2) for v in out_all])
        rows.append([name, 'nearest enemy projectiles'] + [fmt(100 * ratio(v, n), 2) for v in out_enemy])
    table('% of threats left out of the obs if it held the K nearest projectiles (the current obs: 30 of both drones)',
          ['run', 'slots hold'] + [f'K={k}' for k in PROJ_K], rows)

    for name, games in runs:
        ng = len(games)
        new = hist_sum(games, side, 'threatsNew')
        hits = hist_sum(games, side, 'hitsTaken')
        rows = []
        for w, wname in enumerate(WEAPONS):
            if wname == 'unset' or (new[w] == 0 and hits[w] == 0):
                continue
            direct = wname in DIRECT_FIRE
            rows.append([wname, fmt(new[w] / ng, 2), fmt(hits[w] / ng, 2) if direct else '-',
                         fmt(ratio(hits[w], new[w]), 2) if direct else '-'])
        table(f'{name}: enemy projectiles that became threats and hits taken per game, by weapon (hits per threat over 1 '
              'means many hit without being seen as a threat first, e.g. shotgun pellets hitting the step they spawn)',
              ['weapon', 'threats/g', 'hits/g', 'hits/threat'], rows)

    if 'projEnemyHistW' in runs[0][1][0][side]:
        for name, games in runs:
            hw = hist_sum(games, side, 'projEnemyHistW')
            nw_ = hist_sum(games, side, 'projEnemyNearHistW')
            tsw = hist_sum(games, side, 'threatStepsW')
            tow = hist_sum(games, side, 'threatOutEnemyW')
            nh = len(hw) // len(WEAPONS)
            nk = len(PROJ_K)
            alive = sum(sum(hw[w * nh:(w + 1) * nh]) for w in range(len(WEAPONS)))
            rows = []
            for w, wname in enumerate(WEAPONS):
                h = hw[w * nh:(w + 1) * nh]
                hn = nw_[w * nh:(w + 1) * nh]
                if sum(h) == 0:
                    continue
                q = lambda hh, x: (lambda b: f'{b}+' if b == nh - 1 else str(b))(hist_quantile(hh, x))
                out = [fmt(100 * ratio(tow[w * nk + k], tsw[w]), 2) for k in (1, 2, 3)]
                rows.append([wname, fmt(100 * sum(h) / alive, 1), q(h, 0.9), q(h, 0.99), q(hn, 0.9), q(hn, 0.99),
                             str(next((i for i in range(nh - 1, -1, -1) if h[i] > 0), 0))] + out)
            table(f'{name}: enemy projectiles per step by the weapon the enemy holds, and % of threats left out of the K '
                  'nearest enemy projectiles', ['enemy holds', '% steps', 'all p90', 'all p99', '<30 p90', '<30 p99', 'max']
                  + [f'out K={PROJ_K[k]}' for k in (1, 2, 3)], rows)

    rows = []
    for name, games in runs:
        alive = sum(g[side]['alive'] for g in games)
        rows.append([name, fmt(100 * ratio(sum(g[side]['ownBlastDangerSteps'] for g in games), alive), 2),
                     fmt(ratio(sum(g[side]['ownBlastDanger'] for g in games), sum(g[side]['ownBlastDangerSteps'] for g in games)), 2),
                     fmt(ratio(sum(g[side]['ownMines'] for g in games), alive), 3)])
    table('own projectiles: % of live steps within blast range of an own explosive, explosives in range then, mean own '
          'live mines', ['run', '% steps in own blast range', 'explosives then', 'own mines'], rows)


def section_shadow(runs, side):
    if side != 'p':
        return
    for name, games in runs:
        if 'shadow' not in games[0]:
            continue
        rows = []
        for k, sh in enumerate(games[0]['shadow']):
            n = sum(g['shadow'][k]['n'] for g in games)
            kl = [sum(g['shadow'][k]['kl'][a] for g in games) / n for a in range(len(sh['kl']))]
            rows.append([sh['name'], fmt(sum(kl), 3)] + [fmt(x, 3) for x in kl] +
                        [fmt(sum(g['shadow'][k]['dv'] for g in games) / n, 4), fmt(sum(g['shadow'][k]['absDv'] for g in games) / n, 4)])
        rows.sort(key=lambda r: -float(r[1]))
        table(f'{name}: shadow ablations, KL(policy || ablated) per step and value change',
              ['obs group', 'KL total'] + atn_names(len(games[0]['shadow'][0]['kl'])) + ['dV', '|dV|'], rows)


def combo_name(a, b):
    return f'{WEAPONS[a]}+{WEAPONS[b]}'


def section_combos(runs, side):
    if any('utilityLoadout' in g for _, games in runs for g in games):
        section_utility_loadouts(runs, side)
        return
    for name, games in runs:
        games = [g for g in games if 'loadout' in g]
        if not games:
            continue
        # every game scores both drones' loadouts; in self-play both sides
        # are the same policy
        combo_scores = defaultdict(list)
        weapon_scores = defaultdict(list)
        matchup = defaultdict(list)
        combo_kills = defaultdict(lambda: [0.0] * len(WEAPONS))
        combo_games = defaultdict(int)
        combo_swaps = defaultdict(float)
        combo_setup = defaultdict(lambda: [0.0, 0.0])
        combo_held = defaultdict(lambda: [0.0, 0.0])
        nw = len(WEAPONS)
        for g in games:
            l = g['loadout']
            for sd, own, opp in (('p', l[0:2], l[2:4]), ('b', l[2:4], l[0:2])):
                c = tuple(sorted(own))
                o = tuple(sorted(opp))
                sc = g[sd]['score']
                combo_scores[c].append(sc)
                matchup[(c, o)].append(sc)
                for w in c:
                    weapon_scores[w].append(sc)
                for w in range(len(WEAPONS)):
                    combo_kills[c][w] += g[sd]['killsByWeapon'][w]
                combo_games[c] += 1
                combo_swaps[c] += g[sd].get('swaps', 0)
                if 'setupKills' in g[sd]:
                    sk = g[sd]['setupKills']
                    # kills by one loadout weapon set up by the other
                    combo_setup[c][0] += sk[c[0] * nw + c[1]]
                    combo_setup[c][1] += sk[c[1] * nw + c[0]]
                combo_held[c][0] += g[sd]['held'][c[0]] + g[sd]['held'][c[1]]
                combo_held[c][1] += g[sd]['alive']
        rows = []
        for w, xs in sorted(weapon_scores.items(), key=lambda kv: -sum(kv[1]) / len(kv[1])):
            m, ci = mean_ci(xs)
            rows.append([WEAPONS[w], f'{fmt(m)}±{fmt(ci)}', len(xs)])
        table(f'{name}: score of loadouts containing each weapon', ['weapon', 'score', 'n'], rows)

        wmean = {w: sum(xs) / len(xs) for w, xs in weapon_scores.items()}
        rows = []
        for c, xs in sorted(combo_scores.items(), key=lambda kv: -sum(kv[1]) / len(kv[1])):
            m, ci = mean_ci(xs)
            n = combo_games[c]
            kills = combo_kills[c]
            tot = sum(kills)
            # score beyond what the two weapons' marginal scores predict
            synergy = m - (wmean[c[0]] + wmean[c[1]] - 0.5)
            others = tot - kills[c[0]] - kills[c[1]]
            st = combo_setup[c]
            rows.append([combo_name(*c), f'{fmt(m)}±{fmt(ci)}', f'{synergy:+.3f}', fmt(tot / n, 2),
                         fmt(100 * ratio(kills[c[0]], tot), 0), fmt(100 * ratio(kills[c[1]], tot), 0),
                         fmt(100 * ratio(others, tot), 0), fmt(100 * ratio(combo_held[c][0], combo_held[c][1]), 0),
                         fmt(combo_swaps[c] / n, 1), fmt(100 * ratio(st[0], kills[c[0]]), 0),
                         fmt(100 * ratio(st[1], kills[c[1]]), 0)])
        table(f'{name}: combo scores (synergy = score - marginal prediction; kill shares of weapon 1, 2, others; '
              f'held% = loadout weapon in hand; 2>1% = weapon 1 kills set up by a weapon 2 hit in the 2s before)',
              ['combo', 'score', 'synergy', 'kills/g', 'k1%', 'k2%', 'other%', 'held%', 'swaps/g', '2>1%', '1>2%'], rows)

        # loadout setup kills over all combos, by setup and killing weapon
        pair = defaultdict(lambda: [0.0, 0.0])
        for c in combo_scores:
            # combo_setup[c][0] is weapon 1 kills set up by weapon 2
            pair[(c[1], c[0])][0] += combo_setup[c][0]
            pair[(c[0], c[1])][0] += combo_setup[c][1]
            pair[(c[1], c[0])][1] += combo_kills[c][c[0]]
            pair[(c[0], c[1])][1] += combo_kills[c][c[1]]
        rows = [[WEAPONS[a], WEAPONS[b], fmt(100 * ratio(v, k), 1), int(v)]
                for (a, b), (v, k) in sorted(pair.items(), key=lambda kv: -ratio(kv[1][0], kv[1][1])) if v > 0][:20]
        table(f'{name}: loadout pairs by % of the killing weapon\'s kills set up by the other weapon hitting first',
              ['setup', 'killer', '% kills set up', 'kills'], rows)

        # most lopsided matchups and each combo's worst counter
        rows = []
        for c in sorted(combo_scores):
            worst = min(((sum(v) / len(v), o) for (cc, o), v in matchup.items() if cc == c and o != c))
            best = max(((sum(v) / len(v), o) for (cc, o), v in matchup.items() if cc == c and o != c))
            rows.append([combo_name(*c), combo_name(*worst[1]), fmt(worst[0], 2), combo_name(*best[1]), fmt(best[0], 2)])
        rows.sort(key=lambda r: float(r[2]))
        table(f'{name}: each combo\'s worst and best matchup (score; few games per matchup, noisy)',
              ['combo', 'worst vs', 'score', 'best vs', 'score'], rows)

        # weapon vs weapon: score of loadouts containing a vs loadouts containing b
        ws = sorted(weapon_scores, key=lambda w: -wmean[w])
        grid = defaultdict(list)
        for (c, o), v in matchup.items():
            for a in c:
                for b in o:
                    grid[(a, b)].extend(v)
        rows = []
        for a in ws:
            rows.append([WEAPONS[a]] + [fmt(sum(grid[(a, b)]) / len(grid[(a, b)]), 2) if grid[(a, b)] else '-' for b in ws])
        table(f'{name}: score of loadouts with weapon (row) vs loadouts with weapon (column)', ['weapon'] + [WEAPONS[b][:6] for b in ws], rows)


# from the utility slot on a loadout is a weapon and a utility per drone
def section_utility_loadouts(runs, side):
    for name, games in runs:
        games = [g for g in games if 'utilityLoadout' in g]
        if not games:
            continue
        loadout_scores = defaultdict(list)
        weapon_scores = defaultdict(list)
        utility_scores = defaultdict(list)
        utility_matchup = defaultdict(list)
        loadout_kills = defaultdict(lambda: [0.0] * len(WEAPONS))
        for g in games:
            l, u = g['loadout'], g['utilityLoadout']
            for sd, own, opp in (('p', 0, 1), ('b', 1, 0)):
                sc = g[sd]['score']
                lo = (l[own], u[own])
                loadout_scores[lo].append(sc)
                weapon_scores[l[own]].append(sc)
                utility_scores[u[own]].append(sc)
                utility_matchup[(u[own], u[opp])].append(sc)
                for w in range(len(WEAPONS)):
                    loadout_kills[lo][w] += g[sd]['killsByWeapon'][w]
        mean = lambda xs: sum(xs) / len(xs)

        rows = []
        for u, xs in sorted(utility_scores.items(), key=lambda kv: -mean(kv[1])):
            m, ci = mean_ci(xs)
            rows.append([UTILITIES[u], f'{fmt(m)}±{fmt(ci)}', len(xs)])
        table(f'{name}: score of loadouts with each utility', ['utility', 'score', 'n'], rows)
        us = sorted(utility_scores)
        rows = [[UTILITIES[a]] + [fmt(mean(utility_matchup[(a, b)]), 3) if utility_matchup[(a, b)] else '-' for b in us] for a in us]
        table(f'{name}: score of loadouts with utility (row) vs utility (column)', ['utility'] + [UTILITIES[b] for b in us], rows)

        rows = []
        for w, xs in sorted(weapon_scores.items(), key=lambda kv: -mean(kv[1])):
            m, ci = mean_ci(xs)
            rows.append([WEAPONS[w], f'{fmt(m)}±{fmt(ci)}', len(xs)])
        table(f'{name}: score of loadouts with each weapon', ['weapon', 'score', 'n'], rows)

        ws = sorted(weapon_scores, key=lambda w: -mean(weapon_scores[w]))
        rows = []
        for w in ws:
            row = [WEAPONS[w]]
            for u in us:
                xs = loadout_scores[(w, u)]
                if not xs:
                    row += ['-', '-']
                    continue
                kills = loadout_kills[(w, u)]
                tot = sum(kills)
                row += [fmt(mean(xs), 3), fmt(100 * ratio(kills[WEAPONS.index(UTILITY_WEAPONS[u])], tot), 0)]
            rows.append(row)
        table(f'{name}: score of each weapon + utility loadout, and % of its kills by the utility',
              ['weapon'] + [c for u in us for c in (UTILITIES[u], 'util k%')], rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('runs', nargs='+', help='[name=]games.jsonl, the first is the baseline')
    parser.add_argument('--side', choices=['p', 'b'], default='p', help='p: the policy, b: the bot')
    parser.add_argument('--sections', default=None, help=f'comma separated subset of {",".join(SECTIONS + ["combos"])}')
    args = parser.parse_args()
    runs = []
    for spec in args.runs:
        name, _, path = spec.rpartition('=')
        runs.append((name or os.path.splitext(os.path.basename(path))[0], load(path)))
    global WEAPONS
    WEAPONS = weapon_names(runs[0][1][0]['p'])
    sections = args.sections
    if sections is None:
        if 'loadout' in runs[0][1][0]:
            sections = 'combos,utilities,pace,weapons,slots,behavior'
        else:
            sections = ','.join(SECTIONS)
    for section in sections.split(','):
        globals()[f'section_{section}'](runs, args.side)


if __name__ == '__main__':
    main()
