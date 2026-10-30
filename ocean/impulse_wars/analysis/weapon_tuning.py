"""Shows how weapon settings turn into impulses on drones, from settings.h.

Usage:
  python weapon_tuning.py                             all tables
  python weapon_tuning.py -w sniper,shotgun -s hit    some weapons/sections
  python weapon_tuning.py --set sniper.density=3 --set SNIPER_RADIUS=0.4
                                                      what-if, cells show old→new
  python weapon_tuning.py --sweep shotgun.fire=15:35:5
                                                      metrics across a range
  python weapon_tuning.py --set grapple.density=1 --match grapple.fire
                                                      solve grapple.fire so the
                                                      grapple's speed stays as is

--match NAME[=SECTION:METRIC] solves NAME so that METRIC, on NAME's weapon's
row, keeps its current settings.h value after the other overrides apply.
Without a metric, fire, density and radius keep launch speed, recoil keeps
recoil dv and explosion settings keep the trigger impulse. Settings that
aren't weapon specific need the row too: DRONE_DENSITY=hit:sniper:J. Works
with --sweep, which then also shows the solved values.

--set and --sweep take any settings.h constant, or weapon.field where
weapon is a struct name or macro prefix (sniper, machineGun, flak_cannon)
and field is fire, recoil, density, radius, damping, cooldown, charge, ammo,
projectiles, max_distance, bounce, expl_radius, expl_impulse, proximity, or
any other suffix of the weapon's macros. Overridden constants are
re-evaluated through settings.h, so *_MASS and the weapon structs follow.

Formulas mirror game.h and need updating if it changes. Everything is
evaluated against a stationary drone hit head on, with the projectile flying
straight at it from --range units away:

  launch       speed = fireMagnitude / mass; mass = density * pi * r^2
               shotgun fire magnitude jitters by +-SHOTGUN_FIRE_JITTER
  flight       linear damping c makes speed fall linearly with distance:
               v(x) = v0 - c * x, so a damped projectile stops at v0 / c
  recoil       drone dv = recoilMagnitude / DRONE_MASS
  drift        an impulse of dv on a drone carries it dv / DRONE_LINEAR_DAMPING
               units before damping stops it (ignoring movement)
  shot hit     restitution 1 elastic hit plus the reported impulse applied
               again: J = SHOT_HIT_IMPULSE_SCALE * mu * v, where mu is the
               reduced mass m_p * m_d / (m_p + m_d). Shields are kinematic,
               so shots on a shield move the drone 0
  explosion    J = (|EXPLOSION_IMPULSE| + v_toward) * 2 * DRONE_RADIUS
               * (1 - (d / R)^3), d = distance from the blast center to the
               drone's surface; v_toward is the projectile's speed toward the
               drone, only added on direct hits here. Shields scale J by
               DRONE_SHIELD_EXPLOSION_REDUCTION
  reward       hitReward = min(J / DRONE_MASS / HIT_DV_REF, 1)
  black hole   force = (|PULL| + v_toward) * 2 * DRONE_RADIUS * (1 - d / R)
  burst        J = (IMPACT_BASE * c + IMPACT_MIN) * 2 * r * (1 - (d / R)^3),
               R = RADIUS_BASE * c + RADIUS_MIN for burst charge c; boosting
               an own projectile on its fire frame scales J by burstBoostCoef

Per second numbers use cooldown + charge as the fire cycle and assume every
projectile hits. Hit angle, drone velocity, wall bounces and box2d's
discrete integration are ignored, so treat small differences as noise.
"""
import argparse
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from energy_refill import Settings  # noqa: E402

WEAPONS = ['standard', 'machineGun', 'sniper', 'shotgun', 'imploder',
           'flakCannon', 'blackHole', 'nuke', 'grapple', 'mineLauncher', 'portal']
# portal projectiles open portals instead of hitting drones
NO_HIT = {'portal'}
# where explosive projectiles usually go off, as the distance from the blast
# center to the drone's surface; None means on contact
TRIGGER_DISTANCE = {
    'flakCannon': 'FLAK_CANNON_PROXIMITY_RADIUS',
    'mineLauncher': 'MINE_LAUNCHER_PROXIMITY_RADIUS',
}
# mines are welded to walls when they go off, so they add no speed
STATIONARY_EXPLOSIONS = {'mineLauncher'}
# hardcoded in weaponFire in settings.h
SHOTGUN_FIRE_JITTER = 3.0
BOX2D_MAX_SPEED = 400.0

FIELD_ALIASES = {
    'fire': 'FIRE_MAGNITUDE', 'recoil': 'RECOIL_MAGNITUDE', 'charge': 'CHARGE',
    'cooldown': 'COOL_DOWN', 'max_distance': 'MAX_DISTANCE', 'range': 'MAX_DISTANCE',
    'expl_radius': 'EXPLOSION_RADIUS', 'expl_impulse': 'EXPLOSION_IMPULSE',
    'proximity': 'PROXIMITY_RADIUS', 'refill': 'ENERGY_REFILL_COEF',
    'boost': 'BURST_BOOST_COEF', 'pull': 'PULL_MAGNITUDE',
}

WEAPON_ALIASES = {
    'mg': 'machineGun', 'flak': 'flakCannon', 'bh': 'blackHole', 'mine': 'mineLauncher',
    'mines': 'mineLauncher',
}

# metric each kind of weapon setting keeps by default when matched
DEFAULT_MATCH = {
    'FIRE_MAGNITUDE': ('proj', 'speed'),
    'DENSITY': ('proj', 'speed'),
    'RADIUS': ('proj', 'speed'),
    'RECOIL_MAGNITUDE': ('proj', 'recoil dv'),
    'EXPLOSION_IMPULSE': ('expl', 'J trigger'),
    'EXPLOSION_RADIUS': ('expl', 'J trigger'),
}

SECTIONS = ['ref', 'proj', 'hit', 'expl', 'blackhole', 'burst', 'boost']


def load(settings_path, overrides):
    d = os.path.dirname(os.path.abspath(settings_path))
    s = Settings([os.path.join(d, 'helpers.h'), settings_path])
    for name, value in overrides.items():
        if name not in s.exprs:
            sys.exit(f'unknown setting {name}')
        s.exprs[name] = repr(value)
    return s


def weapon_prefix(s, name):
    return re.match(r'(\w+)_FIRE_MAGNITUDE', s.structs[name]['fireMagnitude'].strip()).group(1)


def resolve_name(s, key):
    """Maps sniper.density or SNIPER_DENSITY to a settings.h name."""
    if '.' not in key:
        return key
    weapon, field = key.split('.', 1)
    prefix = None
    weapon = WEAPON_ALIASES.get(weapon.lower(), weapon)
    norm = weapon.replace('_', '').lower()
    for name in WEAPONS:
        p = weapon_prefix(s, name)
        if norm in (name.lower(), p.replace('_', '').lower()):
            prefix = p
    if prefix is None:
        if weapon.upper() in ('DRONE', 'BURST'):
            prefix = 'DRONE_BURST' if weapon.upper() == 'BURST' else 'DRONE'
        else:
            sys.exit(f'unknown weapon {weapon}')
    return f'{prefix}_{FIELD_ALIASES.get(field.lower(), field.upper())}'


def parse_overrides(s, items):
    out = {}
    for item in items:
        key, value = item.split('=', 1)
        out[resolve_name(s, key)] = float(value)
    return out


def setting_weapon(s, name):
    """The weapon whose macros name belongs to, or None."""
    best = None
    for weapon in WEAPONS:
        p = weapon_prefix(s, weapon)
        if name.startswith(p + '_') and (best is None or len(p) > len(weapon_prefix(s, best))):
            best = weapon
    return best


def parse_matches(s, items):
    """Returns [(name, section, row, metric)] for each --match."""
    out = []
    for item in items:
        key, _, spec = item.partition('=')
        name = resolve_name(s, key)
        if name not in s.exprs:
            sys.exit(f'unknown setting {name}')
        weapon = setting_weapon(s, name)
        parts = spec.split(':') if spec else []
        if len(parts) == 3:
            sec, row, metric = parts
            row = WEAPON_ALIASES.get(row.lower(), row)
        elif len(parts) == 2:
            sec, metric = parts
            row = weapon
        elif not parts:
            suffix = next((k for k in DEFAULT_MATCH if weapon and name == f'{weapon_prefix(s, weapon)}_{k}'), None)
            if suffix is None:
                sys.exit(f'no default metric to match {name} on, use {key}=SECTION:METRIC')
            sec, metric = DEFAULT_MATCH[suffix]
            row = weapon
        else:
            sys.exit(f'bad --match {item}, use NAME[=SECTION:[ROW:]METRIC]')
        if row is None:
            sys.exit(f'{name} isn\'t weapon specific, use {key}=SECTION:ROW:METRIC')
        out.append((name, sec, row, metric))
    return out


def metric_value(results, sec, row, metric):
    try:
        return results[sec][row][metric]
    except KeyError:
        rows = results.get(sec, {})
        if row not in rows:
            sys.exit(f'no row {row} in section {sec}, rows: {", ".join(rows)}')
        sys.exit(f'no metric {metric!r} for {row} in {sec}, metrics: {", ".join(rows[row])}')


def solve_matches(settings_path, rng, overrides, matches, targets):
    """Adds the solved value of each matched setting to overrides."""
    out = dict(overrides)
    # matches can affect each other, so repeat until they settle
    for _ in range(4 if len(matches) > 1 else 1):
        for (name, sec, row, metric), target in zip(matches, targets):
            def err(x):
                res = compute(load(settings_path, {**out, name: x}), rng)
                return metric_value(res, sec, row, metric) - target

            x0 = out.get(name, load(settings_path, out)[name])
            out[name] = solve(err, x0, target, name, f'{sec}:{metric}')
    return out


def solve(err, x0, target, name, what):
    tol = 1e-7 * max(abs(target), 1e-6)
    if x0 == 0.0:
        sys.exit(f"can't match {name}, it's 0 so it can't be scaled")
    f0 = err(x0)
    if abs(f0) <= tol:
        return x0
    # secant first, most metrics are linear or smooth in a setting
    a, fa, b = x0, f0, x0 * 1.1
    for _ in range(30):
        fb = err(b)
        if abs(fb) <= tol:
            return b
        if fb == fa:
            break
        a, fa, b = b, fb, b - fb * (b - a) / (fb - fa)
        if b == 0.0 or b * x0 < 0.0:
            break
    # fall back to bracketing by scaling, then bisection
    for k in range(1, 40):
        for x in (x0 * 1.5 ** k, x0 / 1.5 ** k):
            fx = err(x)
            if fx * f0 <= 0.0:
                lo, hi, flo = (x0, x, f0)
                for _ in range(100):
                    mid = 0.5 * (lo + hi)
                    fm = err(mid)
                    if abs(fm) <= tol:
                        return mid
                    lo, hi, flo = (mid, hi, fm) if fm * flo > 0.0 else (lo, mid, flo)
                return 0.5 * (lo + hi)
    sys.exit(f"can't match {name} to keep {what} at {fmt(target)}, the metric may be capped or not depend on it")


def speed_at(v0, damping, x):
    """Projectile speed after flying x units, or 0 if it stops first."""
    return max(v0 - damping * x, 0.0)


def time_to(v0, damping, x):
    if damping == 0.0:
        return x / v0 if v0 > 0 else math.inf
    if damping * x >= v0:
        return math.inf
    return -math.log(1.0 - damping * x / v0) / damping


def sidestep_time(dist, force, mass, damping):
    """Seconds for a drone at rest to move dist sideways at full thrust."""
    v_term = force / (mass * damping)

    def x(t):
        return v_term * (t - (1.0 - math.exp(-damping * t)) / damping)

    lo, hi = 0.0, 1.0
    while x(hi) < dist:
        hi *= 2.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if x(mid) < dist else (lo, mid)
    return hi


def pellets_hitting(s, w, name, rng):
    """Projectiles of a volley that hit a drone at rng when aimed at its center."""
    n = int(w['numProjectiles'])
    if name != 'shotgun' or n == 1:
        return n
    spread = s['SHOTGUN_SPREAD']
    step = 2.0 * spread / (n - 1)
    half_width = s['DRONE_RADIUS'] + w['radius']
    return sum(1 for i in range(n) if rng * math.tan(abs(-spread + i * step)) < half_width)


def falloff(d, radius, cubic=True):
    if d >= radius:
        return 0.0
    rel = max(d, 0.0) / radius
    return 1.0 - (rel ** 3 if cubic else rel)


def reach(w):
    """How far a projectile can travel before being removed or stopping."""
    stop = w['initialSpeed'] / w['damping'] if w['damping'] > 0 else math.inf
    limit = math.inf if w['maxDistance'] < 0 else w['maxDistance']
    return min(stop, limit)


def compute(s, rng):
    """Returns {section: {weapon: {metric: value}}}, metrics in display order."""
    drone_m = s['DRONE_MASS']
    drone_c = s['DRONE_LINEAR_DAMPING']
    drone_r = s['DRONE_RADIUS']
    dv_ref = s['HIT_DV_REF']
    hit_scale = s['SHOT_HIT_IMPULSE_SCALE']
    shield_hp = s['DRONE_SHIELD_HEALTH']
    move_f = s['DRONE_MOVE_MAGNITUDE']

    def reward(j):
        return min(abs(j) / drone_m / dv_ref, 1.0)

    out = {k: {} for k in SECTIONS}
    out['ref']['drone'] = {
        'mass': drone_m,
        'move speed': move_f / (drone_m * drone_c),
        'brake speed': move_f / (drone_m * drone_c * s['DRONE_BRAKE_DAMPING_COEF']),
        'dv for max reward': dv_ref,
        'J for max reward': dv_ref * drone_m,
        'drift per dv': 1.0 / drone_c,
        'shield hp': shield_hp,
        'action period': 1.0 / s['TRAINING_ACTIONS_PER_SECOND'],
    }

    for name in WEAPONS:
        w = s.weapon(name)
        p = weapon_prefix(s, name)
        m = w['mass']
        v0 = w['fireMagnitude'] / m
        w['initialSpeed'] = v0
        recoil_dv = w['recoilMagnitude'] / drone_m
        cycle = w['cooldown'] + w['maxCharge']
        n = w['numProjectiles']
        ammo = w['ammo'] if 'ammo' in w and w['ammo'] != 0 else s.get(f'{p}_UTILITY_AMMO', s.get(f'{p.split("_")[0]}_UTILITY_AMMO'))

        proj = {
            'mass': m,
            'speed': v0,
            'momentum': w['fireMagnitude'],
            f'speed@{rng:g}': speed_at(v0, w['damping'], rng),
            f'time@{rng:g}': time_to(v0, w['damping'], rng),
            'reach': reach(w),
            'recoil dv': recoil_dv,
            'recoil drift': recoil_dv / drone_c,
            'cycle s': cycle,
        }
        if name == 'shotgun':
            proj['speed min'] = (w['fireMagnitude'] - SHOTGUN_FIRE_JITTER) / m
            proj['speed max'] = (w['fireMagnitude'] + SHOTGUN_FIRE_JITTER) / m
        out['proj'][name] = proj

        if name in NO_HIT:
            continue

        # direct hit
        v_hit = speed_at(v0, w['damping'], rng)
        mu = m * drone_m / (m + drone_m)
        reaction = 1.0 / s['TRAINING_ACTIONS_PER_SECOND']
        dodge = sidestep_time(drone_r + w['radius'], move_f, drone_m, drone_c)
        hits = pellets_hitting(s, w, name, rng)
        variants = [(name, v_hit)]
        for row, v in variants:
            j = hit_scale * mu * v
            shield_dmg = v * m * s['DRONE_SHIELD_HEALTH_IMPULSE_COEF']
            out['hit'][row] = {
                'J': j,
                'dv': j / drone_m,
                'reward': reward(j),
                'hits/volley': hits,
                'volley J': j * hits,
                'volley reward': reward(j * hits),
                'J/s': j * hits / cycle if cycle > 0 else math.inf,
                'J/ammo load': j * hits * ammo if ammo and ammo > 0 else math.inf,
                'shield shots': math.ceil(shield_hp / shield_dmg) if shield_dmg > 0 else math.inf,
                'energy/hit': v * m * w['energyRefillCoef'],
                # time left over to react and sidestep after one action step
                'dodge margin s': time_to(v0, w['damping'], rng) - reaction - dodge,
            }

        if w['explosive']:
            radius = s[f'{p}_EXPLOSION_RADIUS']
            impulse = s[f'{p}_EXPLOSION_IMPULSE']
            trig = TRIGGER_DISTANCE.get(name)
            trig_d = s[trig] if trig else w['radius']
            v_toward = 0.0 if name in STATIONARY_EXPLOSIONS else v_hit
            perim = 2.0 * drone_r
            j_trig = (abs(impulse) + v_toward) * perim * falloff(trig_d, radius)
            shield_j = j_trig * s['DRONE_SHIELD_EXPLOSION_REDUCTION']
            expl = {
                'radius': radius,
                'trigger d': trig_d,
                'J trigger': j_trig,
                'dv trigger': j_trig / drone_m,
                'reward trigger': reward(j_trig),
            }
            for frac in (0.25, 0.5, 0.75):
                expl[f'J@{frac:g}R'] = abs(impulse) * perim * falloff(frac * radius, radius)
            expl['dv shielded'] = shield_j / drone_m
            expl['shield dmg'] = shield_j * s['DRONE_SHIELD_HEALTH_EXPLOSION_COEF']
            expl['energy trigger'] = j_trig / drone_m / s['MAX_SPEED'] * s['EXPLOSION_ENERGY_REFILL_COEF']
            # floating walls are hit face on, so the projected perimeter is
            # their thickness
            wall_j = abs(impulse) * s['FLOATING_WALL_THICKNESS']
            expl['wall dv@0'] = wall_j * s['FLOATING_WALL_INV_MASS']
            notes = []
            if impulse < 0:
                notes.append('pulls in')
            if name == 'flakCannon':
                notes.append(f"arms after {fmt(s['FLAK_CANNON_SAFE_DISTANCE'])}u")
            expl['note'] = ', '.join(notes)
            out['expl'][name] = expl

        if name == 'blackHole':
            pull = abs(s['BLACK_HOLE_PULL_MAGNITUDE'])
            radius = s['BLACK_HOLE_PROXIMITY_RADIUS']
            perim = 2.0 * drone_r
            shield = s['DRONE_SHIELD_FORCE_REDUCTION']
            bh = {}
            for d in (0.0, 0.25 * radius, 0.5 * radius, 0.75 * radius):
                f = pull * perim * falloff(d, radius, cubic=False)
                bh[f'F@{d:g}'] = f
            # beyond this distance a drone can outrun the pull
            bh['escape d'] = radius * max(1.0 - move_f / (pull * perim), 0.0)
            bh['escape d shielded'] = radius * max(1.0 - move_f / (pull * perim * shield), 0.0)
            f_half = pull * perim * falloff(0.5 * radius, radius, cubic=False)
            bh[f'flee speed@{0.5 * radius:g}'] = (move_f - f_half) / (drone_m * drone_c)
            bh['energy/s@0'] = pull * perim / drone_m / s['MAX_SPEED'] * s['BLACK_HOLE_PULL_ENERGY_REFILL_COEF']
            bh['move force'] = move_f
            out['blackhole']['blackHole'] = bh

    # burst
    for c in (0.0, 0.25, 0.5, 0.75, 1.0):
        radius = s['DRONE_BURST_RADIUS_BASE'] * c + s['DRONE_BURST_RADIUS_MIN']
        impulse = s['DRONE_BURST_IMPACT_BASE'] * c + s['DRONE_BURST_IMPACT_MIN']
        perim = 2.0 * drone_r
        # touching drones are 2 radii apart, so the surface is 1 radius away
        j_touch = impulse * perim * falloff(drone_r, radius)
        j_half = impulse * perim * falloff(0.5 * radius, radius)
        # pushing off a static wall touching the drone, log compressed
        wall_raw = impulse * s['WALL_THICKNESS'] * falloff(drone_r, radius)
        wall_j = math.log2(max(wall_raw, 1.0)) * (5.0 + 25.0 * c)
        b = {
            'radius': radius,
            'impulse/len': impulse,
            'J touching': j_touch,
            'dv touching': j_touch / drone_m,
            'reward touching': reward(j_touch),
            'dv@0.5R': j_half / drone_m,
            'wall push dv': wall_j / drone_m,
            'energy cost': s['DRONE_BURST_BASE_COST'] + c * (1.0 - s['DRONE_BURST_BASE_COST']),
        }
        out['burst'][f'charge {c:g}'] = b

    # boosting own projectiles on the fire frame, at full and min charge
    for name in WEAPONS:
        if name in NO_HIT:
            continue
        w = s.weapon(name)
        m = w['mass']
        v0 = w['fireMagnitude'] / m
        r = w['radius']
        d = drone_r + 0.5 * r
        row = {}
        for c in (0.0, 1.0):
            radius = s['DRONE_BURST_RADIUS_BASE'] * c + s['DRONE_BURST_RADIUS_MIN']
            impulse = s['DRONE_BURST_IMPACT_BASE'] * c + s['DRONE_BURST_IMPACT_MIN']
            dv = impulse * 2.0 * r * falloff(d, radius) * w['burstBoostCoef'] / m
            row[f'dv c{c:g}'] = dv
            row[f'speed c{c:g}'] = min(v0 + dv, BOX2D_MAX_SPEED)
            row[f'x c{c:g}'] = min(v0 + dv, BOX2D_MAX_SPEED) / v0
        out['boost'][name] = row

    return out


def fmt(v):
    if isinstance(v, str):
        return v
    if v is None:
        return '-'
    if isinstance(v, bool):
        return 'yes' if v else 'no'
    if math.isinf(v):
        return 'inf'
    if isinstance(v, int) or float(v).is_integer() and abs(v) >= 1:
        return f'{int(v)}'
    a = abs(v)
    if a >= 100:
        return f'{v:.0f}'
    if a >= 10:
        return f'{v:.1f}'
    if a >= 1:
        return f'{v:.2f}'
    if a >= 0.001:
        return f'{v:.3f}'
    return f'{v:.2e}' if a > 0 else '0'


def cell(new, old):
    if old is None or fmt(old) == fmt(new):
        return fmt(new)
    return f'{fmt(old)}→{fmt(new)}'


def print_table(title, rows, base_rows=None, only_changed=False):
    if not rows:
        return
    cols = []
    for r in rows.values():
        for k in r:
            if k not in cols:
                cols.append(k)
    table = []
    for name, r in rows.items():
        old = (base_rows or {}).get(name, {})
        cells = [cell(r[k], old.get(k)) if k in r else '' for k in cols]
        if only_changed and base_rows is not None and all('→' not in c for c in cells):
            continue
        table.append([name] + cells)
    if not table:
        return
    header = [''] + cols
    widths = [max(len(str(x)) for x in col) for col in zip(header, *table)]
    print(f'\n== {title} ==')
    print('  '.join(h.rjust(wd) for h, wd in zip(header, widths)))
    for row in table:
        print('  '.join(c.rjust(wd) if i else c.ljust(wd) for i, (c, wd) in enumerate(zip(row, widths))))


TITLES = {
    'ref': 'reference: drone',
    'proj': 'projectile flight and recoil (speeds u/s, times s, distances u)',
    'hit': 'direct hit on an unshielded drone at --range',
    'expl': 'explosions (J = impulse on drone; trigger = where it usually goes off)',
    'blackhole': 'black hole pull force on a drone (vs move force)',
    'burst': 'burst (touching = enemy drone touching the bursting drone)',
    'boost': 'bursting on the fire frame to boost own projectiles, min (c0) and full (c1) charge',
}


def parse_weapons(arg):
    if not arg:
        return None
    return [WEAPON_ALIASES.get(w.strip().lower(), w.strip()).lower() for w in arg.split(',')]


def filter_weapons(rows, weapons):
    if not weapons:
        return rows
    return {k: v for k, v in rows.items()
            if not any(w.lower() in k.lower() for w in WEAPONS) or any(w in k.lower() for w in weapons)}


def run_tables(args, s_base, s_new):
    base = compute(s_base, args.range) if s_new is not None else None
    new = compute(s_new if s_new is not None else s_base, args.range)
    weapons = parse_weapons(args.weapons)
    for sec in args.sections:
        print_table(TITLES[sec], filter_weapons(new[sec], weapons),
                    filter_weapons(base[sec], weapons) if base else None, args.changed)


def run_sweep(args, s_base, overrides, matches, targets):
    key, spec = args.sweep.split('=', 1)
    name = resolve_name(s_base, key)
    parts = [float(x) for x in spec.split(':')]
    if len(parts) == 2:
        parts.append(5)
    lo, hi, steps = parts[0], parts[1], int(parts[2])
    values = [lo + (hi - lo) * i / max(steps - 1, 1) for i in range(steps)]
    base = compute(load(args.settings, solve_matches(args.settings, args.range, overrides, matches, targets)), args.range)
    results = []
    solved = {}
    for v in values:
        o = solve_matches(args.settings, args.range, {**overrides, name: v}, matches, targets)
        solved[f'{name}={fmt(v)}'] = {m[0]: o[m[0]] for m in matches}
        results.append((v, compute(load(args.settings, o), args.range)))
    print(f'sweep {name} (current {fmt(s_base[name])}), showing metrics that change')
    for m, target in zip(matches, targets):
        print(f'matching {m[0]} to keep {m[2]} {m[1]}:{m[3]} at {fmt(target)}')
    if matches:
        print_table('matched settings', solved)
    weapons = parse_weapons(args.weapons)
    for sec in args.sections:
        for row in filter_weapons(base[sec], weapons):
            cols = [k for k in base[sec][row]
                    if any(fmt(r[sec].get(row, {}).get(k)) != fmt(base[sec][row][k]) for _, r in results)]
            if not cols:
                continue
            rows = {f'{name}={fmt(v)}': {k: r[sec][row][k] for k in cols} for v, r in results}
            print_table(f'{TITLES[sec].split(" (")[0]}: {row}', rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--settings', default=os.path.join(here, '..', 'settings.h'))
    parser.add_argument('-r', '--range', type=float, default=15.0,
                        help='shooter to target distance for direct hits (default 15)')
    parser.add_argument('-w', '--weapons', help='comma separated weapon names to show')
    parser.add_argument('-s', '--sections', default=','.join(SECTIONS),
                        help=f'comma separated sections from {",".join(SECTIONS)}')
    parser.add_argument('--set', action='append', default=[], metavar='NAME=VALUE',
                        help='override a setting, repeatable')
    parser.add_argument('--match', action='append', default=[], metavar='NAME[=SECTION:[ROW:]METRIC]',
                        help='solve a setting so a metric keeps its current value, repeatable')
    parser.add_argument('--sweep', metavar='NAME=LO:HI[:STEPS]', help='sweep a setting')
    parser.add_argument('--changed', action='store_true', help='with --set, only show rows that changed')
    args = parser.parse_args()
    args.sections = [x.strip() for x in args.sections.split(',')]
    for sec in args.sections:
        if sec not in SECTIONS:
            sys.exit(f'unknown section {sec}, pick from {",".join(SECTIONS)}')

    s_base = load(args.settings, {})
    overrides = parse_overrides(s_base, args.set)
    matches = parse_matches(s_base, args.match)
    for m in matches:
        if m[0] in overrides:
            sys.exit(f'{m[0]} is both set and matched')
    base_results = compute(s_base, args.range)
    targets = [metric_value(base_results, sec, row, metric) for _, sec, row, metric in matches]
    if args.sweep:
        if any(m[0] == resolve_name(s_base, args.sweep.split('=', 1)[0]) for m in matches):
            sys.exit('the swept setting can\'t also be matched')
        run_sweep(args, s_base, overrides, matches, targets)
        return
    set_names = list(overrides)
    overrides = solve_matches(args.settings, args.range, overrides, matches, targets)
    s_new = load(args.settings, overrides) if overrides else None
    if overrides:
        print('overrides: ' + ', '.join(f'{k}={fmt(s_base[k])}→{fmt(overrides[k])}' for k in set_names))
    for (name, sec, row, metric), target in zip(matches, targets):
        print(f'matched:   {name}={fmt(s_base[name])}→{overrides[name]:.6g} '
              f'(keeps {row} {sec}:{metric} at {fmt(target)})')
    run_tables(args, s_base, s_new)


if __name__ == '__main__':
    main()
