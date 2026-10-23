"""Prints how much energy each weapon refills its shooter, from settings.h.

Usage: python energy_refill.py [--settings path/to/settings.h]

Constants and weapon structs are read from settings.h, so the tables follow
balance changes; the formulas below mirror game.h and need updating if those
change:

  direct hit   lastSpeed * mass * energyRefillCoef      handleProjectileBeginContact
               (projectile momentum; speed is kept through bounces)
  explosion    |impulse| * DRONE_INV_MASS / MAX_SPEED * EXPLOSION_ENERGY_REFILL_COEF
               impulse = (EXPLOSION_IMPULSE + speed) * 2 * DRONE_RADIUS
               at point blank, projectile moving at the drone       explodeCallback
  boost        a burst on the fire frame adds momentum of
               (DRONE_BURST_IMPACT_BASE * c + DRONE_BURST_IMPACT_MIN) * 2 * radius
               * burstBoostCoef for burst charge c                  explodeCallback
  black hole   |force| * dt * DRONE_INV_MASS / MAX_SPEED * BLACK_HOLE_PULL_ENERGY_REFILL_COEF
               force = (|BLACK_HOLE_PULL_MAGNITUDE| + speed) * 2 * DRONE_RADIUS * scale
               scale falls linearly to 0 at the proximity radius   handleBlackHolePull

Damping, shotgun fire jitter and hit angle are ignored. Per pickup and per
second numbers assume every projectile hits; per second uses cooldown plus
charge time, as charging only starts once the cooldown is over.
"""
import argparse
import math
import os
import re

WEAPONS = ['standard', 'machineGun', 'sniper', 'shotgun', 'imploder', 'accelerator',
           'flakCannon', 'mineLauncher', 'blackHole', 'nuke', 'grapple']
# mines are welded to a wall, so stationary, when they explode
STATIONARY_EXPLOSIONS = {'mineLauncher'}


class Settings:
    """Lazily evaluates #defines and static consts from a C header."""

    def __init__(self, paths):
        self.exprs = {}
        self.structs = {}
        for path in paths:
            with open(path) as f:
                src = re.sub(r'/\*.*?\*/', '', f.read(), flags=re.S)
            src = re.sub(r'//[^\n]*', '', src)
            for m in re.finditer(r'^#define\s+(\w+)\s+(.+)$', src, re.M):
                self.exprs[m.group(1)] = m.group(2)
            for m in re.finditer(r'static const \w+ (\w+) = ([^;{]+);', src):
                self.exprs[m.group(1)] = m.group(2)
            for m in re.finditer(r'static const weaponInformation (\w+) = \{(.*?)\};', src, re.S):
                fields = dict(re.findall(r'\.(\w+)\s*=\s*([^,]+),', m.group(2)))
                self.structs[m.group(1)] = fields
        self.cache = {}

    def eval(self, expr):
        expr = re.sub(r'(\d+\.?\d*)f\b', r'\1', expr.strip())
        env = {'MASS': lambda d, r: d * math.pi * r * r, 'INV_MASS': lambda m: 1 / m,
               'SQUARED': lambda x: x * x, 'true': True, 'false': False}
        for name in set(re.findall(r'\b[A-Za-z_]\w*\b', expr)) - env.keys():
            if name in self.exprs:
                env[name] = self[name]
        return eval(expr, env)

    def __getitem__(self, name):
        if name not in self.cache:
            self.cache[name] = self.eval(self.exprs[name])
        return self.cache[name]

    def get(self, name, default=None):
        return self[name] if name in self.exprs else default

    def weapon(self, name):
        # type is an enum value, keep its name to find the weapon's macros
        return {k: v.strip() if k == 'type' else self.eval(v) for k, v in self.structs[name].items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--settings', default=os.path.join(here, '..', 'settings.h'))
    args = parser.parse_args()
    settings_dir = os.path.dirname(os.path.abspath(args.settings))
    s = Settings([os.path.join(settings_dir, 'helpers.h'), args.settings])

    drone_inv_mass = s['DRONE_INV_MASS']
    drone_perimeter = 2 * s['DRONE_RADIUS']
    impulse_refill = drone_inv_mass / s['MAX_SPEED']
    explosion_refill = impulse_refill * s['EXPLOSION_ENERGY_REFILL_COEF']
    energy_max = s['DRONE_ENERGY_MAX']

    def burst_impulse(charge):
        return s['DRONE_BURST_IMPACT_BASE'] * charge + s['DRONE_BURST_IMPACT_MIN']

    def explosion(impulse, speed):
        # an implosion's speed is negated, so speed always strengthens it
        return min(abs(impulse + math.copysign(speed, impulse)) * drone_perimeter * explosion_refill, energy_max)

    min_charge = s['DRONE_BURST_BASE_COST']
    print(f'energy max {energy_max:g}, passive {s["DRONE_ENERGY_REFILL_RATE"]:g}/s '
          f'({energy_max / s["DRONE_ENERGY_REFILL_RATE"]:.0f}s empty to full), '
          f'respawn +{s["DRONE_ENERGY_RESPAWN_REFILL"]:g}, shield break +{s["DRONE_SHIELD_BREAK_ENERGY_REFILL"]:g}')
    print(f'explosion refill per unit impulse {explosion_refill:.6f}; '
          f'boost columns are burst charge {min_charge:g} (cost {min_charge:g}) and 1 (cost 1)\n')

    print('per hit (shotgun is all pellets)')
    print(f'{"weapon":<13} {"speed":>6} {"coef":>8} {"direct":>7} {"boost.1":>7} {"boost1":>7} '
          f'{"expl":>6} {"expl.1":>7} {"expl1":>6}')
    rows = []
    for name in WEAPONS:
        w = s.weapon(name)
        prefix = w['type'].removesuffix('_WEAPON')
        n = w['numProjectiles']
        mass = w['mass']
        momentum = w['fireMagnitude']
        coef = w['energyRefillCoef']

        def direct(charge):
            boost = burst_impulse(charge) * 2 * w['radius'] * w.get('burstBoostCoef', 1.0) if charge is not None else 0
            return (momentum + boost) * coef * n

        expl = [None] * 3
        if w['explosive']:
            impulse = s[f'{prefix}_EXPLOSION_IMPULSE']
            for i, charge in enumerate((None, min_charge, 1.0)):
                speed = 0.0
                if name not in STATIONARY_EXPLOSIONS:
                    boost = burst_impulse(charge) * 2 * w['radius'] * w.get('burstBoostCoef', 1.0) if charge is not None else 0
                    speed = (momentum + boost) / mass
                expl[i] = explosion(impulse, speed)

        fmt = lambda v: f'{v:.3f}' if v is not None else '-'
        print(f'{name:<13} {momentum / mass:6.1f} {coef:8.5f} {direct(None):7.3f} {direct(min_charge):7.3f} '
              f'{direct(1.0):7.3f} {fmt(expl[0]):>6} {fmt(expl[1]):>7} {fmt(expl[2]):>6}')
        # imploders and nukes refill on the direct hit and then explode
        per_hit = direct(None)
        if w['explosive']:
            per_hit = direct(None) + expl[0] if w['explodesOnDroneHit'] else max(direct(None), expl[0])
        per_hit = min(per_hit, energy_max)
        rows.append((name, w, prefix, per_hit))

    print('\nper pickup and sustained, unboosted, every shot hits')
    print(f'{"weapon":<13} {"ammo":>5} {"hit":>6} {"pickup":>7} {"cycle s":>8} {"per s":>6}')
    for name, w, prefix, per_hit in rows:
        ammo = s.get(f'{prefix}_AMMO')
        pickup = '-' if ammo in (None, s['INFINITE']) else f'{ammo * per_hit:.3f}'
        ammo_str = 'inf' if ammo == s['INFINITE'] else str(ammo)
        cycle = w['cooldown'] + w['maxCharge']
        print(f'{name:<13} {ammo_str:>5} {per_hit:6.3f} {pickup:>7} {cycle:8.3f} {per_hit / cycle:6.3f}')

    accel = s.weapon('accelerator')
    accel_speed = accel['fireMagnitude'] / accel['mass']
    bounce = s['ACCELERATOR_BOUNCE_SPEED_COEF']
    cap = s['ACCELERATOR_MAX_SPEED']
    to_cap = math.log(cap / accel_speed) / math.log(bounce)
    by_bounce = ', '.join(f'{b}: {min(accel_speed * bounce ** b, cap) * accel["mass"] * accel["energyRefillCoef"]:.3f}'
                          for b in (5, 10, 20, 30))
    print(f'\naccelerator direct hit by bounces {by_bounce}; '
          f'cap {cap * accel["mass"] * accel["energyRefillCoef"]:.3f} after {to_cap:.0f} bounces')

    pull_coef = s.get('BLACK_HOLE_PULL_ENERGY_REFILL_COEF')
    if pull_coef is not None:
        bh = s.weapon('blackHole')
        bh_speed = bh['fireMagnitude'] / bh['mass']
        force = (abs(s['BLACK_HOLE_PULL_MAGNITUDE']) + bh_speed) * drone_perimeter
        center = force * impulse_refill * pull_coef
        # scale = 1 - d/R averaged over a disc of radius R is 1/3
        print(f'black hole pull refill per second held: {center:.3f} at the center, '
              f'{center / 3:.3f} averaged over the pull radius')


if __name__ == '__main__':
    main()
