# Impulse Wars policy analysis

Plays a trained policy against every scripted bot personality on every map,
or against another trained policy, and compares runs: scores by opponent and
map, weapon value, burst boosting, deaths by source and per step behavior.

```bash
ocean/impulse_wars/analysis/build.sh               # -> build/iw_analyze
ocean/impulse_wars/analysis/run.sh hsp-5b base.jsonl
python ocean/impulse_wars/analysis/compare.py base=base.jsonl
```

## Analyzing a run

`analyze_run.sh` runs the standard set of analyses of a training run and
prints a summary:

```bash
ocean/impulse_wars/analysis/analyze_run.sh accelerator-rework
ocean/impulse_wars/analysis/analyze_run.sh accelerator-rework --baseline weapon-slots-10B
```

It builds the harness from the current tree into the results directory
(`build/iw_analysis/<run>/` by default), then plays the run's latest
checkpoint:

| Result | What | Time on 24 threads |
|---|---|---|
| `self.jsonl`, `weights.tsv` | self-play and first layer weights | ~1 min |
| `bots.jsonl` | every bot personality | ~2.5 min |
| `h2h_<steps>.jsonl` | head to head against its own checkpoints a third and two thirds of the way, to see whether it was still improving | ~2 min |
| `h2h_baseline.jsonl`, `baseline_self.jsonl`, `baseline_bots.jsonl` | with `--baseline`: head to head, and the baseline's self-play and bots on this build | ~5 min |
| `combos.jsonl` | the crate combo tournament | ~16 min |
| `traj/<steps>.jsonl` | self-play of 10 checkpoints spread on a log scale, for `trajectory.py` | ~10 min |

The summary (also written to `summary.txt`) has the head to head scores,
the bot ladder, self-play behavior (pace, weapons, slots, deaths, behavior,
standoffs, kill share by map), the combo tournament, the trajectory and the
weights. Results already in the directory are kept, so an interrupted
analysis resumes where it stopped and rerunning a finished one only prints
the summary again; delete a file to redo it. `--no-bots`, `--no-combos` and
`--traj 0` skip parts, `--ckpt` analyzes another checkpoint than the latest,
and `--baseline-size 512 4` gives the size of a baseline with no ini.

If the env has changed since the run was trained, build from a worktree at
the run's commit with `--src`:

```bash
git worktree add --detach ../iw-run <commit>
ocean/impulse_wars/analysis/analyze_run.sh <run> --src ../iw-run
```

### Comparing against earlier results

Keep results under `build/iw_analysis/` (it's gitignored but survives
checkouts) and any of them can be compared later: `compare.py` takes any
number of `name=games.jsonl` files and shows them side by side, the first
being the baseline.

```bash
B=build/iw_analysis
python ocean/impulse_wars/analysis/compare.py old=$B/run-a/self.jsonl new=$B/run-b/self.jsonl --sections pace,weapons,slots
python ocean/impulse_wars/analysis/trajectory.py $B/run-a/traj    # behavior over a run
```

- Games are deterministic: the same build, checkpoints and seeds replay the
  same games, so results from the same build pair up game by game and score
  differences get tight confidence intervals.
- Results from different env builds compare as means only: the seeds are
  the same but the games aren't. Bot scores in particular only compare on
  the same build, as bot or env changes shift them.
- To compare strength across an env change, play both checkpoints on the
  new build (`--baseline`): head to head and bots. Old checkpoints load as
  long as the obs layout hasn't changed, but they're out of distribution on
  the new env, so it's only a rough check.
- To get an old run's results back, build the harness from a worktree at
  the run's commit (`--src`) and rerun.

`trajectory.py` takes self-play results named by checkpoint step count
(`traj/<steps>.jsonl`, as `analyze_run.sh` writes them) and prints a row per
checkpoint: behavior (kills, swaps, context use, charging, holstered
charges, setup kills, line of sight, crates, standoffs), kills per game by
weapon and time in hand by weapon. `--csv` prints it all as one CSV.

`run.sh` takes a run name (latest checkpoint) or a checkpoint path, plus any
`iw_analyze` flags: `--episodes N` per opponent (default 1400, 100 per map),
`--maps 7,10`, `--seed S`, `--threads T`, or opponent names such as
`scripted turtle` to play only those. A full run is about 2 minutes on 12
cores; `--episodes 140` is a ~15 second smoke test.

Head to head against another policy:

```bash
ocean/impulse_wars/analysis/run.sh new-2b h2h.jsonl --opponent hsp-5b
ocean/impulse_wars/analysis/run.sh hsp-5b self.jsonl --opponent hsp-5b  # self-play
```

Both drones are agents, as in self-play training (so sudden death is on),
and every seed is played twice with the policies swapping drones, so spawn
advantages cancel out. `--opponent` takes a run name or checkpoint like the
policy; its size comes from its ini or `--opponent-hidden` and
`--opponent-layers`, and `--opponent-name` sets how it's labeled. In the
results `p` is always the first policy and `b` the opponent.

`compare.py` treats the first file as the baseline. Games are paired by
opponent, map and seed, so score differences are computed game by game,
which keeps confidence intervals tight. `--sections opp,map,grid,weapons,
boost,deaths,behavior` picks tables and `--side b` summarizes the opponent.
Three more sections:

- `pace`: game length, draws, line of sight, firing without line of sight,
  crate refills and kills. Long games, many draws and blind fire with little
  line of sight mean passive play, like combo drills farming crates.
- `mapweapons`: each weapon's share of kills on each map, with the map's
  featured weapon (its spawn weight override) marked.
- `standoff`: time with line of sight, blocked by a wall close by (under 25
  units) or blocked far away, split by whether the drone is behind, even or
  ahead on lives, with speed towards the enemy and trigger use; and
  standoffs (3 s or more blocked close by), how often they end with a peek
  and whether the peeker kills or dies within 3 s.

## Weapon slots and combos

The `slots` section of `compare.py` covers swaps (by weapon put away and
drawn), forced switches when ammo runs out, holster time and holstered
charges, charged draws, context button use (steps held, and presses:
since 1e168206 only a press acts) and how many did something, where pickups
went (top up, into hand, into the holster, or
replacing the weapon in hand) and what was dropped, and kills right after a
swap or with another weapon fired in the 3 s before.

A combo tournament plays every pair of pickup weapons as a starting
loadout against every other pair, `--episodes` games per matchup:

```bash
ocean/impulse_wars/analysis/run.sh ckpt.bin combos.jsonl --opponent ckpt.bin --combos --episodes 16
python ocean/impulse_wars/analysis/compare.py combos.jsonl
```

Pickups keep spawning, as in combo training episodes, so the kill shares
and `held%` (a loadout weapon in hand) show how long agents keep their
loadout; with normal pickups it's gone within seconds. `--no-pickups`
removes pickups, but weapons still disappear when their ammo runs out, so
drones end up on the standard gun. `--loadout-pickups` keeps pickups but
turns whatever a drone picks up into its missing loadout weapon at full
ammo, which keeps drones on their loadout all game. Both are out of
distribution for a policy trained with normal pickups. `--crates` plays
every game as a combo training episode, where every pickup is an ammo
crate that refills the picker's weapons; use it for policies trained with
crates.

Combos are measured by hits: `2>1%` is the share of weapon 1's kills made
within 2 s of weapon 2 hitting the enemy (or its grapple being attached to
it), and the slots section lists the most common setup and kill pairs.
Black hole pulls aren't hits, so its setups don't show up there.

## Observation ablations and weights

`--list-groups` lists the named obs groups. `--ablate g1,g2` makes the
policy act on observations with those groups zeroed, or with
`--ablate-mode swap` copied from another game in the batch, which keeps
features whose zero isn't a real state in distribution. Compare it to an
unablated run, best in self-play (`--opponent` the same checkpoint), as
the bots don't exercise most enemy state. `--shadow g` (repeatable, up to
16) instead lets a copy of the policy with its own recurrent state see the
ablated obs while the policy acts on the full ones, and records per step
KL of each action head and the value change; the `shadow` section shows
them. It includes drift of the shadow's recurrent state, so it measures
sustained reliance on the group, and costs a forward pass per group.

`--weights-out w.tsv` writes every first layer input's weight column norm
and the input's mean and std over the games; `python weights.py w.tsv`
summarizes them by group and flags large weights on low spread inputs.

## Testing a balance change

Build the harness against the changed env from a worktree, then play the
same checkpoint on both builds:

```bash
git worktree add ../iw-wave1 -b iw-wave1
# edit ../iw-wave1/ocean/impulse_wars/...
ocean/impulse_wars/analysis/build.sh ../iw-wave1 build/iw_analyze_wave1
ocean/impulse_wars/analysis/run.sh hsp-5b base.jsonl
IW_ANALYZE=build/iw_analyze_wave1 ocean/impulse_wars/analysis/run.sh hsp-5b wave1.jsonl
python ocean/impulse_wars/analysis/compare.py base=base.jsonl wave1=wave1.jsonl
```

This shows how a change affects the policy's current strategy, not what a
policy retrained on the change would do, so use it to filter changes before
retraining. Retrained checkpoints are compared the same way, each on the
build it was trained with. Changes the bots exercise on their own can also be
checked bot vs bot with `build_tournament.sh` and `iw_tournament`.

Notes:

- Games run in training mode: 10 actions per second, and against bots no
  sudden death and a draw when time runs out, the same as the bot ladder. Headless eval
  mode acts every frame at 120 fps, which a trained policy wasn't trained
  for, so it isn't supported.
- Actions are sampled, with a seeded rng per game, so runs are reproducible.
- Episode stats come from `droneStats` and `droneAnalysisStats` (the
  latter is kept out of the training logs); behavior stats such as line of
  sight, trigger usage and death wall distance are polled each step.
- The harness replaces puffercpu's conv and linear with an im2col conv and
  a batched linear (about 3x faster) and is built with fast math flags, so
  its results can differ from the reference kernels by float rounding.
