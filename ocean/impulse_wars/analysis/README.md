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
| `combos.jsonl` | the crate weapon + utility loadout tournament | ~8 min |
| `traj/<steps>.jsonl` | self-play of 10 checkpoints spread on a log scale, for `trajectory.py` | ~10 min |

The summary (also written to `summary.txt`) has the head to head scores,
the bot ladder, self-play behavior (pace, weapons, pickups, utilities,
deaths, behavior, standoffs, kill share by map), the loadout tournament, the
trajectory and the
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

### GPU inference

The harness spends most of its time in the policy forward. On a machine
with nvcc, the forward can run on the GPU with the trainer's CUDA net, while
the envs, bots and stats stay on the CPU threads:

```bash
ocean/impulse_wars/analysis/build.sh --gpu          # -> build/iw_analyze_gpu, bf16
ocean/impulse_wars/analysis/build.sh --gpu --fp32   # -> build/iw_analyze_gpu_fp32
ocean/impulse_wars/analysis/run.sh hsp-5b bots.jsonl --gpu
ocean/impulse_wars/analysis/analyze_run.sh accelerator-rework --gpu
```

A GPU build still runs the forward on the CPU without `--gpu`, and
`--gpu 1` picks another device. With `--gpu` each thread plays `--batch`
games (default 64) through one CUDA graph per net per step. On 24 threads
the full bot ladder (9800 games) took 288s on the CPU, 23s on the GPU in
fp32 and 15s in bf16, with bot scores within noise of each other, so the
GPU build is bf16 by default, like training.

- GPU games only pair game by game with results of the same build and
  precision, `--batch` and GPU model. GPU and CPU results compare as means
  only, like results from different env builds. `analyze_run.sh` refuses
  to mix them in one results directory.
- `build.sh --gpu` takes the CUDA net from `src_root`, like the env, so
  `--src` worktrees work as on the CPU.
- `gpu_check.sh <run>` builds all three variants, plays the bot ladder with
  each and prints their times and scores side by side. It also replays some
  games with other `--threads`, which should give identical games. Use it to
  check the GPU build after changing the net.

### 4 drone builds

The harness plays the drone count it's built with: 2 by default, or 4 with
`IW_DRONES=4`, as for the trainer's `build.sh` (`IW_DRONES=4
ocean/impulse_wars/analysis/build.sh`, `IW_DRONES=4 analyze_run.sh <run>`,
and `build_tournament.sh out -DIW_NUM_DRONES=4`). With 4 drones it plays
2v2: the policy
plays a team, team 0 against the bots, and against another policy every
seed is played from both teams. `--ffa` plays free for all instead: the
policy plays one drone against three bots, or against three copies of the
opponent from each of the 4 seats, as in training. `analyze_run.sh` picks
`--ffa` for runs whose ini has `enable_teams = 0`.

- `p` and `b` are summed over each side's drones, and a side's score is its
  best drone's, so p and b scores still add up to 1. In free for all the
  opponent side is three drones, so its totals aren't comparable to the
  policy's, and a random policy scores about 0.25.
- Stats between drones (line of sight, separation, aim error, situations)
  are to each drone's nearest living enemy. Standoffs and peeks are only
  tracked in 1v1.
- The other drones' obs groups and features are named by slot: `ally`, then
  `enemy0`, `enemy1` in 2v2 and `enemy0`..`enemy2` in free for all
  (`--list-groups`). 1v1 keeps the `enemy` names.
- `bot_tournament.c` plays 2v2 and `--ffa` the same way.
- Results only compare to runs with the same drone count and mode.

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
checkpoint: behavior (kills, charging, setup kills, line of sight, crates,
utility presses and kills, standoffs, and swaps, context use and holstered
charges for results from before the utility slot), kills per game by weapon
and time in hand by weapon. `--csv` prints it all as one CSV.

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
advantages cancel out. The two starting utility pickups, one nearest each
drone, cycle over every pair of utilities across seeds and maps, so every
utility is played about as often. `--opponent` takes a run name or checkpoint like the
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
  featured weapon (its spawn weight override) marked. Utilities kill as the
  weapons their projectiles are (grapple, mine), so this is also the
  utilities' kill share by map; siege and moat feature them.
- `standoff`: time with line of sight, blocked by a wall close by (under 25
  units) or blocked far away, split by whether the drone is behind, even or
  ahead on lives, with speed towards the enemy and trigger use; and
  standoffs (3 s or more blocked close by), how often they end with a peek
  and whether the peeker kills or dies within 3 s.

## Projectiles

The `projectiles` section of `compare.py` shows what a projectile obs would
have to cover (the policy has none; projectiles show only through their
effects): live projectiles per step of each drone (all and within 30
units), threats (enemy projectiles whose closest approach at the current
relative velocity comes within hitting or blast range within 2 s), their
distance, relative speed and time to closest approach, how many threats
would be left out of the obs with K slots sorted by distance, threats and
hits taken by weapon (shotgun pellets and sniper shots often hit the step
they spawn, before they could be seen), the same counts by the weapon the
enemy holds, and steps within blast range of an own explosive.

## Transitivity

`round_robin.sh` plays every pair of a set of checkpoints head to head
and `transitivity.py` checks whether one rating per policy explains the
results, or whether some policies counter others (rock paper scissors):

```bash
C=checkpoints/impulse_wars
ocean/impulse_wars/analysis/round_robin.sh build/iw_analysis/rr-long \
    $C/run-a/0000002000000000.bin $C/run-a/0000008000000000.bin \
    b8=$C/run-b/0000008000000000.bin
```

The summary has the score matrix, Elo ratings, the share of the logit
score matrix that is transitive (an Elo fit) vs cyclic with the cyclic
share expected from sampling noise alone, the pairs the Elo fit predicts
worst, and the 3-cycles where every edge is significant. Checkpoints of one
run are biased towards looking transitive, as each was trained against its
predecessors; late checkpoints of different runs are the more telling
pairs. 10 checkpoints are 45 pairs, about an hour on 24 threads at the
default 1400 games per pair; adding a checkpoint later only plays its
pairs.

## Weapons, pickups and utilities

The `slots` section of `compare.py` covers time with only the standard gun,
switches when ammo runs out, where pickups went (top up, into hand
replacing the standard gun, or replacing a pickup weapon) and what was
dropped, and kills with another weapon fired or hitting in the 3 s or 2 s
before. Utilities count as the weapons their projectiles are here, so a
mine hit followed by a kill is a setup kill. For results from before the
utility slot it shows the weapon slot stats instead: swaps, holster time
and charges, charged draws and context button use.

The `utilities` section has, per utility:

- time held and time with ammo to use it, pickups;
- presses and how many did something (fired, started reeling an attached
  grapple or detonated mines), ammo spent;
- hits on drones, kills, assists and self kills, kills per ammo, and kills
  of this drone by the enemy's;
- for the grapple: attaches by target, time attached and reeling, reeling
  speed against the mean speed, deaths while attached and kills by its pull;
- for mines: detonations, kills and self kills.

Utility hits and firing are kept out of the shot stats, as hitting the
enemy often isn't the point of a utility.

A loadout tournament plays every loadout of a pickup weapon and a utility
against every other one and itself, `--episodes` games per matchup:

```bash
ocean/impulse_wars/analysis/run.sh ckpt.bin combos.jsonl --opponent ckpt.bin --combos --episodes 64
python ocean/impulse_wars/analysis/compare.py combos.jsonl
```

It shows the score of loadouts with each utility and each weapon, utility
against utility, and every weapon + utility loadout with the share of its
kills made by the utility. Pickups keep spawning, as in combo training
episodes, so drones pick up other weapons and utilities. `--no-pickups`
removes pickups, but weapons still disappear when their ammo runs out, so
drones end up on the standard gun. `--loadout-pickups` keeps pickups but
turns whatever a drone picks up into its loadout weapon or utility at full
ammo, which keeps drones on their loadout all game. Both are out of
distribution for a policy trained with normal pickups. `--crates` plays
every game as a combo training episode, where every pickup is an ammo
crate that refills the picker's weapon and utility; use it for policies
trained with crates.

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
  an AVX2 matrix multiply over weights packed once and shared by all
  workers, and the record branches only run active records. It's built with
  fast math flags, so its results can differ from the reference kernels by
  float rounding. Every output is computed the same way wherever its game
  is in a batch, so results don't depend on `--threads` or `--batch`.
  Results from before the AVX2 kernels (Oct 23 2026) compare as means only.
- `--batch` defaults to 6, the fastest on 24 threads (a 12900K: 1.9x the
  old kernels). Larger batches are faster on few threads, e.g. 16 on one.
