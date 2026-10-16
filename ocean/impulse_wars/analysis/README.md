# Impulse Wars policy analysis

Plays a trained policy against every scripted bot personality on every map,
or against another trained policy, and compares runs: scores by opponent and
map, weapon value, burst boosting, deaths by source and per step behavior.

```bash
ocean/impulse_wars/analysis/build.sh               # -> build/iw_analyze
ocean/impulse_wars/analysis/run.sh hsp-5b base.jsonl
python ocean/impulse_wars/analysis/compare.py base=base.jsonl
```

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
- The harness is built with `PUF_CPU_BLOCKED_KERNELS` and fast math flags,
  which make CPU inference about 5x faster; its results differ from the
  reference kernels by float rounding.
