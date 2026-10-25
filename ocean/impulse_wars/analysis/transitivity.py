"""Transitivity analysis of a round robin (round_robin.sh): score matrix, Elo
fit, Hodge split of the logit matrix into transitive and cyclic parts,
residuals of the Elo fit and significant 3-cycles.

Usage: python transitivity.py <round robin dir>

Reads every <a>_vs_<b>.jsonl in the dir. Scores are averaged over both seats
of each map and seed, so spawn advantages cancel. A transitive game is one
where a single rating per policy predicts every head to head; the cyclic
part is what no rating can explain (rock paper scissors). Compare it with
the noise floor: with finite games some cyclic part shows up even in a
perfectly transitive game.
"""
import glob, json, math, os, re, sys, itertools, collections, statistics

d = sys.argv[1]
games = {}
for f in glob.glob(os.path.join(d, '*_vs_*.jsonl')):
    a, b = os.path.basename(f)[:-6].split('_vs_')
    units = collections.defaultdict(list)
    for line in open(f):
        g = json.loads(line)
        units[(g['map'], g['seed'])].append(g['p']['score'])
    games[(a, b)] = [sum(v) / len(v) for v in units.values()]

natural = lambda s: [float(t) if re.fullmatch(r'[0-9.]+', t) else t for t in re.split(r'([0-9.]+)', s) if t]
names = sorted({x for k in games for x in k}, key=lambda s: [(0, t) if isinstance(t, float) else (1, t) for t in natural(s)])
n = len(names)
idx = {s: i for i, s in enumerate(names)}
P = [[0.5] * n for _ in range(n)]
SE = [[0.0] * n for _ in range(n)]
for (a, b), u in games.items():
    i, j = idx[a], idx[b]
    m = statistics.mean(u)
    se = statistics.stdev(u) / math.sqrt(len(u))
    P[i][j], P[j][i] = m, 1 - m
    SE[i][j] = SE[j][i] = se
pairs = list(itertools.combinations(range(n), 2))
missing = [(i, j) for i, j in pairs if (names[i], names[j]) not in games and (names[j], names[i]) not in games]
print(f'{len(games)} pairs, {len(missing)} missing')

w = max(len(s) for s in names)
print('\nscore of row vs column')
print(' ' * (w + 1) + ' '.join(f'{i:>6}' for i in range(n)) + '    mean')
for i, s in enumerate(names):
    row = ' '.join('     -' if i == j else f'{P[i][j]:6.3f}' for j in range(n))
    print(f'{s:>{w - 3}} {i:>2} {row}  {statistics.mean(P[i][j] for j in range(n) if j != i):.3f}')
print(f'typical 95% CI ±{1.96 * statistics.median(SE[i][j] for i, j in pairs):.3f}')

# Hodge decomposition on the complete graph: A = logit P is antisymmetric;
# its gradient part r_i - r_j with r_i = mean_j A_ij is the best transitive
# (Elo) fit, the rest is the cyclic part
eps = 1e-3
clip = lambda x: min(max(x, eps), 1 - eps)
A = [[0.0 if i == j else math.log(clip(P[i][j]) / (1 - clip(P[i][j]))) for j in range(n)] for i in range(n)]
r = [sum(A[i]) / n for i in range(n)]
G = [[r[i] - r[j] for j in range(n)] for i in range(n)]
tot = sum(A[i][j] ** 2 for i, j in pairs)
cyc = sum((A[i][j] - G[i][j]) ** 2 for i, j in pairs)
# noise floor: logit SE by the delta method, times the residual dof share
noise = sum((SE[i][j] / (clip(P[i][j]) * (1 - clip(P[i][j])))) ** 2 for i, j in pairs)
dof = len(pairs) - (n - 1)
print(f'\nHodge split of the logit matrix: transitive {1 - cyc / tot:.3f}, cyclic {cyc / tot:.3f}')
print(f'cyclic sum of squares {cyc:.3f} vs ~{noise * dof / len(pairs):.3f} expected from sampling noise alone (dof {dof})')

print('\nElo ratings (400 * log10 scale, mean 0)')
elo = [x * 400 / math.log(10) for x in r]
for i in sorted(range(n), key=lambda i: -elo[i]):
    print(f'  {names[i]:>{w}} {elo[i]:7.0f}')

# residuals of the Elo fit in score space
fit = lambda i, j: 1 / (1 + math.exp(-G[i][j]))
res = sorted(((P[i][j] - fit(i, j)) / SE[i][j], i, j) for i, j in pairs)
res.sort(key=lambda x: -abs(x[0]))
print('\nlargest residuals vs the Elo fit (observed score of row vs column, Elo predicted, z)')
for z, i, j in res[:10]:
    print(f'  {names[i]:>{w}} vs {names[j]:<{w}} obs {P[i][j]:.3f} fit {fit(i, j):.3f} z {z:+.1f}')
print(f'pairs with |z| > 2: {sum(abs(z) > 2 for z, _, _ in res)} of {len(res)} (about {0.046 * len(res):.1f} expected by chance)')
print(f'sum of z^2 {sum(z * z for z, _, _ in res):.1f} vs dof {dof}')

# 3-cycles: each unordered triple is checked in both directions
print('\n3-cycles (a beats b beats c beats a), each edge > 0.5 by 2 SE')
found = weak = 0
for a, b, c in itertools.combinations(range(n), 3):
    for x, y, z in [(a, b, c), (a, c, b)]:
        edges = [(x, y), (y, z), (z, x)]
        if all(P[p][q] > 0.5 for p, q in edges):
            weak += 1
        if all(P[p][q] - 0.5 > 2 * SE[p][q] for p, q in edges):
            print(f'  {names[x]} > {names[y]} > {names[z]} > {names[x]}: ' + ' '.join(f'{P[p][q]:.3f}' for p, q in edges))
            found += 1
print(f'  {found} found')
print(f'3-cycles by point estimate alone: {weak} of {math.comb(n, 3)} triples')
