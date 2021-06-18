# pylint: skip-file

import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import seaborn as sns
from tqdm import tqdm

#%%
from domains.npuzzle import npuzzle
from domains.npuzzle import macros

puz = npuzzle.NPuzzle(15,(0,0))
print(puz)
start_blanks = [(i, j) for i in range(4) for j in range(4)]

macro_list = []
model_list = []
[macro_list.extend([op_list for op_list in macros.learned.macros[s]]) for s in start_blanks];
[model_list.extend([m_list for m_list in macros.learned.models[s]]) for s in start_blanks];

len(macro_list)
len(model_list)
macro, model = [(macro, model)
                for macro, model in zip(macro_list, model_list)
                if len(model[0]) == 3 and len(macro) == 10][0]

puz.apply_macro(model=model)
print(puz)

#%%
import numpy as np
from experiments import search
from domains.suitcaselock import SuitcaseLock, rank_mod2
data = []
for n in range(2,21):
    for k in range(1,n):
        A = (np.sum([np.eye(n,k=-i) for i in range(k)], axis=0) + np.sum([np.eye(n,k=n-i) for i in range(k)], axis=0)).astype(int)
        for i in range(n):
            if rank_mod2(A) == n:
                break
            A[i,i] = (A[i,i] + 1) % 2
        rank = rank_mod2(A)
        effective_k = np.mean(A)*n
        data.append({'N': n, 'k': k, 'rank': rank, 'p': effective_k})
data = pd.DataFrame(data)

sns.scatterplot(data=data,x='p',y='rank', hue='N')
ax = plt.gca()
ax.set_title('Maximum rank found vs. k for various N')
ax.legend(loc='lower right', ncol=5, framealpha=1,bbox_to_anchor=(1.0, -0.5))
ax.set_yticks(range(0,21))
plt.show()

#%%
from domains import cube
from domains.cube import formula, pattern
from domains.cube.macros import learned, expert

import random
random.seed(0)

formula.random_formula(3)
#%%

[print('{}: {}'.format(*entry)) for entry in ([(len(s),' '.join(s)) for s in expert.alg_formulas])]

#
c1 = cube.Cube()
macro = [macro for (macro, model) in zip(learned.macros, learned.models) if len(macro) == 12][0]
seq1 = formula.rotate(macro,cube.Face.D)
c1.apply(sequence=seq1).render()
' '.join(seq1)

#%%
c2 = cube.Cube()
macro = [macro for (macro, model) in zip(expert.macros, expert.models) if len(macro) == 8][-1]
seq2 = formula.inverse(formula.rotate(formula.rotate(macro,cube.Face.R),cube.Face.U,2))
c2.apply(sequence=seq2).render()
' '.join(seq2)
#%%

print(len(c2.summarize_effects()))
print(len(c2.summarize_effects(baseline=c1)))

c2.apply(sequence="R' F".split()).render()

print(len(c2.summarize_effects(baseline=c1)))

#%%
n_values = 4
data = []
for N in tqdm(range(1,21)):
    for k in range(1,N):
        max_rank = 0
        effective_p = 0
        n_full_rank = 0
        if k == 1:
            M = np.eye(N)
            effective_p = 1/N
            max_rank = N
        elif k == N-1:
            M = np.ones((N,N))-np.eye(N)
            effective_p = (N-1)/N
            max_rank = N
        else:
            for i in range(1000):
                M = np.random.choice([0,1], size=(N,N), p=[(1-k/N),k/N])
                rank = rank_mod2(M)
                max_rank = max(max_rank, rank)
                if rank == max_rank:
                    n_full_rank += 1
                    effective_p += np.mean(M)
                if rank == N:
                    break
            effective_p /= n_full_rank
        data.append({'N': N, 'k': k, 'rank': max_rank, 'p': effective_p})
data = pd.DataFrame(data)

#%%
sns.pointplot(data=data,x='k',y='rank', hue='N', height=12)
ax = plt.gca()
ax.set_title('Maximum rank found vs. k for various N')
ax.legend(loc='lower right', ncol=5, framealpha=1,bbox_to_anchor=(1.0, -0.5))
ax.set_yticks(range(0,21))
plt.show()

#%%
sns.pointplot(data=data,x='k',y='p', hue='N', height=12)
ax = plt.gca()
ax.set_title('Average entanglement found vs. k for various N')
ax.legend(loc='lower right', ncol=5, framealpha=1,bbox_to_anchor=(1.0, -0.5))
ax.set_ylabel('Entanglement')
plt.show()


#%%
for k in range(1,10,2):
    name = 'ss_10_k{k}'.format(k=k)
    cmd = 'python3 -m experiments.suitcaselock.solve --n_vars=10 --n_values=4 --max_transitions=1e8 --entanglement={k} -s'.format(k=k)
    launcher = './slurm/run.py --command="{cmd}" --jobname {name} --duration vlong -t 1 -n 100 --mem={mem}'
    def wrap_cmd(launcher, cmd): return launcher.format(cmd=cmd, name=name, mem=2*k)
    print(wrap_cmd(launcher, cmd))
