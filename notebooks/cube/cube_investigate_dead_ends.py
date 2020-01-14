import copy
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import seaborn as sns

from cube import cube
from cube import pattern
from cube import formula
from cube import skills

results_dir = 'results/cube_deadends/'
cube_files = sorted(glob.glob(results_dir+'*.pickle'))
for f in cube_files:
    seed = int(f.split('/')[-1].split('.')[-2].split('-')[-1])
    with open(f, 'rb') as f:
        cubefail = pickle.load(f)
    if seed in [8]:#[1,8,14,35,53,76,100]:
        print(seed)
        cubefail.render()
        break
    pass

#%%
cube_mod = copy.deepcopy(cubefail)
rot = formula.rotate

cube_mod.apply("R R".split())
cube_mod.apply(rot(rot("F F R' F' U' F' U F R F' U U F U U F' U'".split(),cube.Face.L,2),cube.Face.D,2))
cube_mod.apply("R R".split())

# cube_mod.apply("D' F".split())
# cube_mod.apply(rot("F F R' F' U' F' U F R F' U U F U U F' U'".split(),cube.Face.U))
# cube_mod.apply(formula.inverse(rot(skills.orient_2_corners,cube.Face.U)))
# cube_mod.apply("F' D".split())

cube_mod.render()

#%%
newcube = cube.Cube()
newcube.apply(rot(rot("F F R' F' U' F' U F R F' U U F U U F' U'".split(),cube.Face.L,0),cube.Face.D,0))
newcube.render()
# len(newcube.summarize_effects())
