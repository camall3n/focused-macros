import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import seaborn as sns
from tqdm import tqdm

from npuzzle import npuzzle
from npuzzle import options

skill_len = list(map(len,options.generated.options[(0,0)]))
skill_size = list(map(lambda x: len(x[0]),options.generated.models[(0,0)]))

noise = 0.08
x = np.asarray(skill_len)+np.random.normal(0,noise,len(skill_len))
y = np.asarray(skill_size)+np.random.normal(0,noise,len(skill_size))
plt.grid('on')
plt.scatter(x,y)
plt.xlabel('number of primitive actions per skill')
plt.ylabel('number of variables modified')
plt.xlim([0,20])
plt.ylim([0,5])
plt.xticks(range(21))
plt.gca().set_axisbelow(True)
plt.title('Skill effect size vs. skill length (15-puzzle)')
plt.savefig('results/plots/npuzzle_skill_effect_size_vs_length.png')
plt.show()
