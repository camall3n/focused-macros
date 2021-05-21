# Efficient Black-Box Planning Using Macro-Actions with Focused Effects
http://cs.brown.edu/~gdk/pubs/focused_macros.pdf

### Installation

Download the repo:
```
git clone https://github.com/camall3n/focused-macros.git
```

Install the dependencies:
```
cd skills-for-planning
git submodule init
git submodule update
python3 -m venv venv
. venv/bin/activate
pip install -r requirements.txt
```

## Experiments
### SuitcaseLock
Analyze heuristic:
```
# for K in 1..9:
# for SEED in 1..10:
python3 -m experiments.heuristic.heuristic_vs_cost -n 10 -v 2 -k K -s SEED

# for K in 1..4:
# for SEED in 1..10:
python3 -m experiments.heuristic.heuristic_vs_cost -n 5 -v 4 -k K -s SEED
```

Plot results:
```
python3 -m experiments.heuristic.plot_accuracy -n 10 -v 2
python3 -m experiments.heuristic.plot_accuracy -n 5 -v 4
```

Solve SuitcaseLock:
```
# for K in [1,4,7,10,13,16,19]:
# for SEED in 1..100:
python3 -m experiments.suitcaselock.solve --n_vars=20 --n_values=2 --max_transitions=1e8 --entanglement=K -s SEED

# for K in [1,3,5,7,9]:
# for SEED in 1..100:
python3 -m experiments.suitcaselock.solve --n_vars=10 --n_values=4 --max_transitions=1e8 --entanglement=K -s SEED
```

Plot results:
```
python3 -m experiments.plot_planning_time suitcaselock
```

### PDDLGym
Search for PDDLGym macro-actions:
```
# for ENV_NAME, MACRO_BUDGET in [
#    ('depot',         50000),
#    ('doors',          5000),
#    ('ferry',          5000),
#    ('gripper',        5000),
#    ('hanoi',        100000),
#    ('miconic',        5000),
# ]:
python3 -m experiments.pddlgym.macro_search --env_name ENV_NAME --max_transitions MACRO_BUDGET

python3 -m experiments.pddlgym.macro_cleanup --env_name ENV_NAME -n 8
```

Solve PDDLGym domains:
```
# for ENV_NAME, BFWS_PREC in [
#    ('depot',          2),
#    ('doors',          3),
#    ('ferry',          3),
#    ('gripper',        3),
#    ('hanoi',          3),
#    ('miconic',        3),
# ]:
# for ALG in ['gbfs', 'bfws_rg']:
# for MACRO_TYPE in ['primitive', 'learned']:
# for SEED in 1..100:
python3 -m experiments.pddlgym.solve --max_transitions 1e5 -m MACRO_TYPE --env_name ENV_NAME  --search_alg ALG --bfws_precision BFWS_PREC -s SEED
```

Plot PDDLGym results:
```
python3 -m experiments.plot_planning_time pddlgym --pddl_env ENV_NAME --alg ALG
```


### 15-Puzzle
Search for 15-Puzzle macro-actions:
```
# for ROW in 0..3:
# for COL in 0..3:
python3 -m experiments.npuzzle.macro_search -r ROW -c COL

python3 -m experiments.npuzzle.macro_cleanup
```

Visualize learned 15-puzzle macros:
```
python3 -m experiments.npuzzle.plot_entanglement
```

Solve 15-puzzle:
```
# for ALG in ['gbfs', 'bfws_rg']:
# for SEED in 1..100:
python3 -m experiments.npuzzle.solve --max_transitions=1e6 --search_alg ALG -m primitive -s SEED
python3 -m experiments.npuzzle.solve --max_transitions=1e6 --search_alg ALG -m random -s SEED
python3 -m experiments.npuzzle.solve --max_transitions=1e6 --search_alg ALG -m learned -s SEED

python3 -m experiments.npuzzle.solve --max_transitions=1e6 --search_alg ALG -m learned --random_goal -s SEED
```

Plot 15-puzzle results:
```
python3 -m experiments.plot_planning_time npuzzle --alg ALG
```


### Rubik's Cube
Search for Rubik's cube macro-actions:
```
python3 -m experiments.cube.macro_search
python3 -m experiments.cube.macro_cleanup
```

Visualize Rubik's cube macro-actions:
```
python3 -m experiments.cube.plot_entanglement
```

Solve Rubik's cube:
```
# for ALG in ['gbfs', 'bfws_rg']:
# for SEED in 1..100:
python3 -m experiments.cube.solve --buchner2018 --max_transitions=2e6 --search_alg ALG --bfws_precision 2 -m primitive -s SEED
python3 -m experiments.cube.solve --buchner2018 --max_transitions=2e6 --search_alg ALG --bfws_precision 2 -m random -s SEED
python3 -m experiments.cube.solve --buchner2018 --max_transitions=2e6 --search_alg ALG --bfws_precision 2 -m learned -s SEED
python3 -m experiments.cube.solve --buchner2018 --max_transitions=2e6 --search_alg ALG --bfws_precision 2 -m expert -s SEED

python3 -m experiments.cube.solve --buchner2018 --max_transitions=2e6 --search_alg ALG --bfws_precision 2 -m learned --random_goal -s SEED
```

Plot Rubik's cube results:
```
python3 -m experiments.plot_planning_time cube-buchner2018 --alg ALG
```
