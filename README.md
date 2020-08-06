# Finding Disentangled Macro-Actions for Efficient Planning with the Goal-Count Heuristic
https://arxiv.org/abs/2004.13242

### Installation

Download the repo:
```
git clone https://github.com/camall3n/skills-for-planning.git
```

Install the dependencies:
```
cd skills-for-planning
python3 -m venv env
. env/bin/activate
pip install -r requirements.txt
```

## Experiments
### SuitcaseLock
Solve SuitcaseLock:
```
python3 -m experiments.suitcaselock.solve --n_vars=20 --n_values=2 --max_transitions=1e8 --entanglement=1 -s SEED
python3 -m experiments.suitcaselock.solve --n_vars=20 --n_values=2 --max_transitions=1e8 --entanglement=4 -s SEED
python3 -m experiments.suitcaselock.solve --n_vars=20 --n_values=2 --max_transitions=1e8 --entanglement=7 -s SEED
python3 -m experiments.suitcaselock.solve --n_vars=20 --n_values=2 --max_transitions=1e8 --entanglement=10 -s SEED
python3 -m experiments.suitcaselock.solve --n_vars=20 --n_values=2 --max_transitions=1e8 --entanglement=13 -s SEED
python3 -m experiments.suitcaselock.solve --n_vars=20 --n_values=2 --max_transitions=1e8 --entanglement=16 -s SEED
python3 -m experiments.suitcaselock.solve --n_vars=20 --n_values=2 --max_transitions=1e8 --entanglement=19 -s SEED

python3 -m experiments.suitcaselock.solve --n_vars=10 --n_values=4 --max_transitions=1e8 --entanglement=1 -s SEED
python3 -m experiments.suitcaselock.solve --n_vars=10 --n_values=4 --max_transitions=1e8 --entanglement=3 -s SEED
python3 -m experiments.suitcaselock.solve --n_vars=10 --n_values=4 --max_transitions=1e8 --entanglement=5 -s SEED
python3 -m experiments.suitcaselock.solve --n_vars=10 --n_values=4 --max_transitions=1e8 --entanglement=7 -s SEED
python3 -m experiments.suitcaselock.solve --n_vars=10 --n_values=4 --max_transitions=1e8 --entanglement=9 -s SEED
```

Plot results:
```
python3 -m experiments.plot_planning_time suitcaselock
```


### 15-Puzzle
Search for 15-Puzzle macro-actions:
```
python3 -m experiments.npuzzle.macro_search -r 0 -c 0
python3 -m experiments.npuzzle.macro_search -r 0 -c 1
python3 -m experiments.npuzzle.macro_search -r 0 -c 2
python3 -m experiments.npuzzle.macro_search -r 0 -c 3
python3 -m experiments.npuzzle.macro_search -r 1 -c 0
python3 -m experiments.npuzzle.macro_search -r 1 -c 1
python3 -m experiments.npuzzle.macro_search -r 1 -c 2
python3 -m experiments.npuzzle.macro_search -r 1 -c 3
python3 -m experiments.npuzzle.macro_search -r 2 -c 0
python3 -m experiments.npuzzle.macro_search -r 2 -c 1
python3 -m experiments.npuzzle.macro_search -r 2 -c 2
python3 -m experiments.npuzzle.macro_search -r 2 -c 3
python3 -m experiments.npuzzle.macro_search -r 3 -c 0
python3 -m experiments.npuzzle.macro_search -r 3 -c 1
python3 -m experiments.npuzzle.macro_search -r 3 -c 2
python3 -m experiments.npuzzle.macro_search -r 3 -c 3

python3 -m experiments.npuzzle.macro_cleanup
```

Visualize learned 15-puzzle macros:
```
python3 -m experiments.npuzzle.plot_entanglement
```

Solve 15-puzzle:
```
python3 -m experiments.npuzzle.solve -m primitive --max_transitions=1e6 -s SEED
python3 -m experiments.npuzzle.solve -m random --max_transitions=1e6 -s SEED
python3 -m experiments.npuzzle.solve -m learned --max_transitions=1e6 -s SEED

python3 -m experiments.npuzzle.solve -m learned --max_transitions=1e6 --random_goal -s SEED
```

Plot 15-puzzle results:
```
python3 -m experiments.plot_planning_time npuzzle
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
python3 -m experiments.cube.solve -m primitive --max_transitions=2e6 -s SEED
python3 -m experiments.cube.solve -m random --max_transitions=2e6 -s SEED
python3 -m experiments.cube.solve -m learned --max_transitions=2e6 -s SEED
python3 -m experiments.cube.solve -m expert --max_transitions=2e6 -s SEED

python3 -m experiments.cube.solve -m learned --max_transitions=2e6 --random_goal -s SEED
```

Plot Rubik's cube results:
```
python3 -m experiments.plot_planning_time cube
```
