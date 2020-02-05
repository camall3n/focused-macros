# I pickled some objects before changing the directory structure.
#   notebooks.astar -> notebooks.search -> experiments.search
#   cube -> domains.cube
#   npuzzle -> domains.npuzzle
#   suitcaselock ->domains.suitcaselock
#
# This changed several package names and the pickled objects wouldn't load.
#
# So if pickle.load complains, import this module.

import sys

import experiments.search
import domains.cube
import domains.npuzzle
import domains.suitcaselock

sys.modules['notebooks.astar'] = experiments.search
sys.modules['notebooks.search'] = experiments.search
sys.modules['experiments.search'].Node = experiments.search.SearchNode
sys.modules['cube'] = domains.cube
sys.modules['npuzzle'] = domains.npuzzle
sys.modules['suitcaselock'] = domains.suitcaselock
