# I pickled the search results while there was a notebooks.astar
# but subsequently changed the module name to notebooks.search,
# which meant I couldn't load the pickle objects.
#
# So if pickle.load complains about notebooks.astar, use this.

from notebooks.search import *
import notebooks.search
import sys
sys.modules['notebooks.astar'] = notebooks.search
