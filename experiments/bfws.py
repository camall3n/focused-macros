from experiments.search import gbfs
from experiments.width import WidthAugmentedHeuristic
from experiments.iw import get_unique_atoms

def bfws(start, *args, heuristic=None, precision=3, **kwargs):
    """BFWS - best-first width search

    Args:
        heuristic (callable):
            The state->heuristic function to wrap with the BFWS width augmentation
        precision (int):
            The number of width values, w \in {1,...,P}, to use with BFWS
    """
    original_heuristic = heuristic
    heuristic = WidthAugmentedHeuristic(n_variables=len(start),
                                    heuristic=original_heuristic,
                                    precision=precision)
    return gbfs(start, *args, heuristic=heuristic, **kwargs)

def bfwsr(start, *args, heuristic=None, R=set([]), precision=3, **kwargs):
    """BFWS(R) - best-first width search with relevant atoms

    Args:
        heuristic (callable):
            The state->heuristic function to wrap with the BFWS width augmentation
        R (set):
            The set of relevant atoms to use for #r
    """
    orig_heuristic = heuristic
    heuristic = WidthAugmentedHeuristic(n_variables=len(start),
                                    heuristic=orig_heuristic,
                                    R=R,
                                    precision=precision)
    return gbfs(start, *args, heuristic=heuristic, **kwargs)

