from experiments.search import best_first_search
from experiments.width import WidthAugmentedHeuristic
from experiments.iw import get_unique_atoms

class BFWSPriority:
    def __call__(self, node):
        width, goalcount = node.h_score
        return (width, goalcount, node.g_score)

def bfws(start, *args, heuristic=None, R=set([]), precision=3, **kwargs):
    """BFWS(R) - best-first width search with relevant atoms

    Args:
        heuristic (callable):
            The state->heuristic function to wrap with the BFWS width augmentation
        R (set):
            The set of relevant atoms to use for #r, default is R = R0
    """
    width_aug_heuristic = WidthAugmentedHeuristic(
                              n_variables=len(start),
                              heuristic=heuristic,
                              R=R,
                              precision=precision
                          )
    return best_first_search(start, *args, heuristic=width_aug_heuristic,
                             get_priority=BFWSPriority(), **kwargs)
