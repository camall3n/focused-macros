from collections import defaultdict

from tqdm import tqdm

import experiments.priorityqueue as pq

class SearchNode:
    """A single graph search node for forward state-space planning

    Args:
        state:
            The associated state information for this node
        g_score:
            The cost of reaching the node
        h_score:
            The heuristic score from the node to the goal
        gh_weights (tuple, optional):
            Tuple of weights (g_weight, h_weight) for weighted A*
        parent (SearchNode, optional):
            The parent of the node, if there is one
        action (optional):
            The action that transitioned from parent to this node, if there was one
    """
    def __init__(self, state, g_score, h_score, gh_weights=(1,1), parent=None, action=None):
        self.state = state
        self.action = action
        self.g_score = g_score
        self.h_score = h_score
        self.parent = parent
        self.gh_weights = gh_weights
    def __cmp__(self, other):
        return 0
    def __eq__(self, other):
        return True
    @property
    def f_score(self):
        """Compute and return the estimated (weighted) path cost for this node"""
        return self.gh_weights[0]*self.g_score + self.gh_weights[1]*self.h_score

def reconstruct_path(node):
    """Iteratively reconstruct the search path by working backwards from the specified node"""
    states = [node.state]
    actions = []
    while node.parent:
        states.append(node.parent.state)
        actions.append(node.action)
        node = node.parent
    return list(reversed(states)), list(reversed(actions))

def _weighted_astar(start, is_goal, step_cost, heuristic, get_successors,
                    max_transitions=0, save_best_n=1, quiet=False, gh_weights=(1,1)):
    """Core implementation of weighted A*

    Weighted A* is a general search algorithm that performs forward best-first search
    on a graph by expanding nodes according to their estimated total path cost.

    The following algorithms are special cases:
        - Greedy, Best-First Search (GBFS), if gh_weights = (0,1)
        - Dijkstra's Algorithm, if h(x) = 0, gh_weights = (1,0)
        - A*, if gh_weights = (1,1)

    Args:
        start (SearchNode):
            The node at which to begin the search
        is_goal (callable[SearchNode]):
            A function that returns whether a node has satisfied the goal condition
        step_cost (callable):
            A function that takes an action/macro as input and returns its step cost
        heuristic (callable):
            A function that takes a state as input and returns its heuristic value
        get_successors (callable):
            A function that takes a state as input and returns all possible successor states
        max_transitions (int):
            The simulation budget for the search
        save_best_n (int):
            The number of best SearchNodes to maintain during the search
        quiet (boolean):
            Whether to suppress progress bars
        gh_weights (tuple):
            Tuple of weights (g_weight, h_weight) for weighted A*
    """
    n_expanded = 0
    n_transitions = 0
    open_set = pq.PriorityQueue()
    closed_set = set()
    g_score = defaultdict(lambda: float('inf'))
    g_score[start] = 0
    root = SearchNode(state=start, g_score=0, h_score=heuristic(start), parent=None, action=None)
    # Adding root to open set
    open_set.push((root.f_score, root))
    candidates = [(n_transitions, root)]
    best = root
    # save best N nodes, always ejecting the max priority element to make room
    best_n = pq.PriorityQueue(maxlen=save_best_n, mode='max')

    with tqdm(total=max_transitions, disable=quiet) as progress:
        while open_set and n_transitions < max_transitions:
            _, current = open_set.pop()
            n_expanded += 1
            if is_goal(current):
                candidates.append((n_transitions, current))
                # Found goal! Reconstructing path...
                return reconstruct_path(current) + (n_expanded, n_transitions, candidates)

            if current.state in closed_set:
                continue  # Node already in closed set; ignore it
            closed_set.add(current.state)

            if (current.h_score < best.h_score
                    or (current.h_score == best.h_score
                        and current.g_score < best.g_score)):
                # Found better node!
                best = current
                candidates.append((n_transitions, current))

            best_n.push((current.h_score, reconstruct_path(current)[1]))

            # Considering successors...
            successors = get_successors(current.state)
            n_transitions += len(successors)
            progress.update(len(successors))
            for state, action in successors:
                if state in closed_set:
                    continue

                # Evaluating successor node
                g_score_via_current = g_score[current.state] + step_cost(action)
                if g_score_via_current < g_score[state]:
                    # Found better path to `state`
                    g_score[state] = g_score_via_current
                    # We'd like to remove any existing `state` SearchNodes from the
                    # heap, but removing from a heap is tricky. Instead we just add
                    # a new node, allowing duplicates to exist in the heap, and we
                    # wait for them to be pulled out in due time. Duplicates will be
                    # ignored anyway after the first instance of `state` is added to
                    # `closed_set`.
                    neighbor = SearchNode(state=state, g_score=g_score[state],
                                          h_score=heuristic(state), gh_weights=gh_weights,
                                          parent=current, action=action)
                    if is_goal(neighbor):
                        candidates.append((n_transitions, neighbor))
                        # Found goal! Reconstructing path...
                        return reconstruct_path(neighbor) + (n_expanded, n_transitions, candidates)
                    # Improved path to successor node; adding to open set
                    open_set.push((neighbor.f_score, neighbor))

        # No solution found. Reconstructing path to best node...
        if save_best_n > 1:
            return reconstruct_path(best) + (n_expanded, n_transitions, candidates, best_n.items())
        return reconstruct_path(best) + (n_expanded, n_transitions, candidates)

def weighted_astar(*args, **kwargs):
    """Weighted A* search"""
    # This function wraps the core implementation to ensure that the returned list
    # of 'candidate' SearchNodes have each had their parent information stripped
    # so that the results can be pickled without hitting python's recursion limit.
    results = _weighted_astar(*args, **kwargs)
    candidates = results[4]
    for _, node in candidates:
        node.parent = None
    return results

def astar(*args, **kwargs):
    """A* search"""
    return weighted_astar(*args, gh_weights=(1,1), **kwargs)

def dijkstra(*args, **kwargs):
    """Dijkstra's algorithm"""
    return weighted_astar(*args, heuristic=lambda x: 0, gh_weights=(1,0), **kwargs)

def gbfs(*args, **kwargs):
    """Greedy best-first search (GBFS)"""
    return weighted_astar(*args, gh_weights=(0,1), **kwargs)
