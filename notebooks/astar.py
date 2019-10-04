import notebooks.priorityqueue as pq
from collections import defaultdict, deque
from tqdm import tqdm

class Node:
    def __init__(self, state, g_score, h_score, parent=None, action=None):
        self.state = state
        self.action = action
        self.g_score = g_score
        self.h_score = h_score
        self.parent = parent
    def __cmp__(self, other):
        return 0
    def __eq__(self, other):
        return True
    @property
    def f_score(self):
        return self.g_score + self.h_score

def reconstruct_path(node):
    states = [node.state]
    actions = []
    while node.parent:
        states.insert(0, node.parent.state)
        actions.insert(0, node.action)
        node = node.parent
    return states, actions

def search(start, is_goal, step_cost, heuristic, get_successors, max_transitions=0, save_best_n=1, debug_fn=None):
    n_expanded = 0
    n_transitions = 0
    open_set = pq.PriorityQueue()
    closed_set = set()
    g_score = defaultdict(lambda: float('inf'))
    g_score[start] = 0
    root = Node(state=start, g_score=0, h_score=heuristic(start), parent=None, action=None)
    open_set.push((root.f_score, root))
    if debug_fn:
        print('adding root to open set')
        debug_fn(root.state)
    candidates = [(n_transitions, root)]
    best = root
    best_n = deque(maxlen=save_best_n)

    with tqdm(total=max_transitions) as progress:
        while open_set and n_transitions < max_transitions:
            priority, current = open_set.pop()
            n_expanded += 1
            if debug_fn:
                print('pulled node of open set')
                debug_fn(current.state)
            if is_goal(current):
                candidates.append((n_transitions, current))
                if debug_fn:
                    print('found goal. reconstructing path...')
                return reconstruct_path(current) + (n_expanded, n_transitions, candidates)

            if current.state in closed_set:
                if debug_fn:
                    print('node already in closed set')
                continue
            if debug_fn:
                print('adding node to closed set')
                debug_fn(current.state)
            closed_set.add(current.state)

            if (current.h_score < best.h_score
                or (current.h_score == best.h_score
                    and current.g_score < best.g_score)):
                if debug_fn:
                    print('found better node!')
                    debug_fn(current.state)
                    print('previous best was')
                    debug_fn(best.state)
                best = current
                candidates.append((n_transitions, current))

            if current.h_score + current.g_score <= best.h_score+best.g_score:
                best_n.append((current, reconstruct_path(current)[1]))

            if debug_fn:
                print('considering successors...')
            successors = get_successors(current.state)
            n_transitions += len(successors)
            progress.update(len(successors))
            for state, action in successors:
                if state in closed_set:
                    continue

                if debug_fn:
                    print('evaluating successor node')
                    debug_fn(state)
                # Found better path to `state`
                g_score_via_current = g_score[current.state] + step_cost(action)
                if g_score_via_current < g_score[state]:
                    g_score[state] = g_score_via_current
                    # We'd like to remove any existing `state` Nodes from the heap,
                    # but removing from a heap is tricky. Instead we just add a new
                    # node, allowing duplicates to exist in the heap, and we wait
                    # for them to be pulled out in due time. Duplicates will be
                    # ignored anyway after the first instance of `state` is added
                    # to `closed_set`.
                    neighbor = Node(state, g_score[state], heuristic(state), parent=current, action=action)
                    open_set.push((neighbor.f_score, neighbor))
                    if debug_fn:
                        print('improved path to successor node; adding to open set')
                        debug_fn(state)

        if debug_fn:
            print('no solution found; reconstructing path to best node...')
            debug_fn(best.state)
        if save_best_n > 1:
            return reconstruct_path(best) + (n_expanded, n_transitions, candidates, list(best_n))
        else:
            return reconstruct_path(best) + (n_expanded, n_transitions, candidates)
