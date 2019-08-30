import notebooks.priorityqueue as pq
from collections import defaultdict

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
        actions.insert(0, node.parent.action)
        node = node.parent
    return states, actions

def search(start, is_goal, step_cost, heuristic, get_successors):
    open_set = pq.PriorityQueue()
    closed_set = set()
    g_score = defaultdict(lambda: float('inf'))
    g_score[start] = 0
    root = Node(state=start, g_score=0, h_score=heuristic(start), parent=None, action=None)
    open_set.push((root.f_score, root))
    best = root

    while open_set:
        current = open_set.pop()
        if is_goal(current):
            return reconstruct_path(node)

        if current.state in closed_set:
            continue
        closed_set.add(current.state)

        if (current.h_score < best.h_score
            or (current.h_score == best.h_score
                and current.g_score < best.g_score)):
            best = current

        successors = get_successors(current.state)
        for state, action in successors:
            if state in closed_set:
                continue

            # Found better path to `state`
            g_score_via_current = g_score[current.state] + step_cost
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

    return reconstruct_path(best)
