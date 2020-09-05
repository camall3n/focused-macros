from collections import deque
from experiments.search import SearchNode, reconstruct_path, get_unique_atoms
from experiments.width import WidthAugmentedHeuristic

def iw(k, start, get_successors, goal_fns):
    goal_nodes = [None for g in goal_fns]
    width_heuristic = WidthAugmentedHeuristic(n_variables=len(start),
                                              heuristic = lambda x: 0,
                                              precision = k+1)
    width_fn = lambda x: width_heuristic(x)[0]

    n_expanded = 0
    n_transitions = 0
    open_queue = deque()
    seen_set = set()
    closed_set = set()
    root = SearchNode(state=start, g_score=0, h_score=0, parent=None, action=None)

    # Adding root to open set
    seen_set.add(start)
    _ = width_fn(start)# mark the start state as seen by novelty function
    open_queue.append(root)

    while open_queue:
        current = open_queue.popleft()
        if current.state in closed_set:
            continue  # Node already in closed set; ignore it
        closed_set.add(current.state)

        # Check for satisfied goal_fns
        n_expanded += 1
        for i in range(len(goal_fns)):
            if goal_nodes[i] is None:
                is_goal_fn = goal_fns[i]
                if is_goal_fn(current):
                    # Found goal! Saving final SearchNode
                    goal_nodes[i] = current
        if all([g is not None for g in goal_nodes]):
            # Found all goals!
            break

        # Considering successors...
        successors = get_successors(current.state)
        n_transitions += len(successors)
        for state, action in successors:
            # If the state fails the novelty check, prune it
            if width_fn(state) > k:
                seen_set.add(state)
                closed_set.add(state)

            if state in closed_set:
                continue

            if state not in seen_set:
                seen_set.add(state)
                neighbor = SearchNode(state=state, g_score=0, h_score=0, gh_weights=(0,0),
                    parent=current, action=action)
                open_queue.append(neighbor)
    else:
        # At least one goal_fn was not satisfied
        return set([])

    # All goal_fns were satisfied!
    goal_paths = [reconstruct_path(g) for g in goal_nodes]
    trajectories = [states for (states, actions) in goal_paths]
    visited_states = [state for trajectory in trajectories for state in trajectory]
    relevant_atoms = get_unique_atoms(visited_states)
    return relevant_atoms

if __name__ == "__main__":
    from domains.npuzzle.npuzzle import NPuzzle