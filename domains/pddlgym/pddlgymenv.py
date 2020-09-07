import random

def generate_initial_states(env, max_steps=10000):
    """Breadth-first search to find unique states"""

    initial_state, _ = env.reset()

    n_steps = 0
    seen_states = set([initial_state])
    frontier = [initial_state]
    while frontier and n_steps < max_steps:
        state = frontier.pop()
        valid_actions = sorted(list(env.action_space.all_ground_literals(state)))
        for action in valid_actions:
            env.set_state(state)
            next_state = env.step(action)[0]
            n_steps += 1
            if next_state not in seen_states:
                seen_states.add(next_state)
                frontier.append(next_state)
            if n_steps >= max_steps:
                break

    seen_states.remove(initial_state)
    # Sort states using the One True Ordering
    states = sorted(list(seen_states), key=lambda x: sorted(list(x.literals)))
    old_rng_st = random.getstate()
    random.seed(0)
    random.shuffle(states)
    random.setstate(old_rng_st)

    return states

def scramble(env, seed=0, max_steps=3000):
    assert seed >= 0
    if seed == 0:
        initial_state, _ = env.reset()
    else:
        states = generate_initial_states(env, max_steps)
        if seed > len(states):
            raise RuntimeError('Seed {} is undefined (not enough initial states). Try increasing max_steps.')
        initial_state = states[seed-1]
        env.set_state(initial_state)
    return initial_state

def all_scrambles(env, max_steps=3000):
    states = generate_initial_states(env, max_steps)
    return states
