
def scramble(env, seed=0, steps=100):
    state, _ = env.reset()
    old_seed = env.action_space.seed()[0]
    env.action_space.seed(seed)
    for step in range(steps):
        action = env.action_space.sample(state)
        state, reward, done, _ = env.step(action)
        if done:
            print('Found goal by scrambling. Continuing to scramble...')
            pass # Just keep on scrambling
    env.action_space.seed(old_seed)
    return state
