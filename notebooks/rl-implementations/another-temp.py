from trackmania_env import TrackmaniaEnv

num_actions = 3
env = TrackmaniaEnv(num_actions=num_actions)

print(env.action_space.sample())

env.reset()
