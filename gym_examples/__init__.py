from gym.envs.registration import register

register(
    id="gym_examples/GridWorld-v0",
    entry_point="gym_examples.envs:GridWorldEnv",
)
register(
    id="gym_examples/TSP-v0",
    entry_point="gym_examples.envs:TSPEnvironment",
)
register(
    id="gym_examples/Nqueens-v0",
    entry_point="gym_examples.envs:NQueensEnv",
)

register(
    id="gym_examples/Nqueens",
    entry_point="gym_examples.envs:NQ",
)
