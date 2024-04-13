from gym.envs.registration import register

register(
    id='snake-custom-v0',
    entry_point='snake_gym_custom.snake_gym_custom.envs.SnakeEnvCustom:SnakeEnvCustom',
)