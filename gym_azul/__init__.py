from gym.envs.registration import register

register(
    id='azul-v0',
    entry_point='gym_azul.envs:AzulEnv',
)
