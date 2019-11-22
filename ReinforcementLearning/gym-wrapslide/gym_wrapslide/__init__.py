#Gym Wrapslide Environment _init_.py
#JE KNOLL

from gym.envs.registration import register

register(
    id='wrapslide-v0',
    entry_point='gym_wrapslide.envs:WrapslideEnv',
)
