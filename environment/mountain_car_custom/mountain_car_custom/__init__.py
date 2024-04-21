from gym.envs.registration import register
#print("registering mcc")
register(
    id='MountainCarCustom-v0',
    entry_point='mountain_car_custom.mountain_car_custom.envs.mountain_car:MountainCarEnvCustom',
)