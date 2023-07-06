from env import MushrReachEnv
from gym.envs.registration import register

class MushrEnvironmentFactory:
    def __init__(self):
        self.noise_levels = [0.01, 0.02, 0.04]
        self.noise_level_names = ["Low", "Med", "High"]
    
    def register_environments_with_position_goals(self):
        register(id="MushrObsEnv-v0",entry_point=MushrReachEnv, kwargs={'noisy': False, 'use_obs': True, 'use_orientation': False})
        for i, noise_level in enumerate(self.noise_levels):
            register(id="MushrObs"+self.noise_level_names[i]+"NoisyEnv-v0",entry_point=MushrReachEnv, kwargs={'noisy': True, 'use_obs': True, 'use_orientation': False, 'noise_scale': noise_level})
        
    def register_environments_with_position_and_orientation_goals(self):
        register(id="MushrObsEnv-v0",entry_point=MushrReachEnv, kwargs={'noisy': False, 'use_obs': True, 'use_orientation': True})
        for i, noise_level in enumerate(self.noise_levels):
            register(id="MushrObs"+self.noise_level_names[i]+"NoisyEnv-v0",entry_point=MushrReachEnv, kwargs={'noisy': True, 'use_obs': True, 'use_orientation': True, 'noise_scale': noise_level})