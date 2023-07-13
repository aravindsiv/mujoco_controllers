from env import MushrReachEnv
from gym.envs.registration import register

class MushrEnvironmentFactory:
    def __init__(self, return_full_trajectory=False, prop_steps=100):
        self.noise_levels = [0.01, 0.02, 0.04]
        self.noise_level_names = ["Low", "Med", "High"]
        self.return_full_trajectory = return_full_trajectory
        self.prop_steps = prop_steps
    
    def register_environments_with_position_goals(self):
        register(id="MushrObsEnv-v0",entry_point=MushrReachEnv, kwargs={'noisy': False, 'use_obs': True, 'use_orientation': False, 'return_full_trajectory': self.return_full_trajectory, 'prop_steps': self.prop_steps})
        for i, noise_level in enumerate(self.noise_levels):
            register(id="MushrObs"+self.noise_level_names[i]+"NoisyEnv-v0",entry_point=MushrReachEnv, kwargs={'noisy': True, 'use_obs': True, 'use_orientation': False, 'noise_scale': noise_level, 
            'return_full_trajectory': self.return_full_trajectory, 'prop_steps': self.prop_steps})
        
    def register_environments_with_position_and_orientation_goals(self):
        register(id="MushrObsEnv-v0",entry_point=MushrReachEnv, kwargs={'noisy': False, 'use_obs': True, 'use_orientation': True, 'return_full_trajectory': self.return_full_trajectory})
        for i, noise_level in enumerate(self.noise_levels):
            register(id="MushrObs"+self.noise_level_names[i]+"NoisyEnv-v0",entry_point=MushrReachEnv, kwargs={'noisy': True, 'use_obs': True, 'use_orientation': True, 'noise_scale': noise_level, 
            'return_full_trajectory': self.return_full_trajectory, 'prop_steps': self.prop_steps})