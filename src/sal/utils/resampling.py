# This is the file for dynamic resampling
# It supports two features: 
# 1. perturb the particles with low rewards using the self-correction of LLMs and then accept the perturbed particles with a min(1, reward(new)/ reward(old))
# 2. dynamically adjust the number of particles based on the probability of the perturbed particles that greater than a given threshold

from sal.config import Config
from typing import List
import numpy as np



class Particle:
    def __init__(self, temperature=0.8):
        """
        Initializes a particle with a given temperature.
        
        Args:
            temperature (float): The initial temperature of the particle.
        """
        self.trajectory = []  # Tracks the sequence of responses
        self.rewards = []  # Tracks rewards for each step
        self.tokens_num = [] # Tracks the number of tokens in each step
        self.steps = 0  # Steps taken by the particle
        self.active = True  # Indicates if the particle is still evolving
        self.preferred = False  # Indicates if the particle is preferred
        self.temperature = temperature  # Dynamic temperature of the particle

    def add_step(self, response, reward, stop, tokens_num):
        """Adds a step to the particle's trajectory."""
        self.trajectory.append(response)
        self.rewards.append(reward)
        self.tokens_num.append(tokens_num)
        self.steps += 1
        if stop == "EOS" or """\\boxed""" in response:
            self.active = False
        if self.steps >= 40:
            self.active = False

    def get_last_reward(self):
        """Returns the last recorded reward."""
        return self.rewards[-1]

    def is_active(self):
        """Checks if the particle is active."""
        return self.active

    def get_trajectory(self):
        """Returns the full trajectory as a single string."""
        return "\n\n".join(self.trajectory)

    def set_temperature(self, new_temperature):
        """Sets a new temperature for the particle."""
        self.temperature = new_temperature

    def deepcopy(self, numSteps=None):
        """Returns a deep copy of the particle."""
        new_particle = Particle(temperature=self.temperature)

        if numSteps is not None:
            if numSteps >= len(
                self.trajectory
            ):  # capping it so it doesnt go out of bounds
                numSteps = len(self.trajectory)

        if numSteps is not None:
            new_particle.trajectory = self.trajectory[:numSteps]
            new_particle.rewards = self.rewards[:numSteps]
            new_particle.tokens_num = self.tokens_num[:numSteps]
            new_particle.steps = numSteps
            if numSteps == len(self.trajectory):
                new_particle.active = self.active
            else:
                new_particle.active = True
        else:
            new_particle.trajectory = self.trajectory.copy()
            new_particle.rewards = self.rewards.copy()
            new_particle.tokens_num = self.tokens_num.copy()
            new_particle.steps = self.steps
            new_particle.active = self.active

        new_particle.preferred = self.preferred
        return new_particle



class Resampling:
    def __init__(self, config: Config, resampling_threshold: float):
        self.config = config
        self.resampling_threshold = resampling_threshold

    def multinomial_resampling(self, particles: List[Particle], size: int, weights: List[float]):
        """
        Resamples particles based on their weights using multinomial resampling.

        Args:
            particles: List of Particle objects to resample
            weights: List of weights for each particle

        Returns:
            List of resampled particles
        """
        return np.random.choice(particles, size=size, p=weights, replace=True)

    def adaptive_resampling(self, particles: List[Particle], size: int, weights: List[float], probs: List[float]):
        """ 
        Resamples particles based on their weights using adaptive resampling.

        Args:
            particles: List of Particle objects to resample
            weights: List of weights for each particle
        """
        threshold = self.resampling_threshold
        selected_particles = []
        selected_probs = []
        # A greedy approach to resample the particles
        # 1. select the particles with the highest weights
        # 2. verify if the particle is preferred
        def verify_particle(probs):
            prob = 1-np.prod(1-np.array(probs))
            if prob > threshold and len(probs) >= size / 2: # This is to avoid the extreme case
                return True
            else:
                return False
        
        while len(selected_particles) < size:
            # randonmly select a particle and return the index
            index = np.random.choice(range(len(particles)), p=weights)
            selected_particles.append(particles[index])
            selected_probs.append(probs[index])
            # TODO:once add a particle, the weights should be reduced to reduce the chance of being selected again
            if verify_particle(selected_probs):
                print(f"the number of selected particles is {len(selected_particles)} out of {size}")
                return selected_particles
        print(f"the number of selected particles is {len(selected_particles)} out of {size}")
        return selected_particles
            


        


