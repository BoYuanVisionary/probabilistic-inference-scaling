# This is the file for dynamic resampling
# It supports two features: 
# 1. perturb the particles with low rewards using the self-correction of LLMs and then accept the perturbed particles with a min(1, reward(new)/ reward(old))
# 2. dynamically adjust the number of particles based on the probability of the perturbed particles that greater than a given threshold

from sal.config import Config
from typing import List
import numpy as np
from vllm import SamplingParams


CRITIQUE_SINGLE_STEP_PROMPT = """You are an expert mathematician reviewing a specific step in a partially solved math problem.

**Your Task:**

1.  **Context:** Review the 'Original Problem', the 'Previous Steps' (assume these are correct), and the specific 'Step {step_number_to_analyze} Under Review'.
2.  **Analyze:** Determine *why* 'Step {step_number_to_analyze} Under Review' is incorrect based on the preceding steps and the original problem. If it happens to be correct, state that.
3.  **Output:** Your entire output should be ONLY:
    * A concise explanation of the error found in Step {step_number_to_analyze}.
    * The correctly modified text for Step {step_number_to_analyze}. Do NOT include any other steps (previous or subsequent).

**Example Output Structure:**
Reasoning: [Explain the error in Step {step_number_to_analyze} here.]

Corrected Step {step_number_to_analyze}: [Provide the full text of the corrected Step {step_number_to_analyze} here.]

---
**Original Problem:**
{question}
---
**Previous Steps (Steps 1 to {step_number_to_analyze_minus_1}):**
{steps_before_suspicious_text}
---
**Step {step_number_to_analyze} Under Review:**
{suspicious_step_text}
---

**Your Task:** Provide the reasoning and the single corrected step text below.
"""


FIRSTCRITIQUE_SINGLE_STEP_PROMPT = """You are an expert mathematician reviewing a specific step in a partially solved math problem.

**Your Task:**

1.  **Context:** Review the 'Original Problem', and the specific 'Step {step_number_to_analyze} Under Review'.
2.  **Analyze:** Determine *why* 'Step {step_number_to_analyze} Under Review' is incorrect based on the original problem. If it happens to be correct, state that.
3.  **Output:** Your entire output should be ONLY:
    * A concise explanation of the error found in Step {step_number_to_analyze}.
    * The correctly modified text for Step {step_number_to_analyze}. Do NOT include any other steps (previous or subsequent).

**Example Output Structure:**
Reasoning: [Explain the error in Step {step_number_to_analyze} here.]

Corrected Step {step_number_to_analyze}: [Provide the full text of the corrected Step {step_number_to_analyze} here.]

---
**Original Problem:**
{question}
---
**Step {step_number_to_analyze} Under Review:**
{suspicious_step_text}
---

**Your Task:** Provide the reasoning and the single corrected step text below.
"""


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
        self.system_prompt_MCMC = """"
        
        Perform a self-evaluation: You may include reasoning to verify correctness. 
        please identify the mistake in your previous reasoning, revise your reasoning path.

        Here is your previous reasoning:
        {previous_reasoning}
        """

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
            if prob > threshold and len(probs) >= size / 4: # This is to avoid the extreme case
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
    
    def self_refinement(
        llm,
        particles: List[Particle],
        question: str,
    ) -> List[Particle]:
        """
        Batch-refines particles by sending all refinement prompts in one API call.
        Assumes the last step in each particle is the one needing refinement.

        Args:
            llm: The language model instance (must support batch 'generate').
            particles: A list of Particle objects with a 'trajectory' attribute.
            question: The original math problem statement.

        Returns:
            A list of new Particle objects containing the refined solutions.
        """
        refined_particles: List[Particle] = []

        # Sampling parameters for the LLM call (common to all requests)
        sampling_params = SamplingParams(
            temperature=0.8,
            max_tokens=2048,
            stop=["<|eot_id|>"]
        )

        tokenizer = llm.get_tokenizer()

        # Build batch of prompts
        batch_prompts = []
        for particle in particles:
            if not particle.trajectory:
                raise ValueError("Particle trajectory is empty")


            n = len(particle.trajectory)
            if n == 1:
                prompt = FIRSTCRITIQUE_SINGLE_STEP_PROMPT.format(
                    question=question,
                    step_number_to_analyze=1,
                    suspicious_step_text=particle.trajectory[0]
                )
            else:
                idx = n - 1
                before = particle.trajectory[:idx]
                before_text = "\n\n".join(before)
                prompt = CRITIQUE_SINGLE_STEP_PROMPT.format(
                    question=question,
                    step_number_to_analyze_minus_1=idx,
                    step_number_to_analyze=n,
                    steps_before_suspicious_text=before_text,
                    suspicious_step_text=particle.trajectory[idx]
                )
            messages = [{"role": "system", "content": prompt}]
            batch_prompts.append(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )

        # Batch generate
        try:
            batch_outputs = llm.generate(batch_prompts, sampling_params)
        except Exception as e:
            # On failure, return originals
            print(f"Batch generation error: {e}")
            return [p.deepcopy() for p in particles]
        
        # Map each output back to its particle
        refined_particles: List[Particle] = []
        for particle, output in zip(particles, batch_outputs):
            text = output.outputs[0].text.strip()
            if not text:
                refined_particles.append(particle.deepcopy())
            else:
                new_particle = particle.deepcopy(len(particle.trajectory)-1)
                new_particle.trajectory.append(text.split("\n\n")[1])
                refined_particles.append(new_particle)
        return refined_particles


