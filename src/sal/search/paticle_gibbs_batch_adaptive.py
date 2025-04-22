# TODO: 
# - At the step, perturb the particles with low rewards using the self-correction of LLMs
# - Then accept the perturbed particles with a min(1, reward(new)/ reward(old))
# - Design a adaptive particle filtering where the number of particles is selected based on the probability of the perturbed particles
#   that greater than a given threshold.


from vllm import LLM, SamplingParams
import torch
from sal.config import Config
from sal.models.reward_models import load_prm
import os
from sal.utils.data import get_dataset, save_dataset
from datasets import load_dataset
import numpy as np


from glob import glob

from datasets import load_dataset
from sal.utils.math import *
from sal.utils.grader import *
from sal.utils.resampling import *

from sal.utils.qwen_math_parser import *
from collections import defaultdict
import json
import numpy as np
import random

import pickle
import os
import logging



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def softmax(x):
    """
    Compute softmax values for a vector x.

    Args:
        x (numpy.ndarray): Input array of shape (n,)

    Returns:
        numpy.ndarray: Softmax probabilities of shape (n,)
    """
    # Subtract max for numerical stability
    # This prevents overflow when computing exp
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()


def inverse_sigmoid(x):
    """
    Calculate the inverse sigmoid (logit) of a value x.

    Args:
        x (float): Input value between 0 and 1 (exclusive)

    Returns:
        float: The inverse sigmoid value

    Raises:
        ValueError: If x is not between 0 and 1
    """
    # Add small epsilon to prevent log(0)
    eps = np.finfo(float).eps
    x = np.clip(x, eps, 1 - eps)

    return np.log(x) - np.log(1 - x)  # More stable than np.log(x/(1-x))


def take_a_step_for_batch(question, llm, config, particles_steps_so_far=[[]], first=False, temperature=0.8, n_particles=1):
    # you throw a list of questions into the llm.generate function call 
    tokenizer = llm.get_tokenizer()
    system = [
        {
            "role": "system",
            "content": config.system_prompt,
        }
    ]
    sampling_params = SamplingParams(
        temperature=temperature,  # Dynamic temperature
        max_tokens=2048,
        top_p=1.0,
        stop=["\n\n", "<|eot_id|>"],
    )

    if particles_steps_so_far==[[]]:
        if first:
            prompt = tokenizer.apply_chat_template(
                system + [{"role": "user", "content": question}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = tokenizer.apply_chat_template(
                system + [{"role": "user", "content": question}], tokenize=False,
                add_generation_prompt=True,
            )
            prompt = prompt + "\n\n".join(particles_steps_so_far[0]) + "\n\n"
        
        particles_prompts = [prompt]*n_particles

    else:
        # we need to integrate the previous particle steps so far into the prompt
        particles_prompts = []
        for steps_so_far in particles_steps_so_far:
            if first:   
                prompt = tokenizer.apply_chat_template(
                    system + [{"role": "user", "content": question}], tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                prompt = tokenizer.apply_chat_template(
                    system + [{"role": "user", "content": question}], tokenize=False,
                    add_generation_prompt=True,
                )
                prompt = prompt + "\n\n".join(steps_so_far) + "\n\n"
            particles_prompts.append(prompt)

    particles_res_lists = llm.generate(particles_prompts, sampling_params)
    particles_responses = [res.outputs[0].text for res in particles_res_lists]
    particles_response_tokens = [res.outputs[0].token_ids for res in particles_res_lists]
    def stop_reason(response_tokens):
        if tokenizer.eos_token_id in response_tokens:
            return "EOS"
        else:
            return "END OF STEP"
    particles_stops = [stop_reason(response_tokens) for response_tokens in particles_response_tokens]
    particles_response_tokens_num = [len(response_tokens) for response_tokens in particles_response_tokens]
    return particles_responses, particles_stops, particles_response_tokens_num


def temperature_linear_annealing(starting_temp, ending_temp, total_steps, current_step):
    """
    Computes the temperature at a given step using linear annealing.

    Args:
        starting_temp (float): Initial temperature.
        ending_temp (float): Final temperature.
        total_steps (int): Total number of annealing steps.
        current_step (int): Current step number (0-indexed).

    Returns:
        float: Temperature at the current step.
    """
    if current_step < 0:
        raise ValueError("current_step must be >= 0.")

    if current_step >= total_steps:
        # Return constant ending temperature after the total steps.
        return ending_temp

    if total_steps <= 1:
        # Return ending temperature directly if there's only 1 or no step.
        return ending_temp

    temp_range = starting_temp - ending_temp
    step_fraction = current_step / (total_steps - 1)  # Adjust for 0-indexing
    return starting_temp - (temp_range * step_fraction)


def particle_gibbs_kernel_adaptive(
    question,
    llm,
    prm,
    config,
    n_particles,
    softmax_temp,
    resample_inactive,
    resampling_threshold,
    reference_particle=None,
    temperature_annealing=(),
    llm_sampling_temp=0.8,
):
    """
    Implements particle Gibbs sampling for response generation.

    Args:
        question: The input question/prompt
        llm: Language model instance
        prm: Parameter object containing reward model
        config: Configuration for LLM
        n_particles: The maximal number of particles to maintain
        resample_inactive: Whether to resample inactive particles
        adaptive_resampling_param: Parameters for adaptive resampling. Vanile multinomial resampling if None
    Returns:
        List of trajectories and their scores
    """
    logger.info("Starting Particle Gibbs sampling...")
    logger.info(f"Particles: {n_particles}")
    logger.info(f"Resample inactive: {resample_inactive}")
    logger.info(f"LLM sampling temperature: {llm_sampling_temp}")
    logger.info(f"Softmax temperature: {softmax_temp}")
    logger.info(f"Temperature annealing: {temperature_annealing}")
    logger.info(f"Resampling threshold: {resampling_threshold}")

    stepwise_particle_tracker_before = []
    stepwise_particle_tracker_after = []

    resampling = Resampling(config, resampling_threshold)

    # Initialize particles
    if reference_particle is None:
        particles = [Particle(temperature=llm_sampling_temp) for _ in range(n_particles)]
    else:
        particles = [Particle(temperature=llm_sampling_temp) for _ in range(n_particles - 1)]

    # print(f"Initialized {n_particles} particles.")

    # Initial step for all particles
    # for idx, particle in enumerate(particles):
    #     response, stop = take_a_step(question, llm, config, first=True, temperature=llm_sampling_temp)
    #     reward = prm.score([question], [[response]])[-1][-1][-1]
    #     particle.add_step(response, reward, stop, tokens_num)

    responses, stops, tokens_num = take_a_step_for_batch(question, llm, config, first=True, temperature=llm_sampling_temp, n_particles=len(particles))
    rewards = [prm.score([question], [[response]])[-1][-1][-1] for response in responses]

    for idx, particle in enumerate(particles):
        particle.add_step(responses[idx], rewards[idx], stops[idx], tokens_num[idx])


    stepwise_particle_tracker_before.append([p.deepcopy() for p in particles])

    step = 1

    while any(particle.is_active() for particle in particles):
        if resample_inactive:
            rewards = [particle.get_last_reward() for particle in particles]

            if reference_particle is not None:
                if step >= len(reference_particle.rewards):
                    rewards.append(reference_particle.rewards[-1])
                else:
                    rewards.append(reference_particle.rewards[step])

            logits = [inverse_sigmoid(r) for r in rewards]
            logits = np.array(logits)
            
            if temperature_annealing:
                softmax_temp = temperature_linear_annealing(
                    starting_temp=temperature_annealing[0],
                    ending_temp=temperature_annealing[1],
                    total_steps=temperature_annealing[2],
                    current_step=step,
                )

            logger.info(f"Softmax temperature: {softmax_temp}")
            weights = softmax(logits / softmax_temp)


            # Sample new particles based on weights
            if reference_particle is None:
                sampled_particles = resampling.multinomial_resampling(
                    particles,
                    size=n_particles, # Size is the upper bound of the number of possible particles
                    weights=weights,
                    probs=rewards,
                )
            else:
                sampled_particles = resampling.multinomial_resampling(
                    particles + [reference_particle],
                    size=n_particles, # Size is the upper bound of the number of possible particles
                    weights=weights,
                    probs=rewards,
                )

            # TODO: change this to allow inactive particles to be sampled (bc perhaps the score of an incomplete still-evolving particle is higher than that of a completed particle)
            # print(
            #    f"Sampled particle indices: {[particles.index(p) for p in sampled_particles+[reference_particle]]}"
            # )
            particles = [
                particle.deepcopy(numSteps=step) for particle in sampled_particles
            ]

        else:
            # Check active particles
            active_particles = [
                particle for particle in particles if particle.is_active()
            ]

            # print(f"Active particles: {len(active_particles)} / {n_particles}")

            # Calculate rewards and weights
            rewards = [particle.get_last_reward() for particle in active_particles]

            if reference_particle is not None:
                if step >= len(reference_particle.rewards):
                    rewards.append(reference_particle.rewards[-1])
                else:
                    rewards.append(reference_particle.rewards[step])

            logits = [inverse_sigmoid(r) for r in rewards]
            logits = np.array(logits)

            if temperature_annealing:
                softmax_temp = temperature_linear_annealing(
                    starting_temp=temperature_annealing[0],
                    ending_temp=temperature_annealing[1],
                    total_steps=temperature_annealing[2],
                    current_step=step,
                )

            logger.info(f"Softmax temperature: {softmax_temp}")
            weights = softmax(logits / softmax_temp)

            # print(f"Rewards of active particles: {rewards}")
            # print(f"Logits (inverse sigmoid): {logits}")
            # print(f"Weights (softmax): {weights}")

            # Sample new particles based on weights
            if reference_particle is None:
                print("len(particles)", len(particles))
                print("len(weights)", len(weights))
                sampled_particles = resampling.multinomial_resampling(
                    active_particles,
                    size = n_particles-(len(particles)-len(active_particles)),
                    weights=weights,
                    probs=rewards,
                )
            else:
                sampled_particles = resampling.multinomial_resampling(
                    active_particles + [reference_particle],
                    size= n_particles-(len(particles)-len(active_particles)),
                    weights=weights,
                    probs=rewards,
                )

            # print(
            #    f"Sampled particle indices: {[particles.index(p) for p in sampled_particles]}"
            # )
            particles = [
                particle.deepcopy()
                for particle in particles
                if not particle.is_active()
            ]

            for i in range(len(sampled_particles)):
                particles.append(sampled_particles[i].deepcopy(numSteps=step))

        stepwise_particle_tracker_after.append([p.deepcopy() for p in particles])

        # use the take_a_step_for_batch function to take a step for each particle
        responses, stops, tokens_num = take_a_step_for_batch(question, llm, config, first=False, particles_steps_so_far=[particle.trajectory for particle in particles], n_particles=len(particles))
        responses_to_pass_for_score = ["\n\n".join(particle.trajectory) + "\n\n" + response for response, particle in zip(responses, particles)]
        rewards = [prm.score([question], [[response]])[-1][-1][-1] for response in responses_to_pass_for_score]
        for idx, particle in enumerate(particles):
            if not particle.is_active():
                continue
            particle.add_step(responses[idx], rewards[idx], stops[idx], tokens_num[idx])

        # Self-correction to get better particles
        problematic_particles = []
        right_particles = []
        for particle in particles:
            if particle.get_last_reward() < 0.8:
                problematic_particles.append(particle)
            else:
                right_particles.append(particle)
        corrected_particles = resampling.self_refinement(llm, problematic_particles, question)
        particles = right_particles + corrected_particles

        step = step + 1
        stepwise_particle_tracker_before.append([p.deepcopy() for p in particles])

    if reference_particle is None:
        return particles, stepwise_particle_tracker_before, stepwise_particle_tracker_after 
    else:
        return particles + [reference_particle], stepwise_particle_tracker_before, stepwise_particle_tracker_after

import copy
def particle_gibbs_batch_adaptive(
    x,
    config,
    llm,
    prm,
    total_timesteps=1,
    n_particles=4,
    resample_inactive=True,
    resampling_threshold=0.95,
    softmax_temp=1.0,
    temperature_annealing=(),
    llm_sampling_temp=0.8,
):
    
    particle_intermediate_storage = []
    if isinstance(x["unique_id"], int):
        question_id = x["unique_id"]
    else:
        question_id = x["unique_id"].replace("/", "_").strip(".json")
    logger.info(f"Processing question: {question_id}")
    particles_tracker = []
    current_timestep = 1
    current_particles, tracker_before, tracker_after = particle_gibbs_kernel_adaptive(
                            x["problem"], 
                            llm, 
                            prm,   
                            config, 
                            n_particles, 
                            resample_inactive=resample_inactive,
                            resampling_threshold=resampling_threshold,
                            softmax_temp=softmax_temp,
                            temperature_annealing=temperature_annealing,
                            llm_sampling_temp=llm_sampling_temp,
    )
    particles_tracker.append(current_particles)
    particle_intermediate_storage.append([copy.deepcopy(current_particles)])

    # compute_budget_used = sum([len(p.trajectory) for p in current_particles])
    if total_timesteps > 1:
        while current_timestep < total_timesteps:
            # preferred_particle = current_particles[
            #     np.argmax([p.rewards[-1] for p in current_particles])
            # ]
            rewards = [particle.get_last_reward() for particle in current_particles]

            logits = [inverse_sigmoid(r) for r in rewards]
            logits = np.array(logits)

            if temperature_annealing:
                softmax_temp = temperature_linear_annealing(
                    starting_temp=temperature_annealing[0],
                    ending_temp=temperature_annealing[1],
                    total_steps=temperature_annealing[2],
                    current_step=current_timestep,
                )

            logger.info(f"Softmax temperature: {softmax_temp}")
            weights = softmax(logits / softmax_temp)

            preferred_particle = np.random.choice(
                current_particles,
                size=1,
                p=weights,
                replace=True,                             # before particles was active_particles
            )[0]

            preferred_particle.preferred = True

            current_particles, tracker_before, tracker_after = particle_gibbs_kernel_adaptive(
                x["problem"],
                llm,
                prm,
                config,
                n_particles,
                softmax_temp,
                resample_inactive=resample_inactive,
                resampling_threshold=resampling_threshold,
                reference_particle=preferred_particle,
                temperature_annealing=temperature_annealing
            )
            particles_tracker.append(current_particles)
            current_timestep += 1
            particle_intermediate_storage.append([copy.deepcopy(current_particles)])

    # # Create output directory if it doesn't exist
    os.makedirs(config.output_dir, exist_ok=True)

    save_path = os.path.join(config.output_dir, f"{question_id}.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(particles_tracker, f)

    intermediate_dir = os.path.join(config.output_dir, "intermediate")
    os.makedirs(intermediate_dir, exist_ok=True)
    save_path_intermediate = os.path.join(intermediate_dir, f"{question_id}_intermediate.pkl")
    with open(save_path_intermediate, "wb") as f:
        pickle.dump(particle_intermediate_storage, f)
        
    #with open(save_path.replace(".pkl", "_before.pkl"), "wb") as f:
    #    pickle.dump(tracker_before, f)
    
    #with open(save_path.replace(".pkl", "_after.pkl"), "wb") as f:
    #    pickle.dump(tracker_after, f)

    logger.info(f"Saved particles to: {save_path}")

    return x
