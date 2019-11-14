#!/usr/bin/env python
# coding: utf-8
# author

import pdb
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle

from collections import deque

from unityagents import UnityEnvironment

# Get the DDPG agent
from src.ddpg_agent import Agent


def train(env,
          max_sim_time=1000,
          min_performance=30,
          num_episodes=10,
          window_size=100,
          actor_local_save_filename='actor_local_weights.pt',
          actor_target_save_filename='actor_target_weights.pt',
          critic_local_save_filename='critic_local_weights.pt',
          critic_target_save_filename='critic_target_weights.pt',
          actor_local_load_filename=None,
          critic_local_load_filename=None,
          actor_target_load_filename=None,
          critic_target_load_filename=None,
          random_seed=12345):
    """Train an agent using DDPG Agent.

        Params
        ======
            env                         : Unity environment
            max_sim_time                : maximum number of time steps in an
                                          episode
            num_episodes(int)           : maximum number of episodes to use for
                                          training the agent
            window_size (int)           : the length of the running average
                                          window to compute the average score
            actor_local_save_filename   : the file where the local NN weights
                                          will be saved
            critic_local_save_filename  : the file where the target NN weights
                                          will be saved
            actor_target_save_filename  : the file where the local NN weights
                                          will be saved
            critic_target_save_filename : the file where the target NN weights
                                          will be saved
            actor_local_load_filename   : if given, the initial weights
                                          of the actor local NN
            critic_local_load_filename  : if given, the initial weights
                                          of the critic local NN
            actor_target_load_filename  : if given, the initial weights
                                          of the actor target NN
            critic_target_load_filename : if given, the initial weights
                                          of the critic target  NN
       """
    # Environments contain **_brains_** which are responsible for deciding
    # the actions of their associated agents.
    # Here we check for the first brain available,
    # and set it as the default brain we will be controlling from Python.

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents in the environment
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)
    # number of actions
    action_size = brain.vector_action_space_size
    print('Number of actions:', action_size)
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('States have length:', state_size)

    print('There are {} agents. Each observes a state with length: {}'.
          format(num_agents, state_size))
    # print('The state for the first agent looks like:', states)

    # Create the agent
    agent = Agent(state_size, action_size, random_seed)

    # Load weights if they exist
    agent.load(actor_local_load_filename,
               actor_target_load_filename,
               critic_local_load_filename,
               critic_target_load_filename)

    # Loop over number of episodes
    # Storage for scores
    scores = []
    avg_scores = []
    scores_window = deque(maxlen=window_size)
    for e in range(num_episodes):
        # initialize the score
        score = np.zeros(num_agents)
        # reset the environment
        env_info = env.reset(train_mode=True)[brain_name]
        # get the current state
        states = env_info.vector_observations

        tstart = time.time()
        for sim_t in range(max_sim_time):
            print("""Simulation time: {}""".format(str(sim_t)), end='\r')
            # select an action
            action = agent.act(states, add_noise=True)
            # send the action to the environment
            env_info = env.step(action)[brain_name]
            # get the next state
            next_states = env_info.vector_observations
            # get the reward
            rewards = env_info.rewards
            # see if episode has finished
            dones = env_info.local_done

            # update the replay buffer and train if necessary
            agent.step(states,
                       action,
                       rewards,
                       next_states,
                       dones,
                       sim_t)
            # update the score
            score += rewards
            # roll over the state to next time step
            states = next_states
            # exit loop if episode finished
            if np.any(dones):
                break
        ttotal = time.time() - tstart

        # Append the scores
        scores.append(score)
        scores_window.append(score)
        # Check if the minimum threshold for the reward has been achieved
        avg_score = np.mean(scores_window)
        avg_scores.append(np.mean(avg_score))
        if (e+1) % 1 == 0:
            print("""Episode: {} Episode Duration: {:.0f}s min_score: {:.2f} max_score: {:.2f} average_score: {:.2f}""".format(
                (e+1), ttotal, float(np.min(score)), float(np.max(score)), float(np.mean(avg_score))))
        if np.mean(avg_score) >= min_performance:
            print('\nEnvironment solved in {:d} episodes! \tAverage Score: {:.2f}'.format(
                (e+1), float(avg_score)))
            break

    agent.save(actor_local_save_filename,
               actor_target_save_filename,
               critic_local_save_filename,
               critic_target_save_filename)

    # Save the scores in case of any issues
    with open('training_results.pkl', 'wb') as f:
        pickle.dump([e, avg_scores, scores], f)

    # When finished, you can close the environment.
    return(e, avg_scores, scores)


def run(env,
        num_episodes=1,
        sim_max_time=1000,
        actor_local_load_filename='actor_local_weights.pt'):
    """
        Params
        ======
            env : Unity environment
            num_episodes(int)           : number of episodes to use
                                          to evaluate the agent
            sim_max_time                : maximum time steps in an episode
            actore_local_load_filename  : the file that contains
                                          the weights of the actor NN
        """
    # Environment Setup
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    # Load the
    env_info = env.reset(train_mode=False)[brain_name]
    # number of actions
    num_agents = len(env_info.agents)
    action_size = brain.vector_action_space_size
    # Get the state space information
    states = env_info.vector_observations
    state_size = states.shape[1]
    # Create the agent with the trained weights
    agent = Agent(state_size, action_size, 12345)
    agent.load(actor_local_load_filename)

    # Create the scores storage
    scores = []
    for e in range(num_episodes):
        # initialize the score
        score = np.zeros(num_agents)
        # reset the environment
        env_info = env.reset(train_mode=False)[brain_name]
        # get the current state
        states = env_info.vector_observations
        for t in range(sim_max_time):
            # select an action
            action = agent.act(states, add_noise=False)
            # send the action to the environment
            env_info = env.step(action)[brain_name]
            # get the next state
            next_states = env_info.vector_observations
            # get the reward
            rewards = env_info.rewards
            # see if episode has finished
            dones = env_info.local_done
            score += rewards
            # roll over the state to next time step
            states = next_states
            # exit loop if episode finished
            if np.any(dones):
                break

        # Append the scores
        scores.append(score)
        return(scores)


def main():
    # Load the unity app
    env = UnityEnvironment(file_name="Reacher20.app")
    num_episodes,  avg_scores, scores = train(env, num_episodes=500)
    print("Done training")

    # Plot the training scores
    mean_scores = [np.mean(m) for m in scores]
    plt.ion()
    fig = plt.figure()
    _ = fig.add_subplot(111)
    plt.plot(np.arange(len(mean_scores)), mean_scores)
    plt.plot(np.arange(len(avg_scores)), avg_scores)
    plt.ylabel('Score and Average Scores ')
    plt.xlabel('Episode #')
    plt.legend(['Score', 'Average Score'], loc='upper left')
    plt.show()

    scores = run(env,
                 num_episodes=1,
                 actor_local_load_filename='actor_local_weights.pt')
    print("Done simulating")
    env.close()
