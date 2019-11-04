#!/usr/bin/env python
# coding: utf-8
# author

import pdb
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

from unityagents import UnityEnvironment

# Get the DDPG agent
from src.ddpg_agent import Agent

def train( env,
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
            num_episodes(int)           : maximum number of episodes to use to train the agent
            window_size (int)           : the length of the running average
                                          window to compute the average score
            actor_local_save_filename   : the file where the local NN weights
                                          will be saved
            critic_local_save_filename  : the file where the target NN weights will be saved  
            actor_target_save_filename  : the file where the local NN weights will be saved
            critic_target_save_filename : the file where the target NN weights will be saved  
            actor_local_load_filename   : if given, the initial weights of the local NN
            critic_local_load_filename  : if given, the initial weights if the target NN
            actor_target_load_filename  : if given, the initial weights of the local NN
            critic_target_load_filename : if given, the initial weights if the target NN
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
    print('Number of agents:', len(env_info.agents))
    # number of actions
    action_size = brain.vector_action_space_size
    print('Number of actions:', action_size)
    state = env_info.vector_observations[0]
    state_size = state.shape[0]
    print('States have length:', state_size)

    print('There are {} agents. Each observes a state with length: {}'.
          format(len(env_info.agents), state_size))
    print('The state for the first agent looks like:', state)

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
        score = 0
        # reset the environment
        env_info = env.reset(train_mode=True)[brain_name]
        # get the current state
        state = env_info.vector_observations[0]
        while True:
            # select an action
            action = agent.act(state, add_noise=True)
            # send the action to the environment
            env_info = env.step(action)[brain_name]
            # get the next state
            next_state = env_info.vector_observations[0]
            # get the reward
            reward = env_info.rewards[0]
            # see if episode has finished
            done = env_info.local_done[0]

            # update the replay buffer and train if necessary
            agent.step(state,
                       action,
                       reward,
                       next_state,
                       done)
            # update the score
            score += reward
            # roll over the state to next time step
            state = next_state
            # exit loop if episode finished
            if done:
                break

        # Append the scores
        scores.append(score)
        scores_window.append(score)
        # Check if the minimum threshold for the reward has been achieved
        avg_score = np.mean(scores_window)
        avg_scores.append(avg_score)
        if (e+1) % 10 == 1:
            print("""Episode: {} Score:  {:.2f} average score: {:.2f}  over episodes: {}""".format((e+1), score, avg_score, min((e+1), window_size)))
        if avg_score >= min_performance:
            print('\nEnvironment solved in {:d} episodes! \tAverage Score: {:.2f}'.format((e+1), np.mean(scores_window)))
            break

    agent.save(actor_local_save_filename,
               actor_target_save_filename,
               critic_local_save_filename,
               critic_target_save_filename)
    # When finished, you can close the environment.
    return(e, avg_scores, scores)


def run(env,
        num_episodes=1,
        actor_local_load_filename='actor_local_weights.pt'):
    """
        Params
        ======
            env : Unity environment
            num_episoodes(int): number of episodes to use
                                to evaluate the agent
            actore_local_load_filename: the file that contains
                                        the weights of the actor NN
        """
    # Environment Setup
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    # Load the
    env_info = env.reset(train_mode=False)[brain_name]
    # number of actions
    action_size = brain.vector_action_space_size
    # Get the state space information
    state = env_info.vector_observations[0]
    state_size = len(state)
    # Create the agent with the trained weights
    agent = Agent(state_size, action_size, 12345)
    agent.load(actor_local_load_filename)

    ### Create the scores storage
    scores = []
    for e in range(num_episodes):
        # initialize the score
        score = 0
        # reset the environment
        env_info = env.reset(train_mode=False)[brain_name]
        # get the current state
        state = env_info.vector_observations[0]
        while True:
            # select an action
            action = agent.act(state, add_noise=False)
            # send the action to the environment
            env_info = env.step(action)[brain_name]
            # get the next state
            next_state = env_info.vector_observations[0]
            # get the reward
            reward = env_info.rewards[0]
            # see if episode has finished
            done = env_info.local_done[0]
            
            # Update the replay buffer and train if necessary
            agent.step(state, action, reward, next_state, done)
            # update the score
            score += reward
            # roll over the state to next time step
            state = next_state
            # exit loop if episode finished
            if done:
                break

        ## Append the scores
        scores.append(score)
        return(scores)


def main():
    # Load the unity app
    env = UnityEnvironment(file_name="Reacher.app")
    num_episodes,  avg_scores, scores = train(env, num_episodes=100)
    print("Done training")

    # Plot the training scores
    plt.ion()
    fig = plt.figure()
    _ = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.plot(np.arange(len(avg_scores)), avg_scores)
    plt.ylabel('Score and Average Scores ')
    plt.xlabel('Episode #')
    plt.legend(['Score', 'Average Score'], loc='upper left')
    plt.show()

    scores = run(env, num_episodes=1,
                 actor_local_load_filename='actor_local_weights.pt')
    print("Done simulating")
    env.close()
