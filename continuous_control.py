#!/usr/bin/env python
# coding: utf-8
# author

import pdb
import matplotlib.pyplot as plt
from collections import deque

from unityagents import UnityEnvironment
import numpy as np

### Get the DQN agent
from src.ddpg_agent import Agent

def train( env 
           ,min_performance = 30
          , num_episodes    = 10
          , window_size     = 100
          , local_save_filename   = 'model.pt'
          , target_save_filename  = 'model_target.pt'
          , local_load_filename   =  None
          , target_load_filename  =  None ):

    """Train an agent using DDPG Agent.
        
        Params
        ======
            env : Unity environment
            num_episoodes(int):  maximum number of episodes to use to train the agent
            window_size (int) :  the length of the running average window used to compute the average score
            local_save_filename: the file where the local NN weights will be saved
            target_save_filename: the file where the target NN weights will be saved  
            local_load_filename: if given, the initial weights of the local NN
            target_load_filename: if given, the initial weights if the target NN
        """

    
    # Environments contain **_brains_** which are responsible for deciding the actions of their associated agents.
    # Here we check for the first brain available, and set it as the default brain we will be controlling from Python.
    
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
    state_size = len(state)
    print('States have length:', state_size)
    
    #### Create the agent 
    agent = Agent(state_size, action_size, 12345)    
    ### Loop over number of episodes
    eps = 1.0
    ### Storage for scores 
    scores     = []
    avg_scores = []
    scores_window = deque(maxlen = window_size)
    for e in range(num_episodes):
        # initialize the score
        score    = 0     
        env_info = env.reset(train_mode=True)[brain_name] # reset the environment
        state    = env_info.vector_observations[0]            # get the current state
        while True:
            eps        = max(eps*0.8, 0.0001)     
            action     = agent.act(state, eps)                 # select an action 
            env_info   = env.step(action)[brain_name]          # send the action to the environment
            next_state = env_info.vector_observations[0]       # get the next state
            reward     = env_info.rewards[0]                   # get the reward
            done       = env_info.local_done[0]                # see if episode has finished

            ### Update the replay buffer and train if necessary
            agent.step(state, action, reward, next_state, done)
            score += reward                                # update the score
            state = next_state                             # roll over the state to next time step
            if done:                                       # exit loop if episode finished
                break

        ## Append the scores
        scores.append(score)
        scores_window.append(score)
        ## Check if the minimum threshold for the reward has been achieved
        avg_score = np.mean(scores_window) 
        avg_scores.append(avg_score)
        if (e+1) % 50 == 1: 
            print("""Episode: {} Score:  {:.2f} average score: {:.2f}  over episodes: {}""".format((e+1), score, avg_score, min((e+1), window_size))) 
        if avg_score >= min_performance:
            print('\nEnvironment solved in {:d} episodes! \tAverage Score: {:.2f}'.format((e+1), np.mean(scores_window)))
            break
    agent.save(local_save_filename, target_save_filename)
    # When finished, you can close the environment.
    return(e, avg_scores, scores)

def run(env, num_episodes = 1, local_filename = 'model.pt'):
    """
        Params
        ======
            env : Unity environment
            num_episoodes(int):  number of episodes to use to evaluate the agent
            local_filename:      the file that contains the weights of the trained agent
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
    #### Create the agent with the trained weights
    agent = Agent(state_size, action_size, 12345, local_filename = local_filename)

    ### Create the scores storage
    scores = []
    eps = 0.000
    for e in range(num_episodes):
        # initialize the score
        score    = 0     
        env_info = env.reset(train_mode=False)[brain_name] # reset the environment
        state = env_info.vector_observations[0]            # get the current state
        while True:
            action     = agent.act(state, eps)                 # select an action 
            env_info   = env.step(action)[brain_name]          # send the action to the environment
            next_state = env_info.vector_observations[0]       # get the next state
            reward     = env_info.rewards[0]                   # get the reward
            done       = env_info.local_done[0]                # see if episode has finished

            ### Update the replay buffer and train if necessary
            agent.step(state, action, reward, next_state, done)
            score += reward                                # update the score
            state = next_state                             # roll over the state to next time step
            if done:                                       # exit loop if episode finished
                break

        ## Append the scores
        scores.append(score)
        return(scores)
    
def main():
    ## Load the unity app
    env    = UnityEnvironment(file_name="Banana.app")
    num_episodes,  avg_scores, scores = train(env, num_episodes = 1000)
    
    ### Plot the training scores
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.plot(np.arange(len(avg_scores)), avg_scores)
    plt.ylabel('Score and Average Scores ')
    plt.xlabel('Episode #')
    plt.legend(['Score', 'Average Score'], loc = 'upper left')
    plt.show()
    ###
    
    scores = run(env, num_episodes = 1, local_filename = 'model.pt')
    env.close()
    
    
