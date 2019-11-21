
# Project 2: Continuous Control

### Introduction

For this project, the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment is used to train the agent. In this environment, a double-jointed arm moves to target locations. the agent is trained so that the arm follows a desired trajectory.

#### Reward
A reward of +0.1 is provided for each time step that the agent's hand is in the goal location.

#### States
The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. 

#### Actions
Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector is a number between -1 and 1.

#### Number of Arms in the Environment 

The code in this repo solves two separate versions of the Unity environment. 
- The first version contains a single arm.
- The second version contains 20 identical arms, each with its own copy of the environment. In this version, multiple arms are used to collect feedback in parallel. This version is useful for algorithms like [PPO](https://arxiv.org/pdf/1707.06347.pdf), [A3C](https://arxiv.org/pdf/1602.01783.pdf), and [D4PG](https://openreview.net/pdf?id=SyZipzbCb) that use multiple (non-interacting, parallel) copies of the same agent.  

### Success criteria for Training

 Agents must get an average score of +30 (over 100 consecutive episodes, and over all agents). 
 - Specifically, after each episode, the rewards that each agent received (without discounting) is added up, to get a score for each agent. Then average of these scores is taken. 
- This yields an **average score** for each episode (where the average is over all agents).
- The environment is considered solved, when the average (over 100 episodes) of those average scores is greater than 30. 

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:

    - **_Version 1: One (1) Agent_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

    - **_Version 2: Twenty (20) Agents_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

2. Place the file in the directory where you have cloned this project, and unzip (or decompress) the file.
3. Name the directory where the 20-agent version is unzipped (decompressed) as Reacher20.app and the one agent version as Reacher.app 
4. The repo contains the zip files for the Mac OSX operating system.

### Instructions
Run the `Continuous_Control_Submission.ipynb` notebook to get started.  

### Implementation Details
The details can be found in `Report.ipynb` file





