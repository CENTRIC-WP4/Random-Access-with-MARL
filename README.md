# Random-Access-with-MARL
The repository contains RL-gym environment for random access in wireless networks. 

## Overview
The repository may be used to design random access techniques that are traffic-aware and service-aware. 
Different reward functions are provided to test the performance using different traffic arrival models at the UEs. 
Unlike regular traffic arrival in slotted ALOHA theroatical analysis, one may use custom traffic arrival models to learn RA techniques. We will* also provide traffic models that can be used.  

## Getting Started
The code has been tested to work on Python 3.7 

1. Get the code:
    ```
    git clone https://github.com/CENTRIC-WP4/Random-Access-with-MARL.git
    ```
3. The following command may be used to run a random policy provided. 
    ```
    cd Random-Access-with-MARL
    ```
    ```
   python main.py
    ```
## Usage

`env.py` contains the rl-gym environment for random access
`main.py` should be used to execute the code and it contains random policy
`utils.py` contains methods to support the usage of the environment for batch states and rewards etc to be used with MARL
### About the environment

The environment has,

- N users and and K orthogonal resources (channels)
- Each user may or may not have a packet in its buffer
- Every user receives feedback from environment at each time slot (discrete time)
- Binary feedback --> {collision, no-collision} or {success, no-success} --> can be switched
- Ternary feedback --> {collision, idle, success}
- Broadcast feedback - M bits after each time slot for each channel

The following are calculated per agent...
    
**Observation**:
the observation space for each agents includes: 
    - IDs of agents if not False - otherwise IDs are not used
    - Previous action - one-hot encoded for num_channels > 1
    - Binary Indicator whether user belongs to event-traffic {1} or regular {0}
    - feedback (for m channels) {1,0} for each channel - broadbast
    - G_n(k) whether user n's transmission was success
**Actions space**:
- Two actions per agent {transmit or not transmit} over the mth channel --> {0, 1, ..., M}
0 means silent
- The action space is number of reseources + 1, sometimes I may write action_space[0].n, which is basically the same thing. 0 index is just 1st agent but since all are the same, so it doesn't really matter in this case
        
**Reward**:
Several rewards may be used depending on the requirements of the task or objective; 
The objective might be to increase the throughput, or/and to have fairness among user or decreasing packet delay
- *Throughput*:
    - success or no-success --> {0,1} | Can be defined differently (binary, ternary with success, idle and collision as feedback)
- *Reducing collisions*
    - This function penalizes (negative value) agents when they collide, 0 when they are idle and positive when they are successful
Agents are *cooperative* and therefore they all share the same reward; agents can also be *competitive*, having different rewards per agent. 
Agents are homogeneous, i.e., they all have the same state, action spaces and rewards

**Note:** `actions_n` is provided as  as an argument to the step function `step(actions_n)` that follows the RL-gym syntex. However, apart from containing actions of each agent and also other information such as buffer_state. This is a bit different from other RL-gym so be careful when using.

## Baselines
The designed schemes may be compared with backoff schemes such as exponential backoff in terms of throughput and fairness.

## How to Contribute
This repository may be used to extend the environment of random access using MARL. For instance,
1. to designing robust and scalable techniques (with or without MARL) - it would be interesting to see if we can model this problem to be solved with Transformers and design a robust policy
2. to incorporate custom traffic models for performance evaluation and learning policies for random access
3. compare different MARL algorithms for centralized and decentralized training and decentralized execution
4. to propose new reward functions as per the system requirements or to propose new ways to calculate the reward for fairness
5. to learn back-off factor in back-off schemes for random access

## References
1. [Open AI Gym Documentation](http://gym.openai.com/docs/)
2. [How to create new environments for Gym](https://github.com/openai/gym/blob/master/docs/creating-environments.md)
3. [Learning Random Access Schemes for Massive Machine-Type Communication with MARL
](https://arxiv.org/abs/2302.07837)
