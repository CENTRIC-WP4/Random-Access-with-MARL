#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Muhammad Awais Jadoon
Description: RL-gym environment for random access with multiple users
Date: 17-10-2023
"""
import gym
from gym import spaces
import numpy as np


class RA_Env(gym.Env):
    metadata = {'render.modes': ['human']}
    """
    ALOHA-based random access environment - openai gym compliant
    
        N users and and K orthogonal resources (channels)
        Each user may or may not have a packet in its buffer
        Every user receives feedback from environment at each time slot (discrete time)
        Binary feedback --> {collision, no-collision} or {success, no-success} --> can be switched
        Ternary feedback --> {collision, idle, success}
        Broadcast feedback - M bits after each time slot for each channel
    
    The following are calculated per user...
    
    Observation (state space):
        the observation space for each agents includes: 
         - IDs of agents if not False - otherwise IDs are not used
         - Previous action - one-hot encoded for num_channels > 1
         - Binary Indicator whether user belongs to event-traffic {1} or regular {0}
         - feedback (for m channels) {1,0} for each channel - broadbast
         - G_n(k) whether user n's transmission was success
    Actions space:
        not transmit or transmit over the mth channel --> {0, 1, ..., M}
        0 means silent
        The action space is number of reseources + 1, sometimes I may write action_space[0].n, 
        which is basically the same thing. 0 index is just 1st agent but since all are the same, 
        so it doesn't really matter
        
    Reward:
        success or no-success --> {0,1} | Can be defined differently
        agents are cooperative and therefore they all share the same reward
    
    Agents are homogeneous, i.e., they all have the same state, action spaces and rewards
    
    """
    
    
    def __init__(self, 
                 num_users, 
                 num_channels, 
                 num_events,
                 arr_rate,
                 event_prob,
                 fb_type='success_broadcast',
                 use_ids = False):
        
        super(RA_Env, self).__init__()
        
        
        # total number of users/agents in an environment.
        self.n_agents = num_users
        self.num_channels = num_channels
        assert self.num_channels > 0, \
        f"number of channels greater than 0 expected, got: {self.num_channels}"
        
        self.num_events = num_events
        # the average arrival rate 
        self.lmb = arr_rate
        self.event_prob = event_prob
        self.use_ids = use_ids
        self.activ_prob_usr = 1
        
        ''' Define action and observation space '''
        
        self.action_space = [spaces.Discrete(self.num_channels+1) 
                                             for _ in range(self.n_agents)]
        
        
        # number of past actions and feedback, and current buffer state (1)
        obs_size = (1 if self.num_channels==1 else self.num_channels+1) +  \
                                        self.num_channels + \
                                        (1 if self.num_channels>1 else 0) +1 + \
                                        (self.n_agents if self.use_ids == True else 0) #+1
        
        self._obs_low = np.zeros(obs_size, np.float64)
        self._obs_high = np.ones(obs_size, np.float64)

        self.observation_space = [spaces.Box(np.float32(self._obs_low), 
                                             np.float32(self._obs_high)) 
                                          for _ in range(self.n_agents)]
        
        self.fb_type=fb_type
        
        self.agents_type = 'cooperative'
        
    def step(self, actions_n):
        '''
        takes the list of actions from all the agents and returns the list of obs (state),
        rewards, done, and info for all the agents
        
        actions_n argument contain actions of each agent and also other information
        such as buffer_state. This is a bit different from other RL-gym so be careful when using.
        '''
        
        actions = actions_n[0]
        
        # this contains users buffer status and also events status
        self.users_buffer = actions_n[1]

        self.ch_feedback = self.get_feedback(actions)
        successful_users_id = self.ch_feedback[2]
        users_reporting_events = actions_n[2]
        success_per_usr = actions_n[3]
        rewards_n = self.get_reward(self.ch_feedback, self.users_buffer[0], actions, success_per_usr)
        
        info_n = self.get_info_n()
        done_n = self.get_done_n()
        
        # update buffer
        self.pkts_left = self.update_buffer(self.users_buffer[0], successful_users_id)
        self.buffer_left = [self.pkts_left, self.users_buffer[1]]
        
        self.new_buffer = self.gen_traffic(self.buffer_left, users_reporting_events)
        
        state = self.gen_state(self.new_buffer, actions, self.ch_feedback)
        
        return state, rewards_n, done_n, info_n
        
    def get_reward(self, ch_feedback, buffer, actions, successes_n):
        
        rewards_n = np.zeros([self.n_agents])
        success_users = np.zeros([self.n_agents], np.int32)
        
        buffer_vs_actions = np.zeros(self.n_agents)
        for b in range(self.n_agents):
            if buffer[b] == 0 and actions[b] ==1:
                buffer_vs_actions[b] = 1
                
        '''
        ch_feedback gives a list containing broadcast feedback, 
        transmitting users indices,
        successful users indices and 
        colliding users indices 
        '''
        
        # successful agents receive reward 1 
        for i in ch_feedback[2]:
            rewards_n[i] = 1
            # this also updates the packet success user variable in the main loop
            successes_n[i] +=1

        self.reward_each_agent = rewards_n
  
        if self.agents_type=='cooperative':
        # for cooperative learning, reward is same for all the agents
            rewards_n = np.sum(rewards_n)*np.ones(self.n_agents)
        
        return rewards_n
    
    def get_done_n(self):
        done_n = []
        for i in range(self.n_agents):
            done_n.append(False)
        return done_n
    
    def get_info_n(self):
        info_n = {'n': []}
        for i in range(self.n_agents):
            info_n['n'].append({})
        return info_n
    
    def get_feedback(self, actions):
        
        ch_feedback = np.zeros([self.num_channels])

        ''' calculate feedback - it can be binary or ternary or individual
            individual feedback is the function of feedback, i.e. 
            as seen by each user if their own transmission
           was success, idle (no transmission) or collided'''
        
        '''
        the function returns: 
            broadcast feedback
            indices of transmitting users
            indices of successful users and 
            indices of colliding users
            
        '''
        successful_users = np.zeros([self.n_agents], np.int32)
        colliding_users = np.zeros([self.n_agents], np.int32)
        
        # channel on which collision happened (0/1)
        collision_slot = np.zeros([self.action_space[0].n], np.int32)

        # shows how many users transmitted over each channel
        no_of_tr_over_m_channel = np.zeros([self.action_space[0].n], np.int32)    #0 for no channel access
    
        j = 0
        for each in actions:
          no_of_tr_over_m_channel[each] += 1
          j += 1
        
        channel_status = no_of_tr_over_m_channel
        #print("channel_status: ", channel_status)
        for i in range(1, self.action_space[0].n):
            if no_of_tr_over_m_channel[i] > 1:
                collision_slot[i] += 1
                no_of_tr_over_m_channel[i] = 0

        # find users that were successful and 0 if idle
        for i in range(self.n_agents):
            successful_users[i] = no_of_tr_over_m_channel[actions[i]]
            # for ternary feedback this 0 value can be replaced with 2 or something else
            if actions[i] == 0:
                successful_users[i] = 0
        
        self.success_feedback = no_of_tr_over_m_channel[1:]
        print("success_feedback: ", self.success_feedback)
        self.collision_feedback = collision_slot[1:]
        print("collision_feedback: ", self.collision_feedback)
        
        if self.fb_type=='success_broadcast':
            ch_feedback = self.success_feedback
        
        if self.fb_type=='collision_broadcast':
            # status of each channel whether there was a collision over it or not
            ch_feedback = self.collision_feedback
        
        if self.fb_type=='success_individual':
            ch_feedback = successful_users
            
        if self.fb_type=='ternary_broadcast':
            for i in range(self.n_agents):
                if self.success_feedback[i] == 1:
                    ch_feedback[i] = 1
                
                if  self.collision_feedback[i] ==1:
                    ch_feedback[i] = -1
                    
        print("successful_users", successful_users)
        
        successful_users_ind = np.where(successful_users == 1)[0]
        # check which users transmitted
        transmitting_users_ = np.where(actions !=0)[0]
        
        colliding_users = self.remove_success_user(successful_users_ind, transmitting_users_)
        print("colliding users: ", colliding_users)
        

        return list((ch_feedback, transmitting_users_, successful_users_ind, colliding_users))
    
    def reset(self):
        # Reset the state of the environment to an initial state
        buffer_left = self.reset_buffer()
        event_status_previous = np.zeros(self.n_agents, np.int32)
        
        users_reporting_events = [[]]*self.num_events
        users_buffer = self.gen_traffic([buffer_left,event_status_previous], users_reporting_events)
        self.init_buffer = users_buffer
        actions = np.zeros(self.n_agents, np.int32)
        ch_feedback = np.zeros([self.num_channels], np.int32)
        tmp_zero_array = np.array([], np.int32)
        ch_feedback = list((ch_feedback,tmp_zero_array,tmp_zero_array,tmp_zero_array))
        
        state = self.gen_state(self.init_buffer, actions, ch_feedback)
        
        return state
    
    def seed(self,seed=None):
        print("Seed = %d"%seed)
        self.np_random,seed=seeding.np_random(seed)
    
    def render(self, mode='human'):
        pass

    def close(self):
        pass
    
    def remove_success_user(self, successful_u, transmitting_u):
        
        tmp = transmitting_u.tolist()
        successful_u = successful_u.tolist()
    
        for each in successful_u:
            if each in tmp:
                tmp.remove(each)
        
        return np.array(tmp)
    
    def gen_traffic(self, buffer_previous, users_reporting_events):
        '''
        packets arrive to the system with lmb arrival rate and few users have 
        packets to send if there are more than one packet arrive per user, 
        we only keep 1 and discard others
        '''
        buffer_left = buffer_previous[0]
        self.event_status_users = buffer_previous[1]
        event_status_previous = buffer_previous[1]
        
        # packets arrival with Poisson process
        self.pkts_arr = np.random.binomial(1, self.lmb, self.n_agents)
        self.event_activation = np.random.binomial(1, self.event_prob, self.num_events)
        
        # if user has delivered the packet in previous time slot,
        # the event reporting status is set to 0
        for ev in range(self.n_agents):
            if event_status_previous[ev] ==1 and buffer_left[ev] == 0:
                self.event_status_users[ev] = 0
        
        # if no user is reporting the event then it is set to 0
        for ii in range(self.num_events):
            if not users_reporting_events[ii]:
                self.event_activation[ii] = 0
        
        
        '''ED packets if event is active'''
        for event_i in range(self.num_events):
            if self.event_activation[event_i]:
                for usr_ in users_reporting_events[event_i]:
                    if np.random.random(1)[0] < self.activ_prob_usr:
                        print("yes it is")
             
                        self.event_status_users[usr_] = 1
                        if self.pkts_arr[usr_] == 0:
                            self.pkts_arr[usr_] += 1
        print("arrived: ", self.pkts_arr)
        
        '''add new arrivals to the packets that were left in 
            the previous time slot'''
        buffer = self.pkts_arr + buffer_left
        
        '''calculate the total number of newly arrived packets'''
        self.new_packets = np.sum(self.pkts_arr)
        
        #buffer = buffer+1
        
        '''calculate packet drop for each time slot
        if packets are added by both event and regular traffic 
        then that packet is only considered ED packet'''
        self.pkt_drop = 0
        bb = 0
        for b in buffer:
            if b > 1:
                buffer[bb] = 1 
                self.pkt_drop += 1
                
            bb+=1

        return list((buffer, self.event_status_users))

    
    # creates one hot vector
    def one_hot(self, num, length):
        assert 0 <= num < length, "error"
        vec = np.zeros([length], np.int32)
        vec[num] = 1
        return vec
    
    def gen_state(self, users_buffer, actions, feedback):        
        
        '''
        state is 
         - IDs of agents if not False - otherwise IDs are not used
         - Previous action - one-hot encoded for num_channels > 1
         - feedback (for m channels) {1,0} for each channel - broadbast
         - G_n(k) whether user n's transmission was success
        
        '''
        
        success_user = np.zeros(self.n_agents, np.int32)
        bc_fb, successful_usr_ids = feedback[0], feedback[2]
        
        for usr in successful_usr_ids:
            success_user[usr] = 1
        
        obs_ = []
        
        for n in range(self.n_agents):
            if self.use_ids == True:
                obs_n = self.one_hot(n, self.n_agents)
                
            
            if self.num_channels == 1:
                if self.use_ids == True:
                    obs_n = np.append(obs_n, actions[n])
                else:
                    obs_n = actions[n]
            
            elif self.num_channels > 1:
                if self.use_ids == True:
                    obs_n = np.append(obs_n, self.one_hot(actions[n], self.num_channels+1))
                else:
                    obs_n = self.one_hot(actions[n], self.num_channels+1)
            
                obs_n = np.append(obs_n, success_user[n])
            
            # indicator whether user is associated with an event
            #obs_n = np.append(obs_n, self.event_status_users[n])
            
            for ch in bc_fb:
                obs_n = np.append(obs_n, ch)
            
            # for buffer no buffer instead of buffer size in state vector
        
            if users_buffer[0][n]:
                obs_n = np.append(obs_n, 1)
            else:
                obs_n = np.append(obs_n, 0)
         
            obs_.append(obs_n.tolist())
        return obs_

    def update_buffer(self, users_buffer, success_users_id):
        for i in success_users_id:
            users_buffer[i] = 0
        return users_buffer
    
    def reset_buffer(self):
        return np.zeros([self.n_agents], np.int32)
    
    def render(self, mode='human', close=False):
        # Render the environment to the screen
        pass

    def random_actions(self):
        x = np.random.choice(self.action_space[0].n, size=self.n_agents)
        return x

