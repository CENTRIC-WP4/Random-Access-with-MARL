#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 10:53:37 2023

@author: Muhammad Awais Jadoon
@organization: Interdigital Inc. London
"""

import numpy as np
from locations import Locations
from utils import Utils
from env import RA_Env
import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    # definitions 
    num_users = 8
    num_events = 1
    num_channels = 2
    d_th = 0.15 

    # activation probability of events 
    lmb_events = 0.00001
    # arrival rate for users - regular traffic
    lmb_users = 0.3#/num_users
    
    max_time_slots = 20

    # type of feedback to be used --- check environment for other types
    fb_type='success_broadcast'
    
    if num_users < 2:
        raise NotImplementedError("Choose number of users > 1")
    
    env = RA_Env(num_users, 
                 num_channels, num_events, lmb_users, lmb_events, 
                 fb_type, use_agent_ids=True)
    
    # initializations
    loc = Locations(num_users, num_events)
    utils = Utils(num_users, num_channels)
    
    
    '''
    get locations of devices and event epicenters and provide a matrix 
    of size num event epicenters * num of users. the distance of each device 
    from each epicenter
    '''
    distance_matrix = loc.get_distance_matrix()

    '''
    the following provides the user indices of each user that are close to each epicenter
    these users become active with probability 1 whenever the corresponding event occurs.
    d_th is the threshold distance within which the users become active
    '''
    users_reporting_events = loc.get_users_around_event(distance_matrix, d_th)

channel_m_util = np.zeros(num_channels) # utilization per channel
state = env.reset()

# to calculate the packet delay in slots for each user
pkt_delay_usrs = np.zeros(num_users)

# outputs users buffer status and status of events 
# e.g., if any user packet is associated with an event or not --> {0,1}
users_buffer = env.init_buffer         
score = 0
pkt_success_per_user = np.zeros(num_users, np.int32)
for t in range(max_time_slots):

    action = np.random.randint(2, size=num_users)
    
    print("actions: ", action)
    actions_n = []
    actions_n.append((np.array(action, np.int32), users_buffer, 
                        users_reporting_events, pkt_success_per_user))
    
    event_activation_prev = env.event_activation
    print(actions_n)
    next_state, reward, done, info = env.step(actions_n[0])
    
    score += sum(reward)/num_users
    
    for usr in range(num_users):
        if usr in env.ch_feedback[2]:
            # ch_feedback[2] gives indices of successful users
            pkt_delay_usrs[usr] = pkt_delay_usrs[usr]
        elif usr not in env.ch_feedback[2] and users_buffer[0][usr] !=0:
            pkt_delay_usrs[usr] += 1
    
    state = next_state
    print("reward", reward)
    
    # generate new packets
    users_buffer = env.new_buffer
    
    # some metrices for performance that may be used
    new_packets_arrived = env.new_packets
    print("new packets", env.new_packets)
    pkt_dropped = env.pkt_drop
    pkt_succeeded = sum(env.success_feedback)
    channel_m_util += env.success_feedback
    no_of_collisions = sum(env.collision_feedback)
    print("success indices: ", env.ch_feedback[2])
    
    for usr in  env.ch_feedback[2]:
        pkt_success_per_user[usr] += 1
    



