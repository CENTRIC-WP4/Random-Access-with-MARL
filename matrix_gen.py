#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 16:03:20 2022

@author: majadoon
"""
import numpy as np
from locations import Locations
from utils import Utils
from RA_Env import RA_Env
import pickle 

if __name__ == '__main__':
    
    np.random.seed(17) 
    # definitions
    num_users = 500
    num_events = 20
    num_channels = 2
    d_th = 0.05
    lmb_events = 0.2
    lmb_users = 0.001
    max_time_slots = 1
    
    # initializations
    loc = Locations(num_users, num_events)
    utils = Utils(num_users, num_channels)
    
    env = RA_Env(num_users, 
                 num_channels, num_events, lmb_users, lmb_events, 
                 fb_type='success_broadcast')
    
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
    

    state = env.reset()
    users_buffer = env.init_buffer
    #print("buffer before: ", users_buffer[0])
    
    user_events_correlation_matrix = np.zeros([num_users, num_events])
    
    for t in range(1000000):
        print("******************* TIME STEP ", str(t), "******************************")
        
        action = np.random.choice(num_channels+1, num_users)
        actions_n = []
        actions_n.append((action, users_buffer, users_reporting_events))
        next_state, reward, done, info = env.step(actions_n[0])
        
        for e_id, user in enumerate(users_reporting_events):
            if env.event_activation[e_id]==1:
                for usr in user:
                    user_events_correlation_matrix[usr][e_id] +=1
        
        users_buffer = env.new_buffer
    
    f = open('results/correlation_matrix_v01.pckl', 'wb')
    pickle.dump(user_events_correlation_matrix.tolist(), f)
    f.close()
    with open('results/correlation_matrix_v01.csv', 'w') as f1:
        f1.write(str(user_events_correlation_matrix.tolist()))
    with open("results/users_reporting_events.txt", "w") as f2:
        f2.write(str(users_reporting_events))
        
        
            

                