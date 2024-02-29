#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 19 14:01:04 2022

@author: majadoon
"""
import numpy as np
import scipy.stats

class Locations:

    def __init__(self, num_users, num_events):

        self.num_users = num_users
        self.num_events = num_events
        
        #Simulation window parameters
        self.xMin=0; self.xMax=1
        self.yMin=0; self.yMax=1
        self.xDelta=self.xMax-self.xMin
        self.yDelta=self.yMax-self.yMin #rectangle dimensions
        areaTotal=self.xDelta*self.yDelta
        
        self.users_loc = self.get_users_location()

        self.events_loc = self.get_events_location()
        
    def get_users_location(self):

        '''this function provides the location of users/agents'''
        
        xx = (self.xDelta*scipy.stats.uniform.rvs(0,1,((self.num_users,1)))
              +self.xMin).tolist() #x coordinates
       
        yy = (self.yDelta*scipy.stats.uniform.rvs(0,1,((self.num_users,1)))
              +self.yMin).tolist() # y coordinates
        
        usr_loc = zip(xx, yy)
        usr_loc_ = list(usr_loc)
        usr_loc_ = np.reshape(usr_loc_, [self.num_users, 2])
      
        return usr_loc_
    
    def get_events_location(self):
        
        '''this function provides the location of event epicentres'''
        #uncomment the following if different locations of events are desired

        xx = (self.xDelta*scipy.stats.uniform.rvs(0,1,((self.num_events,1)))
              +self.xMin).tolist() #x coordinates
  
        yy = (self.yDelta*scipy.stats.uniform.rvs(0,1,((self.num_events,1)))
              +self.yMin).tolist() # y coordinates
        
        event_loc = zip(xx, yy)
        event_loc_ = list(event_loc)
        event_loc_ = np.reshape(event_loc_, [self.num_events, 2])
        
        return event_loc_
    
    def get_distance_matrix(self):
        # finding the euclidean distance between event epicentres and users
        dist_matrix = np.zeros([self.num_events, self.num_users])
        ev_row = 0
        for ev_loc in self.events_loc:
            usr_col = 0
            for usr_loc in self.users_loc:
                dist_matrix[ev_row][usr_col] = np.sqrt(np.sum(np.square(ev_loc-usr_loc)))
                usr_col+=1
            ev_row+=1
            
        return dist_matrix
    
    def get_users_around_event(self, dist_matrix, d_th):
        '''
        indices of users are are closer to event epicenter
        this closeness is given by a threshold distance d_th of user from epicenter
        '''
        users_reporting_events = []
        for event_i in dist_matrix:
            usr_indices = np.where(event_i < d_th)[0]
            
            users_reporting_events.append(usr_indices.tolist())
        return users_reporting_events