#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import collections
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

USE_WANDB = False  # if enabled, logs data on wandb server

class VDNMixer(nn.Module):
    def __init__(self):
        super(VDNMixer, self).__init__()
    def forward(self, agent_qs):
        return torch.sum(agent_qs, dim=1, keepdim=True)

class QMixer(nn.Module):
    def __init__(self, observation_space, hidden_dim=128, hx_size=64, recurrent=True):
        super(QMixer, self).__init__()
        state_size = sum([_.shape[0] for _ in observation_space])
        self.hidden_dim = hidden_dim
        self.hx_size = hx_size
        self.n_agents = len(observation_space)
        self.recurrent = recurrent

        hyper_net_input_size = state_size
        if self.recurrent:
            self.gru = nn.GRUCell(state_size, self.hx_size)
            hyper_net_input_size = self.hx_size
        self.hyper_net_weight_1 = nn.Linear(hyper_net_input_size, self.n_agents * self.hidden_dim)
        self.hyper_net_weight_2 = nn.Linear(hyper_net_input_size, self.hidden_dim)

        self.hyper_net_bias_1 = nn.Linear(hyper_net_input_size, self.hidden_dim)
        self.hyper_net_bias_2 = nn.Sequential(nn.Linear(hyper_net_input_size, self.hidden_dim),
                                              nn.ReLU(),
                                              nn.Linear(self.hidden_dim, 1))

    def forward(self, q_values, observations, hidden):
        batch_size, n_agents, obs_size = observations.shape
        state = observations.view(batch_size, n_agents * obs_size)

        x = state
        if self.recurrent:
            hidden = self.gru(x, hidden)
            x = hidden

        weight_1 = torch.abs(self.hyper_net_weight_1(x))
        weight_1 = weight_1.view(batch_size, self.hidden_dim, n_agents)
        bias_1 = self.hyper_net_bias_1(x).unsqueeze(-1)
        weight_2 = torch.abs(self.hyper_net_weight_2(x))
        bias_2 = self.hyper_net_bias_2(x)

        x = torch.bmm(weight_1, q_values.unsqueeze(-1)) + bias_1
        x = torch.relu(x)
        x = (weight_2.unsqueeze(-1) * x).sum(dim=1) + bias_2
        return x, hidden

    def init_hidden(self, batch_size=1):
        return torch.zeros((batch_size, self.hx_size))

class QNet_(nn.Module):
    def __init__(self, observation_space, action_space, hidden_dim=128, hx_size=64, recurrent=True):
        super(QNet_, self).__init__()
        self.num_agents = len(observation_space)
        self.hidden_dim = hidden_dim
        self.hx_size = hx_size
        agent_input_size = observation_space[0].shape[0]
        self.recurrent = recurrent
        
        self.fc1 = nn.Linear(agent_input_size, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hx_size)
        if self.recurrent:
            self.rnn = nn.GRUCell(self.hx_size, self.hx_size)
        self.fc3 = nn.Linear(self.hx_size, action_space[0].n)

    def forward(self, obs, hidden):
        q_values = [torch.empty(obs.shape[0], )] * self.num_agents
        next_hidden = [torch.empty(obs.shape[0], 1, self.hx_size, )] * self.num_agents
        for agent_i in range(self.num_agents):
            x = obs[:, agent_i, :]
            h_in = hidden[:, agent_i, :]
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            if self.recurrent:
                h = self.rnn(x, h_in)
                q = self.fc3(h)
                next_hidden[agent_i] = h.unsqueeze(1)
            else:
                q = self.fc3(x)
            q_values[agent_i] = q.unsqueeze(1)
        return torch.cat(q_values, dim=1), torch.cat(next_hidden, dim=1)

    def init_hidden(self, batch_size=1):
        return torch.zeros((batch_size, self.num_agents, self.hx_size))

    def get_q_values(self, obs, hidden):
        out, hidden = self.forward(obs, hidden)
        return out, hidden

class QNet(nn.Module):
    def __init__(self, observation_space, action_space, hidden_dim=128, hx_size=64, recurrent=False):
        super(QNet, self).__init__()
        self.num_agents = len(observation_space)
        self.recurrent = recurrent
        self.hidden_dim = hidden_dim
        self.hx_size = hx_size
        #for agent_i in range(self.num_agents):
        n_obs = observation_space[0].shape[0]
        setattr(self, 'agent_feature_{}'.format(0), nn.Sequential(nn.Linear(n_obs, self.hidden_dim),
                                                                            nn.ReLU(),
                                                                            nn.Linear(self.hidden_dim, self.hx_size),
                                                                            nn.ReLU()))
        if recurrent:
            setattr(self, 'agent_gru_{}'.format(0), nn.GRUCell(self.hx_size, self.hx_size))
        setattr(self, 'agent_q_{}'.format(0), nn.Linear(self.hx_size, action_space[0].n))

    def forward(self, obs, hidden):
        q_values = [torch.empty(obs.shape[0], )] * self.num_agents
        next_hidden = [torch.empty(obs.shape[0], 1, self.hx_size, )] * self.num_agents
        for agent_i in range(self.num_agents):
            x = getattr(self, 'agent_feature_{}'.format(0))(obs[:, agent_i, :])
            
            if self.recurrent:
                x = getattr(self, 'agent_gru_{}'.format(0))(x, hidden[:, agent_i, :])
                next_hidden[agent_i] = x.unsqueeze(1)
            q_values[agent_i] = getattr(self, 'agent_q_{}'.format(0))(x).unsqueeze(1)

        return torch.cat(q_values, dim=1), torch.cat(next_hidden, dim=1)
 
    def get_q_values(self, obs, hidden):
        out, hidden = self.forward(obs, hidden)
        return out, hidden

    def init_hidden(self, batch_size=1):
        return torch.zeros((batch_size, self.num_agents, self.hx_size))
