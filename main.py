#Actor Network

import torch
import torch.nn as nn
import torch.nn.functional as f

class Actor(nn.Module):
  def __init__(self,state_size,action_size,flatten_size,device):
    super(Actor,self).__init__()
    self.state_size = state_size
    self.action_size = action_size
    self.flatten_size = flatten_size
    self.conv1    = nn.Conv2d(1,12,kernel_size = 3, stride= 1).to(device)
    self.conv2    = nn.Conv2d(12,32,kernel_size= 3, stride= 1).to(device)
    self.lin1     = nn.Linear(self.flatten_size,32).to(device)
    self.lin2     = nn.Linear(32,64).to(device)
    self.lin3     = nn.Linear(64,self.action_size).to(device)
    self.max      = nn.MaxPool2d(kernel_size=3)
    self.init_weight()

  def forward(self,x):
    x = self.conv1(x)
    x = self.max(x)
    x = f.relu(self.lin2(f.relu(self.lin1(x))))
    x = f.softmax(self.lin3(x))
    return x

  def init_weight(self):
    for i in self.modules:
      if isinstance(i,nn.Con2d):
        nn.init.kaiming_normal()
      elif isinstance(i,nn.Linear):
        nn.init.xavier_uniform()


#Critic Network

import torch
import torch.nn as nn
import torch.nn.functional as f

class Critic(nn.Module):
  def __init__(self,state_size,action_size,flatten_size,device):
    super(Critic,self).__init__()
    self.state_size = state_size
    self.action_size = action_size
    self.flatten_size = flatten_size
    self.conv1    = nn.Conv2d(1,12,kernel_size = 3, stride= 1).to(device)
    self.conv2    = nn.Conv2d(12,32,kernel_size= 3, stride= 1).to(device)
    self.lin1     = nn.Linear(self.flatten_size,32).to(device)
    self.lin2     = nn.Linear(32,64).to(device)
    self.lin3     = nn.Linear(64,1).to(device)
    self.max      = nn.MaxPool2d(kernel_size=3)
    self.init_weight()

  def forward(self,x):
    x = self.conv1(x)
    x = self.max(x)
    x = f.relu(self.lin2(f.relu(self.lin1(x))))
    x = self.lin3(x)
    return x

  def init_weight(self):
    for i in self.modules:
      if isinstance(i,nn.Con2d):
        nn.init.kaiming_normal()
      elif isinstance(i,nn.Linear):
        nn.init.xavier_uniform()


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import csv
import cv2
import matplotlib.pyplot as plt

from torch.autograd import Variable
# from actor import Actor
# from critic import Critic

class PPO:
  def __init__(self,env,state_size,action_size,flatten_size,lamada,gamma,device,n_games,steps):
    self.state_size = state_size
    self.action_size = action_size
    self.flatten_size = flatten_size
    self.device       = device
    self.lamada       = lamada
    self.gamma        = gamma
    self.env          = env
    self.n_games      = n_games
    self.steps        = steps
    self.clip         = 0.2
    self.actor_lr     = 0.00075
    self.critic_lr    = 0.00095
    self.paths        = self.paths()
    self.actor        = Actor(self.state_size,self.action_size,self.flatten_size,self.device)
    self.critic       = Critic(self.state_size,self.action_size,self.flatten_size,self.device)

    self.actor_optim  = optim.Adam(self.actor.parameters() ,lr = self.actor_lr)
    self.critic_optim = optim.Adam(self.critic.parameters(),lr = self.critic_lr)

    self.rewards      = []
    self.values       = []
    self.next_values  = []
    self.loss         = []
    self.value_loss   = []
    self.critic_loss  = []
    self.policy_loss  = []
    self.entropy      = []
    self.ratio        = []
    self.advantage    = []

    self.csv          = csv.writer(open("main-csv.csv",'w'))

  def paths(self):
    main = "data\weights"
    actor_path = os.path.join(main,"actor.pth")
    critic_path = os.path.join(main,"critic.pth")
    return [actor_path,critic_path]

  def choose_action(self,state):
    state = cv2.cvtColor(state,cv2.COLOR_BGR2GRAY)
    state = torch.tensor(state).float().to(self.device)
    state = state.reshape(1,state.shape[0],state.shape[1])
    dist  = self.actor(state)
    return dist
  def gae_returns(self,reward,done,value,next_value):
    gae = 0
    returns = []
    for i in range(self.action_size):
      delta = reward + self.gamma*(1-done)*next_value - value
      gae   = delta * self.lamada
      returns.append(gae)
    return returns

  def appending(self,loss,actor_loss,critic_loss,value,next_value,ratio,advantage,entropy):
    self.loss.append(loss.item())
    self.actor_loss.append(actor_loss.item())
    self.critic_loss.append(critic_loss.item())
    self.values.append(value.item())
    self.next_values.append(next_value.item())
    self.ratio.append(ratio.item())
    self.advantage.append(advantage.item())
    self.entropy.appennd(entropy.item())

  def csv_writing(self,episode,step,reward,loss,actor_loss,critic_loss,value,next_value,ratio,advantage,entropy):
    self.csv.writerow([episode,step,reward,loss,actor_loss,critic_loss,value,next_value,ratio,advantage,entropy])

  def save_and_load(self):
    torch.save(self.actor.state_dict() , self.path[0])
    torch.save(self.critic.state_dict() , self.path[1])

  def value(self,state):
    state = cv2.cvtColor(state,cv2.COLOR_BGR2GRAY)
    state = torch.tensor(state).float().to(self.device)
    state = state.reshape(1,state.shape[0],state.shape[1])
    value = self.critic(state)
    return value

  def learn(self,state,next_state,reward,done,action,episode,step):
    p_action = self.choose_action(state)
    n_action = self.choose_action(next_state)
    p_old    = torch.log(p_action)
    p_new    = torch.log(n_action)
    ratio    = p_new/p_old
    value    = self.value(state)
    next_value = self.value(next_state)
    advantage= self.gae_returns(reward,done,value,next_value)
    s1       = ratio * advantage
    s2       = torch.clamp(ratio,0.8,1.2)*advantage
    actor_loss = torch.min(s1,s1)
    critic_loss= (advantage- value)**2
    entropy    = torch.special.entr(p_action)
    loss       = actor_loss - 0.5*critic_loss +0.99*entropy
    loss       = loss.mean()
    self.actor_optim.zero_grad()
    self.critic_optim.zero_grad()
    loss.backward()
    self.actor_optim.step()
    self.critic_optim.step()
    self.appending(loss,actor_loss.mean(),critic_loss.mean(),value,next_value,ratio.mean(),advantage.mean(),entropy.mean())
    self.csv_writing(episode,step,reward,loss,actor_loss.mean(),critic_loss.mean(),value,next_value,ratio.mean(),advantage.mean(),entropy.mean())
    self.save_and_load()
    return loss.item()

  def learn(self):
    for i in range(self.n_games):
      losss,rewards = 0,0
      state = self.env.reset()
      done  = False
      for j in range(self.steps):
        dist = self.choose_action(state)
        next_state,reward,done,info = self.env.step(dist.argmax(0).item())
        rewards+=reward
        if not done:
          loss = self.learn(state,next_state,reward,done,dist,i,j)
          state = next_state
          losss+=loss
        else:
          state = next_state
      losss = losss/self.steps
      print(f'episode: {i} rewards: {rewards} loss: {losss}  ')
  def testing(self):
    pass



