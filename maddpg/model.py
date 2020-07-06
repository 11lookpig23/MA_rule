import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torchsnooper

class CriticAC(nn.Module):
    def __init__(self, n_agent, dim_observation, dim_action,netPara):
        super(CriticAC, self).__init__()
        self.n_agent = n_agent
        self.dim_observation = dim_observation
        self.dim_action = dim_action
        obs_dim = dim_observation * n_agent
        act_dim = self.dim_action * n_agent

        self.conv1 = nn.Conv2d(3, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        #self.fc1 = nn.Linear(2100,200)#(12 * 2*11 * 4, 100)
        self.fc2 = nn.Linear(200+act_dim, 32)
        self.fc3 = nn.Linear(32, 1)
        self.fc0 = nn.Linear(544,200)
    # obs: batch_size * obs_dim
    #@torchsnooper.snoop()
    def forward(self, obs, acts):
        x = self.pool(F.relu(self.conv1(obs)))
        x = self.pool(F.relu(self.conv2(x)))
        #size = 1
        #for k in x.shape:
        #    size = k*size
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc0(x))
        combined = th.cat([x, acts], 1)
        x = F.relu(self.fc2(combined))
        x = self.fc3(x)
        return x
class Actor(nn.Module):
    def __init__(self, dim_observation, dim_action,netPara):
        super(Actor, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 10, 3)
        #self.fc1 = nn.Linear(10 * 5 * 5, 64)
        self.fc3 = nn.Linear(64, dim_action)
        self.fc0 = nn.Linear(250,64)

    # action output between -2 and 2
    
    def forward(self, obs):
        x = F.relu(self.conv1(obs))
        x = self.pool(F.relu(self.conv2(x)))
        #size = 1
        #for k in x.shape:
        #    size = k*size
        x = x.view(x.size(0),-1)#int(size/x.size(0))
        #size1 = int(size/x.size(0))
        #print(size1,"  size1... ")
        #self.fc0 = nn.Linear(size1,64)
        x = F.relu(self.fc0(x))#(nn.Linear(int(size/x.size(0)),64)(x))
        x = self.fc3(x) 
        result = F.softmax(x)
        return result

class Social(nn.Module):
    def __init__(self, dim_observation, dim_action,netPara):
        super(Social, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 10, 3)
        self.fc1 = nn.Linear(10 * 5 * 5, 64)
        self.fc3 = nn.Linear(64, dim_action)

    # action output between -2 and 2
    def forward(self, obs):
        x = F.relu(self.conv1(obs))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 10 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.fc3(x)
        x = F.softmax(x, dim=-1)
        #print(x)
        dist = Categorical(x)
        return dist
    def printprob(self,obs):
        x = F.relu(self.conv1(obs))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 10 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.fc3(x)
        return F.softmax(x, dim=-1)


class Critic(nn.Module):
    def __init__(self, n_agent, dim_observation, dim_action,netPara):
        super(Critic, self).__init__()
        self.n_agent = n_agent
        self.dim_observation = dim_observation
        self.dim_action = dim_action
        obs_dim = dim_observation * n_agent
        act_dim = self.dim_action * n_agent

        self.conv1 = nn.Conv2d(3, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 12, 3)
        self.fc1 = nn.Linear(2100,200)#(12 * 2*11 * 4, 100)
        self.fc2 = nn.Linear(200+act_dim, 32)
        self.fc3 = nn.Linear(32, 1)
        self.fc0 = nn.Linear(544,200)
    # obs: batch_size * obs_dim
    #@torchsnooper.snoop()
    def forward(self, obs, acts):
        x = F.relu(self.conv1(obs))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,2100)
        x = F.relu(self.fc1(x))
        combined = th.cat([x, acts], 1)
        x = F.relu(self.fc2(combined))
        x = self.fc3(x)
        return x
'''
class Critic(nn.Module):
    def __init__(self, n_agent, dim_observation, dim_action,netPara):
        super(Critic, self).__init__()
        self.n_agent = n_agent
        self.dim_observation = dim_observation
        self.dim_action = dim_action
        obs_dim = dim_observation * n_agent
        act_dim = self.dim_action * n_agent
        hide = netPara[0]
        hide2 = netPara[1]
        hide3 = netPara[2]
        self.FC1 = nn.Linear(obs_dim, hide)
        self.FC2 = nn.Linear(hide+act_dim, hide2)
        self.FC3 = nn.Linear(hide2, hide3)
        self.FC4 = nn.Linear(hide3, 1)

    # obs: batch_size * obs_dim
    def forward(self, obs, acts):
        result = F.relu(self.FC1(obs))
        #print("critics...actions...",acts)
        combined = th.cat([result, acts], 1)
        result = F.relu(self.FC2(combined))
        return self.FC4(F.relu(self.FC3(result)))


class Actor(nn.Module):
    def __init__(self, dim_observation, dim_action,netPara):
        super(Actor, self).__init__()
        hide = netPara[0]
        hide2 = netPara[1]
        self.FC1 = nn.Linear(dim_observation, hide)
        self.FC2 = nn.Linear(hide, hide2)
        self.FC3 = nn.Linear(hide2, dim_action)

    # action output between -2 and 2
    def forward(self, obs):
        result = F.relu(self.FC1(obs))
        result = F.relu(self.FC2(result))
        result = F.tanh(self.FC3(result))
        result = F.softmax(result)
        #result = F.softmax(result)
        return result

class Social(nn.Module):
    def __init__(self, dim_observation, dim_action,netPara):
        super(Social, self).__init__()
        hide = netPara[0]
        hide2 = netPara[1]
        self.FC1 = nn.Linear(dim_observation, hide)
        self.FC2 = nn.Linear(hide, hide2)
        self.FC3 = nn.Linear(hide2, dim_action)

    # action output between -2 and 2
    def forward(self, obs):
        result = F.relu(self.FC1(obs))
        result = F.relu(self.FC2(result))
        result = self.FC3(result)
        dist = Categorical(F.softmax(result, dim=-1))
        return dist
    def printprob(self,obs):
        result = F.relu(self.FC1(obs))
        result = F.relu(self.FC2(result))
        result = self.FC3(result)
        return F.softmax(result, dim=-1)

'''

class Constrain(nn.Module):
    def __init__(self, dim_observation, dim_action):
        super(Constrain, self).__init__()
        hide = 12#500
        hide2 = 6#128
        self.FC1 = nn.Linear(dim_observation, hide)
        self.FC2 = nn.Linear(hide, hide2)
        self.FC3 = nn.Linear(hide2, dim_action)

    # action output between -2 and 2
    def forward(self, obs):
        result = F.relu(self.FC1(obs))
        result = F.tanh(self.FC2(result))
        result = self.FC3(result)
        #result = F.softmax(result)
        #result = F.softmax(result)
        return result