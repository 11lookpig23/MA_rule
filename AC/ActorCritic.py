import gym, os
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
#from acmodel import Actor,Critic
from maddpg.model import CriticAC, Social
from torch.autograd import Variable
import torchsnooper

class ActorCritic:
    def __init__(self, n_agents, dim_obs,dim_actprob,setting):
        ifload = setting["ifload"]
        batch_size = setting["batch"]
        device = setting["device"]
        self.critics = []
        if ifload:
            self.actor = torch.load(setting["paraName"][1]+".pth",map_location = torch.device('cuda'))
            for j in range(n_agents):
                self.critics.append(torch.load(setting["paraName"][0]+str(j)+".pth",map_location = torch.device('cuda')))
        else:
            self.actor = Social(dim_obs, dim_actprob,setting["ActorNet"]).to(device)
            self.critics = [CriticAC(n_agents, dim_obs,
                                setting["dim_act"],setting["criticNet"]).to(device) for i in range(n_agents)]
        self.optimizerA = optim.Adam(self.actor.parameters())
        self.optimizerCs = [optim.Adam(self.critics[ag].parameters()) for ag in range(n_agents)]
        self.schedulers = [optim.lr_scheduler.StepLR(self.optimizerA,step_size = 40, gamma = 0.7)]
        for opt in self.optimizerCs:
            self.schedulers.append(optim.lr_scheduler.StepLR(opt,step_size = 40, gamma = 0.7))
        self.n_agents = n_agents
        self.emptyBuffer()
        self.device = device
    #@torchsnooper.snoop()
    def update(self):
        ### update Critic  
        for j in range(len(self.acts)):
            tem = []
            for ag in range(self.n_agents):
                state, thact = self.transTensor2(self.obss[j],self.acts[j])
                tem.append(self.critics[ag](state, thact).squeeze().unsqueeze(0) )
            self.Qvalues.append(tem)
        self.Qvalues = np.array(self.Qvalues)
        self.Qvalues = self.Qvalues.T
        self.rewards = np.array(self.rewards)
        #print(self.Qvalues)
        for ag in range(self.n_agents):
            returns = self.compute_returns(self.next_value[ag], self.rewards[:,ag], self.masks)
            returns = torch.cat(returns).detach()
            values = torch.cat(list(self.Qvalues[ag,:]))
            advantage = returns - values
            critic_loss = advantage.pow(2).mean()
            self.optimizerCs[ag].zero_grad()
            critic_loss.backward()
            self.optimizerCs[ag].step()
        ### update actor
        for ag in range(self.n_agents):
            sumR = np.array([sum(rew) for rew in self.rewards])
            R_ep = self.compute_returns(sum(self.next_value),sumR,self.masks)
            advantage = R_ep - np.array([ sum(x) for x in self.Qvalues.T ])   
            advantage = np.array([ x.squeeze().cpu().detach() for x in advantage])
            advantage = Variable(torch.Tensor(advantage)).to(self.device)
            log_prob_a = torch.cat(list(np.array(self.log_probs)[:,ag]))
            ### ????
            actor_loss = -(log_prob_a * advantage).mean()
            self.optimizerA.zero_grad()
            actor_loss.backward()
            self.optimizerA.step()
        self.emptyBuffer()
        return actor_loss
    def getPos(self,state,hei):
        busy = state[-1]
        mypos = state[:hei*2]
        mypos = mypos.reshape((hei,2))
        finalpos = np.where(mypos==1)
        return busy,finalpos
    def select_action(self,state):
        actions = []
        log_probags = []
        for ag in range(self.n_agents):
            state0 = state[ag]
            state1,_ = self.transTensor2([state0],[])
            dist = self.actor(state1)
            #print(self.actor.printprob(state1))
            action = dist.sample()

            log_prob = dist.log_prob(action).unsqueeze(0)
            actions.append(action)
            log_probags.append(log_prob)
        return dist,actions,log_probags,dist

    def storeSample(self,state,log_prob,reward_n,mask,act_n):
        self.log_probs.append(log_prob)
        self.rewards.append(reward_n)
        self.masks.append(mask)
        self.acts.append(act_n)
        self.obss.append(state.data.cpu().numpy())

    def emptyBuffer(self):
        self.log_probs = []
        self.Qvalues = []
        self.rewards = []
        self.masks = []
        self.acts = []
        self.obss = []
        self.next_value = []
    def compute_returns(self,next_value, rewards, masks, gamma=1):
        R = next_value
        returns = []
        #print("...",len(rewards),len(masks))
        for step in reversed(range(len(rewards))):
            R = rewards[step] + gamma * R * masks[step]
            returns.insert(0, R)
        return returns

    def transTensor(self,state,acts):
        ### state is obs_n
        state = np.stack(state)
        state0 = state
        sta = []
        for j in range(self.n_agents):
            sta = sta+list(state0[0])
        #state = torch.FloatTensor([sta]).to(device)
        state = torch.FloatTensor([sta]).to(self.device)
        if acts!=[]:
            thact = Variable(torch.Tensor([acts]))
        else:
            thact = []
        return state,thact
    #@torchsnooper.snoop()
    def transTensor2(self,state,acts):
        #print("===============",len(state))
        ka = [ sa.T/255 for sa in state]
        try:
            kt = [ sa.cpu().numpy() for sa in ka ]
        except AttributeError:
            kt = [ sa for sa in ka ]
        #sta.append(np.column_stack(tuple(state.data.cpu())))
        #return th.Tensor(sta).to(self.device)

        obsarr = np.column_stack(tuple(kt))
        obsinp = torch.tensor([obsarr]).to(self.device)
        obsinp = obsinp.float()
        st = obsinp
        if acts!=[]:
            thact = Variable(torch.Tensor([acts])).to(self.device)
        else:
            thact = []
        return st,thact