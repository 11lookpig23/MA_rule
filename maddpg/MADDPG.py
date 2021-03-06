import os
from model import Critic, Actor, Constrain, CriticAC
import torch as th
from copy import deepcopy
from memory import ReplayMemory, Experience
from torch.optim import Adam
from randomProcess import OrnsteinUhlenbeckProcess
import torch.nn as nn
import numpy as np
from params import scale_reward
from torch.autograd import Variable
from scipy.special import softmax
import math
import torchsnooper

def soft_update(target, source, t):
    for target_param, source_param in zip(target.parameters(),
                                          source.parameters()):
        target_param.data.copy_(
            (1 - t) * target_param.data + t * source_param.data)


def hard_update(target, source):
    for target_param, source_param in zip(target.parameters(),
                                          source.parameters()):
        target_param.data.copy_(source_param.data)


class MADDPG:
    def __init__(self, act_dim ,n_agents, dim_obs, dim_act, batch_size,
                 capacity, episodes_before_train,setting):
        self.setting = setting
        self.device = th.device("cuda")
        self.actors = [Actor(dim_obs, dim_act,setting["ActorNet"]).to(self.device) for i in range(n_agents)]
        self.critics = [CriticAC(n_agents, dim_obs,#criticNet
                               dim_act,setting["criticNet"]).to(self.device) for i in range(n_agents)]
        
        ifload = setting["ifload"]
        if ifload:
            for i in range(n_agents):
                name1 = setting["paraName"][0]+str(i)+".pth"#[1]+"actor_"+setting["actor_name"]+str(i)+".pth"
                name2 = setting["paraName"][1]+str(i)+".pth"#+"critic_"+setting["critic_name"]+str(i)+".pth"
                #print(name1)
                self.actors[i].load_state_dict(th.load(name1,map_location = th.device('cuda')))
                self.critics[i].load_state_dict(th.load(name2,map_location = th.device('cuda')))
        self.actors_target = deepcopy(self.actors)
        self.critics_target = deepcopy(self.critics)
        ## Constrain........
        self.constrain = Constrain(dim_obs,2)
        self.act_dim = act_dim
        self.n_agents = n_agents
        self.n_states = dim_obs
        self.n_actions = dim_act
        self.memory = ReplayMemory(capacity)
        self.batch_size = batch_size
        self.use_cuda = th.cuda.is_available()
        self.episodes_before_train = episodes_before_train

        self.GAMMA = 0.95
        self.tau = 0.01

        self.var = [1.0 for i in range(n_agents)]
        self.critic_optimizer = [Adam(x.parameters(),
                                      lr=0.0008) for x in self.critics]
        self.actor_optimizer = [Adam(x.parameters(),
                                     lr=0.0002) for x in self.actors]

        self.constrain_optimizer = Adam(self.constrain.parameters(),lr = 0.0006)

        if self.use_cuda:
            for x in self.actors:
                x.cuda()
            for x in self.critics:
                x.cuda()
            for x in self.actors_target:
                x.cuda()
            for x in self.critics_target:
                x.cuda()
            self.constrain.cuda()
        self.steps_done = 0
        self.episode_done = 0

    def transHarv(self,state_batch):
        sta = []
        for state in state_batch:
            sta.append(np.column_stack(tuple(state.data.cpu())))
        return th.Tensor(sta).to(self.device)

    def update_policy(self):
        # do not train until exploration is enough
        if self.episode_done <= self.episodes_before_train:
            return None, None

        ByteTensor = th.cuda.ByteTensor if self.use_cuda else th.ByteTensor
        FloatTensor = th.cuda.FloatTensor if self.use_cuda else th.FloatTensor

        c_loss = []
        a_loss = []
        for agent in range(self.n_agents):
            transitions = self.memory.sample(self.batch_size)
            batch = Experience(*zip(*transitions))
            non_final_mask = ByteTensor(list(map(lambda s: s is not None,
                                                 batch.next_states)))
            # state_batch: batch_size x n_agents x dim_obs
            state_batch = th.stack(batch.states).type(FloatTensor)
            action_batch = th.stack(batch.actions).type(FloatTensor)
            reward_batch = th.stack(batch.rewards).type(FloatTensor)
            # : (batch_size_non_final) x n_agents x dim_obs
            non_final_next_states = th.stack(
                [s for s in batch.next_states
                 if s is not None]).type(FloatTensor)

            # for current agent
            #print("state_batch ...   ...  ",action_batch.shape)
            whole_state = self.transHarv(state_batch)
            #print("whole_state ...   ...  ",whole_state.shape)
            #whole_state = state_batch.view(self.batch_size, -1)
            whole_action = action_batch.view(self.batch_size, -1)
            #print("whole_action .. ... ",whole_action.shape)
            self.critic_optimizer[agent].zero_grad()
            #print("whole_action",whole_action)
            current_Q = self.critics[agent](whole_state, whole_action)

            non_final_next_actions = [
                self.actors_target[i](non_final_next_states[:,
                                                            i,
                                                            :]) for i in range(
                                                                self.n_agents)]
            non_final_next_actions = th.stack(non_final_next_actions)
            non_final_next_actions = (
                non_final_next_actions.transpose(0,
                                                 1).contiguous())

            target_Q = th.zeros(
                self.batch_size).type(FloatTensor)
            non_final_next_states = self.transHarv(non_final_next_states)
            allsize = 1
            for s in non_final_next_actions.shape:
                allsize = allsize*s
            target_Q[non_final_mask] = self.critics_target[agent](
                non_final_next_states,#.view(-1, self.n_agents * self.n_states),
                non_final_next_actions.view(int(allsize/(self.n_agents * self.n_actions)),-1)
            ).squeeze()
            # scale_reward: to scale reward in Q functions

            target_Q = (target_Q.unsqueeze(1) * self.GAMMA) + (
                reward_batch[:, agent].unsqueeze(1) * scale_reward)

            loss_Q = nn.MSELoss()(current_Q, target_Q.detach())
            loss_Q.backward()
            self.critic_optimizer[agent].step()

            self.actor_optimizer[agent].zero_grad()
            state_i = state_batch[:, agent, :]
            action_i = self.actors[agent](state_i)
            #print("action_i....",action_i)
            #actProb = [x for x in act.detach()[0]]
            #at = np.argmax(np.array(actProb))
            #act = Variable(th.Tensor([[at]]))
            ac = action_batch.clone()
            ac[:, agent, :] = action_i
            whole_action = ac.view(self.batch_size, -1)
            actor_loss = -self.critics[agent](whole_state, whole_action)
            actor_loss = actor_loss.mean()
            actor_loss.backward()
            self.actor_optimizer[agent].step()
            c_loss.append(loss_Q)
            a_loss.append(actor_loss)
        ifsave = self.setting["ifsave"]
        if self.steps_done % self.setting["targetNetstep"] == 0 and self.steps_done > 0:
            for i in range(self.n_agents):
                soft_update(self.critics_target[i], self.critics[i], self.tau)
                soft_update(self.actors_target[i], self.actors[i], self.tau)
                if self.steps_done % self.setting["save_train_step"] == 0 and ifsave: 
                    th.save(self.critics[i].state_dict(),self.setting["paraName"][1]+str(i)+".pth")
                    th.save(self.actors[i].state_dict(),self.setting["paraName"][0]+str(i)+".pth")
        return c_loss, a_loss

    def select_rule_action(self,state_batch):
        true_act = []
        rules = []
        for id in range(2):
            obs = state_batch[:,id,:]
            act = self.actors[id](obs)
            act = th.clamp(act, 0.0, 1.0)  ## ??
            act = act.detach().numpy()
            act_prob = [ [1-x[0],x[0] ]   for x in act  ] #[ 1-act[0], act[0]]
            #act_prob = Variable(th.Tensor( act_prob))
            self.constrain_optimizer.zero_grad()
            rule = self.constrain(obs)
            rules.append(rule)
            rule0 = rule.detach().numpy()

            scale_act = [ softmax(np.array(rule0[i])*np.array(act_prob[i])) for i in range(self.batch_size) ]
            #scale_act = softmax(scale_act)
            action = [ np.random.choice(2,1,p = x) for x in scale_act ]
            true_act.append(action)
        true_act = np.array(true_act).reshape(self.batch_size,2)
        return true_act,rules

    def update_rule(self):
        if self.episode_done <= self.episodes_before_train:
            return None
        FloatTensor = th.cuda.FloatTensor if self.use_cuda else th.FloatTensor
        transitions = self.memory.sample(self.batch_size)
        batch = Experience(*zip(*transitions))

        state_batch = th.stack(batch.states).type(FloatTensor)
        action_batch = th.stack(batch.actions).type(FloatTensor)

        whole_state = state_batch.view(self.batch_size, -1)
        whole_action = action_batch.view(self.batch_size, -1)        

        #for ag in range(self.n_agents):
        true_act,rules = self.select_rule_action(state_batch)
        
        if self.steps_done%2==0:
            id = 0
        else:
            id = 1

        Q = []

        for ag  in range(self.n_agents):
            Q.append( self.critics[ag](whole_state, Variable(th.Tensor( true_act))) )
        Qsum = sum(Q)
        if self.steps_done%600==0:
            print("true_act..",true_act[15])
            print("rule..",rules[id][15])
            print("Qsum..",Qsum[15])
        loss_r = -rules[id]*Qsum
        loss_r = loss_r.mean()
        loss_r.backward()
        self.constrain_optimizer.step()
        return loss_r

    def transRule(self,act):
        #try:
        #print(act)
        #act2 = [a if  for a in act]
        act2 = []
        for a in act:
            if a>0.999:
                a = 0.999
            elif a<1e-8:
                a = 1e-7
            act2.append(a)
        #rule = softmax([math.log(1-a) for a in act2])
        rule = [ 1-a for a in act ]
        return rule

    def getLaw(self,rule_prob,action_prob):
        #forbidden_prob = [rule_prob[1],rule_prob[0]]
        new_action_prob = []
        forbidden_prob = self.transRule(rule_prob)#self.reverse(rule_prob)
        for k in range(len(action_prob)):
            if action_prob[k] < forbidden_prob[k]:
                new_action_prob.append(0)
                #action_prob[k] = 0
            else:
                new_action_prob.append(action_prob[k])
        return new_action_prob
    #@torchsnooper.snoop()
    def select_action(self,state_batch,rule_prob):
        # state_batch: n_agents x state_dim
        actions_prob = th.zeros(
            self.n_agents,
            self.n_actions)#self.n_actions)
        actions = th.zeros(
            self.n_agents,
            self.act_dim)
        FloatTensor = th.cuda.FloatTensor if self.use_cuda else th.FloatTensor
        for i in range(self.n_agents):
            #print(";;;;;;;",state_batch[i, :].shape)
            sb = state_batch[i, :].detach()
            #print(".. sb  .. ",sb.unsqueeze(0).shape)
            act = self.actors[i](sb.unsqueeze(0))#.squeeze()
            #print("act....",act)
            act += th.from_numpy(
                np.random.randn(self.n_actions) * self.var[i]).type(FloatTensor)

            if self.episode_done > self.episodes_before_train and\
               self.var[i] > 0.05:
                self.var[i] *= 0.999998

            act = th.clamp(act, 0.0, 1.0).to(self.device)
            #print("act...",act)
            #actProb = [1-act[0][0],act[0][0]]
            #actProb = self.reverse(act)
            
            actProb = act.squeeze()#[x for x in act[0]]
            #print("actProb....",actProb)
            new_action_prob = self.getLaw(rule_prob[i],actProb)
            
            at = np.argmax(np.array(new_action_prob))
            determ_act = Variable(th.Tensor([[at]]))
            #print("act..prob..",actions_prob[i, :])
            actions_prob[i, :] = act
            actions[i,:] = determ_act
        self.steps_done += 1

        return actions,actions_prob

    def reverse(self,act):
        #prob = [x[0] for x in act.detach()]
        s0 = np.argsort(act)
        s1 = s0[::-1]
        new = [0]*self.n_actions
        for i in range(self.n_actions):
            new[s1[i]] = act[s0[i]]
        return new


    def select_eval_action(self, state_batch,rule_prob,rule):
        # state_batch: n_agents x state_dim
        actions = th.zeros(
            self.n_agents,
            self.n_actions)
        #FloatTensor = th.cuda.FloatTensor if self.use_cuda else th.FloatTensor
        for i in range(self.n_agents):
            sta = Variable(th.Tensor( [[0]]))
            act = self.actors[i](sta)#.squeeze()
            act = th.clamp(act, 0.0, 1.0)
            if rule:
                #actProb = [1-act[0][0],act[0][0]]
                actProb = [x[0] for x in act.detach()]
                action_prob = self.getLaw(rule_prob,actProb)
                #if law:#act[0][0]>0.88:
                at = np.argmax(np.array(action_prob))
                #print("at... ",at)
                act = Variable(th.Tensor([[at]]))
            actions[i, :] = act
        self.steps_done += 1
        return actions

    def select_eval_action2(self,state_batch,rule_prob,rule):
        # state_batch: n_agents x state_dim
        actions_prob = th.zeros(
            self.n_agents,
            self.n_actions)#self.n_actions)
        actions = th.zeros(
            self.n_agents,
            self.act_dim)
        FloatTensor = th.cuda.FloatTensor if self.use_cuda else th.FloatTensor
        for i in range(self.n_agents):
            sb = state_batch[i, :].detach()
            act = self.actors[i](sb.unsqueeze(0))#.squeeze()

            act = th.clamp(act, 0.0, 1.0)

            actProb = [x for x in act.detach()[0]]

            new_action_prob = self.getLaw(rule_prob[i],actProb)
            if rule:
                at = np.argmax(np.array(new_action_prob))#action_prob))
            else:
                at = np.argmax(np.array(actProb))
            determ_act = Variable(th.Tensor([[at]]))
            actions_prob[i, :] = act
            actions[i,:] = determ_act

        self.steps_done += 1

        return actions#,actions_prob