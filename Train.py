import os
import sys
sys.path.append("maddpg")
from MADDPG import MADDPG
sys.path.append("..")
import random
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import torch as th
#th.cuda.set_device(1)
import os
from utility_funcs import make_video_from_image_dir
#from envs import matrixgame
from scipy.special import softmax
from torch.autograd import Variable
#from ActorCritic import ActorCritic
#from ac2 import trainAC
from nouse.cnn import CNNNet
from AC.ActorCritic import ActorCritic
from envs.ElevatorENV.Lift import Lift
from setexp import Setting
from envs.SocialDilemmaENV.social_dilemmas.envir.harvest import HarvestEnv
device = th.device("cuda" if th.cuda.is_available() else "cpu")
from maddpg.model import Critic
'''
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
'''
class Trainer:
    def __init__(self):
        self.Setting  = Setting()
        self.envname = "harvest"
        if self.envname == "lift":
            height,self.n_agents,self.n_states,self.n_actions,act_dim = self.Setting.get_elevator_setting()
            self.world = Lift(self.n_agents,height)
        elif self.envname == "harvest":
            self.n_agents,self.n_states,self.n_actions,act_dim = self.Setting.get_harvest_setting()
            self.world = HarvestEnv(num_agents=self.n_agents)
        else:
            pass
        capacity,batch_size,self.n_episode,self.max_steps,episodes_before_train = self.Setting.getTrain_Setting()
        DDPG_dict = self.Setting.DDPG_dict
        self.maddpg = MADDPG(act_dim,self.n_agents, self.n_states, self.n_actions, batch_size, capacity,
        episodes_before_train,DDPG_dict)

    def saveParaRew(self,model = "pure_ddpg",eval_value = [],baseline = {},info = {}):
        pass
    def transferHarvClean(self,data,mode):
        sta = []
        actdict = {}
        if mode == 'a':
            for i in range(len(data)):
                name = "agent-"+str(i)
                actdict[name] = data[i].item()
            return actdict
        
        elif mode == 's':
            for key,value in data.items():
                value = value.flatten()
                sta.append(value)
        elif mode == 'sc':
            for key,value in data.items():
                sta.append(value)
        elif mode == "ddpg_s":
            for key,value in data.items():
                sta.append(value)
            value0 = [sa.T/255 for sa in sta]
            sta = value0
        else:
            for key,value in data.items():
                sta.append(value)
        return sta   
    
 
    def mkdir(self,path):
    
        folder = os.path.exists(path)
    
        if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(path)            #makedirs 创建文件时如果路径不存在会创建这个路径
            print("---  new folder...  ---")
            print("---  OK  ---")
		

class maddpgTrain(Trainer):
    def __init__(self):
        Trainer.__init__(self)
        self.FloatTensor = th.cuda.FloatTensor if self.maddpg.use_cuda else th.FloatTensor        
    
    def saveParaRew(self,model = "pure_ddpg",eval_value = [],baseline = {},info = {}):
        result = {"n_agents":self.n_agents,"n_episode":self.n_episode,"max_steps":self.max_steps,"model":model,
        "envname":self.envname,"eval_step":self.Setting.n_eval_step,"info":info,"baseline":baseline,"eval":eval_value}
        np.save(self.Setting.prename[3]+self.envname+"/data/"+self.Setting.result_setting["savename"]+"showParaRew_"+".npy",result)

    def randomAgent(self):
        cnn = CNNNet()
        reward_record = []
        critic = Critic(2,4,1,[])
        for i_episode in range(60):
            obs = self.world.reset()
            total_reward = 0.0
            #def rotate_agent(self, agent_id, new_rot):
            #self.world.agents['agent-0'].update_agent_rot('UP')
            #self.world.agents['agent-1'].update_agent_rot('UP')
            for t in range(100):
                adict = {1:3,2:7}
                action = [random.randint(0,3) for i in range(self.n_agents) ]#[3,adict[random.randint(1,2)]]#[random.randint(0,7) for i in range(self.n_agents) ]
                action0 = {}
                for i in range(len(action)):
                    name = "agent-"+str(i)
                    action0[name] = action[i]
                #action0 = self.Setting.transferHarvClean(action,'a')
                obs_, reward, done, _ = self.world.step(action0)
                #print(obs_)
                '''
                sta = self.transferHarvClean(obs_,'sc')
                obsarr = np.column_stack((sta[0].T/255,sta[1].T/255))
                obsinp = th.tensor([obsarr])
                obsinp = obsinp.float()
                print("ouput........ ",critic(obsinp,Variable(th.Tensor([[1,2]]))))
                '''
                #self.mkdir("images/harvest/random2")
                #self.world.render("images/harvest/random2"+"/im"+str(t)+".png")
                #print("action0 .... ",action0,"reward .... ",reward)
                reward = th.FloatTensor(self.transferHarvClean(reward,'r')).type(self.FloatTensor)
                total_reward+= reward.sum()
            reward_record.append(total_reward)
            print("ep  ", i_episode ,"  total_reward... ",total_reward)
        print("average....",sum(reward_record)/len(reward_record))
        #self.saveParaRew(eval_value=[(173+168)/2],baseline={"random":sum(reward_record)/len(reward_record)})
    def maddpgTrainer(self,Tmode,AC):
        #####
        if Tmode=="train":
            print(" MADDPG --- trainning --- ")
            epoches = self.n_episode
        else:
            print(" MADDPG --- testing --- ")
            epoches = self.Setting.n_eval_step
        #####
        reward_record = []
        for i_episode in range(epoches):
            obs = self.world.reset()
            ################
            if self.envname!="lift":
                obs = self.transferHarvClean(obs,'ddpg_s')
            obs = np.stack(obs)
            if isinstance(obs, np.ndarray):
                obs = th.from_numpy(obs).float()
            obeyrule = True
            if i_episode>int(self.Setting.n_eval_step/2):
                obeyrule = False
            ################
            total_reward = 0.0
            rr = np.zeros((self.n_agents,))
            for t in range(self.max_steps):
                if (i_episode>20 and i_episode%24==0) and Tmode == "test":
                    self.mkdir("images/harvest/HARVEST_AG5_ACT8_ruleDDPG_/iter"+str(i_episode))
                    self.world.render("images/harvest/HARVEST_AG5_ACT8_ruleDDPG_/iter"+str(i_episode)+"/im"+str(t)+".png")
                #rule_prob = [np.zeros(self.n_actions) for i in range(self.n_agents)]#rule[i].detach().numpy() #AC.actor.printprob(obs)
                rule_prob =[  AC.actor.printprob((obsi/255).unsqueeze(0).cuda())[0] for obsi in obs ]
                #print("rule_prob.......     ",rule_prob)
                obs = obs.type(self.FloatTensor)
                ################
                if Tmode=="train":
                    action,action_prob = self.maddpg.select_action(obs,rule_prob)
                    action_prob = action_prob.data.cpu()
                else:
                    action_prob = [0]
                    action = self.maddpg.select_eval_action2(obs,rule_prob,obeyrule)
                ################
                
                ################
                if self.envname!="lift":
                    action = action.data.cpu()
                    action0 = self.transferHarvClean(action,'a')
                    obs_, reward, done, _ = self.world.step(action0)
                    reward = th.FloatTensor(self.transferHarvClean(reward,'r')).type(self.FloatTensor)
                    obs_ = self.transferHarvClean(obs_,'ddpg_s')
                else:
                    action = action.data.cpu()
                    obs_, reward, done, _ = self.world.step(action.numpy())
                    reward = th.FloatTensor(reward).type(self.FloatTensor)
                ################
                obs_ = np.stack(obs_)
                obs_ = th.from_numpy(obs_).float()
                if t != self.max_steps - 1:
                    next_obs = obs_
                else:
                    next_obs = None

                total_reward += reward.sum()
                rr += reward.cpu().numpy()
                if Tmode=="train":
                    self.maddpg.memory.push(obs.data, action_prob, next_obs, reward)
                    c_loss, a_loss = self.maddpg.update_policy()
                obs = next_obs
            self.maddpg.episode_done += 1
            print('Episode: %d, reward = %f' % (i_episode, total_reward))
            reward_record.append(total_reward)
        ###############
            if Tmode=="train" and i_episode>8 and i_episode%10==0:
                self.Setting.saveRes(reward_record,1)
            else:
                self.Setting.saveEval(reward_record,1)
        if Tmode == "test":
            print(" before.....  ",sum(reward_record[:int(epoches/2)]),"after......  ",sum(reward_record[int(epoches/2):]))


class ACTrain(Trainer):
    def __init__(self):
        Trainer.__init__(self)
        self.dim_act = 1
    def transTensor(self,next_state,acts,n_agents):
        next_state = np.stack(next_state)
        next_state0 = next_state
        sta = []
        for j in range(n_agents):
            sta = sta+list(next_state0[0])
        next_state = th.FloatTensor([sta]).to(device)
        if acts!=[]:
            thact = Variable(th.Tensor([acts]))
        else:
            thact = []
        return next_state,thact

    def trainAC(self):
        lr,batch_size,n_iters,step_n,setp_save = self.Setting.getACTrainSetting()
        device = th.device("cuda" if th.cuda.is_available() else "cpu")
        ACsettings = self.Setting.AC_dict
        ACsettings["device"] = device
        ACsettings["dim_act"] = 1
        ACsettings["batch"] = batch_size
        criticNet,ActorNet = self.Setting.getNetworkSetting()
        ACsettings["criticNet"] = criticNet
        ACsettings["ActorNet"] = ActorNet
        print("n_actions.........",self.n_actions)
        AC = ActorCritic(self.n_agents, self.n_states,self.n_actions,ACsettings)
        reward_record = []
        for iter in range(n_iters):
            for sch in AC.schedulers:
                sch.step()
            state = self.world.reset()
            ## for cleanup
            state = self.transferHarvClean(state,'sc')
            state = np.stack(state)
            done = False
            for i in range(step_n):
                if (iter>200 and iter%500==0):
                    pass
                    #self.mkdir("images/harvest/maddpg_HARVEST_AG1_ACT8_/iter"+str(iter))
                    #self.world.render("images/harvest/maddpg_HARVEST_AG1_ACT8_/iter"+str(iter)+"/im"+str(i)+".png")
                state = th.FloatTensor(state).to(device)
                dist,actions,log_probs,act_prob = AC.select_action(state)
                #print(AC.actor.printprob(state))
                #if iter==2 and i%3==0:
                #    print("act_prob... ... ",AC.actor.printprob(state[0]))
                acts0 = [act.detach() for act in actions]
                acts = self.transferHarvClean(acts0,'a')
                obs_n,reward_n,_, _ = self.world.step(acts)
                #print("reward_n.... ",reward_n)
                ## for cleanup
                reward_n = self.transferHarvClean(reward_n,'r')
                if i == step_n-1:
                    done = True
                next_state = obs_n
                ####

                ####
                AC.storeSample(state,log_probs,reward_n,1-done,acts0)
                state = next_state
                state = self.transferHarvClean(state,'sc')
                if done:
                    if iter%3==0:
                        print('Iteration: {}, Score: {}'.format(iter, np.sum(np.array(AC.rewards)) ))
                    break
            reward_record.append(np.sum(np.array(AC.rewards)))
            next_state = self.transferHarvClean(next_state,'sc')
            next_state0,thact = self.transTensor(next_state,acts0,self.n_agents)
            for ag in range(AC.n_agents):
                kt = [ sa.T/255 for sa in next_state]
                obsarr = np.column_stack(tuple(kt))
                obsinp = th.tensor([obsarr]).to(device)
                obsinp = obsinp.float()
                thact = thact.to(device)
                next_value = AC.critics[ag](obsinp, thact)
                #print("next_value.....",next_value)
                AC.next_value.append(next_value)
            if iter%1 == 0 and iter >0:
                actloss = AC.update()
                if iter%25 == 0:
                    print("action_loss  ",actloss)
            if iter%200 == 0:
                self.Setting.saveRes(reward_record,0)
            ifsave = ACsettings["ifsave"]
            if (iter)%setp_save == 0 and ifsave:
                th.save(AC.actor, ACsettings["paraName"][1]+".pth")
                for j in range(AC.n_agents):
                    th.save(AC.critics[j],ACsettings["paraName"][0]+str(j)+".pth")
        return AC

#run_maddpg = maddpgTrain()
#run_maddpg.maddpgTrainer("train")
#run_maddpg.maddpgTrainer("test")


#run_maddpg = maddpgTrain()
#run_maddpg.randomAgent()
#run_maddpg.mkdir("videos/random")
#make_video_from_image_dir("videos/random", "images/harvest/random", video_name='harvest3_trajectory', fps=4)



run_acTrain = ACTrain()
AC = run_acTrain.trainAC()

run_maddpg = maddpgTrain()
run_maddpg.maddpgTrainer("train",AC)
run_maddpg.maddpgTrainer("test",AC)

#run_acTrain.mkdir("videos/ag4/iter10")
#make_video_from_image_dir("videos/iter10", "images/harvest/iter"+str(10), video_name='harvest1_trajectory', fps=3)

#make_video_from_image_dir("videos/ag4/iter10", "images/harvest/ag4/iter"+str(10), video_name='harvest2_trajectory', fps=3)
'''
img_folder = "images/harvest/ag4/iter"+str(10)
images = [img for img in os.listdir(img_folder) if img.endswith(".png")]
namelist = [ int(im[2:-4]) for im in images ]
ind = np.argsort(np.array(namelist))
images = np.array(images)[ind]
print(images)
'''