import numpy as np
import matplotlib.pyplot as plt

class Setting:
    def __init__(self):
    ##############################
    ###   save&load settings   ###
    ##############################
        xxx = "AC_HARVEST_AG5_ACT8_" #"ddpg_HARVEST_AG5_ACT8_cuda" #"maddpg_HARVEST_AG5_ACT8_"#"ac_harvest_ACT5_HARVEST5_v2_"#"ac_harvest_ag2_action4_" # ddpg_harvest_"#"ac_harvest_ag1_" #"ddpg_lift_";
        xxx1 = "HARVEST_AG5_ACT8_ruleDDPG_"
        self.choseENV = 1
        self.n_eval_step = 60
        self.envsname = ["cleanup/","harvest/","lift/"]
        self.prename = ["trainParameters/stage1/","trainParameters/stage2/","AC_results/","results/"]
        self.trainParaName_s1 =self.prename[0]+self.envsname[self.choseENV]+"actor_"+xxx
        self.trainParaName_s2 =self.prename[0]+self.envsname[self.choseENV]+"critic_"+xxx
        self.AC_dict = {"ifload":True,"ifsave":False, "actor_name":xxx,"critic_name":xxx,"paraName":(self.trainParaName_s1,self.trainParaName_s2)}
        self.result_setting = {"showPic":False,"saveRew":True,"savename":["ac_reward_"+xxx1,"rule_reward_"+xxx1]}
        self.trainParaName1 =self.prename[1]+self.envsname[self.choseENV]+"actor_"+xxx1
        self.trainParaName2 =self.prename[1]+self.envsname[self.choseENV]+"critic_"+xxx1
        criticNet,ActorNet = self.getNetworkSetting()
        self.DDPG_dict = {"ifload":False,"ifsave":True,"actor_name":xxx1,"critic_name":xxx1,"paraName":(self.trainParaName1,self.trainParaName2)
        ,"save_train_step":3000,"criticNet":criticNet,"ActorNet":ActorNet,"targetNetstep":1500}

##############################
###      ENV settings      ###
##############################

#####   Elevator   #####
    def get_elevator_setting(self):
        height = 4#8
        n_agents = 2#5
        n_states = 4*height+1
        n_actions = 3
        act_dim = 1
        return height,n_agents,n_states,n_actions,act_dim
#####   Harvest   #####
    def get_harvest_setting(self):
        HARVEST_VIEW_SIZE = 7
        n_agents = 5
        n_states = (HARVEST_VIEW_SIZE*2+1)*(HARVEST_VIEW_SIZE*2+1)*3
        n_actions = 8
        act_dim = 1
        return n_agents,n_states,n_actions,act_dim

#####   Cleanup   #####


    def saveRes(self,data,stage):
        if self.result_setting["saveRew"]:
            np.save(self.prename[stage+2]+self.envsname[self.choseENV]+"data/"+self.result_setting["savename"][stage]+".npy",data)#("rule_reward_ag3h6.npy",reward_record)
        if self.result_setting["showPic"]:
            plt.plot(range(len(data)),data)
            plt.savefig(self.prename[stage+2]+self.envsname[self.choseENV]+"figure/"+self.result_setting["savename"][stage]+".jpg")
    
    def saveEval(self,data,stage):
        if True:
            np.save(self.prename[stage+2]+self.envsname[self.choseENV]+"data/"+self.result_setting["savename"][stage]+"Eval_"+".npy",data)#("rule_reward_ag3h6.npy",reward_record)

    def getACTrainSetting(self):
        lr = 0.0007     
        batch_size = 128
        n_iter = 2#13#1001
        step_n = 20#500
        setp_save = 250
        return lr,batch_size,n_iter,step_n,setp_save

    def getNetworkSetting(self):
        criticNet = (1024,128,32)
        ActorNet = (512,64)
        return criticNet,ActorNet
##############################
### RL&train_test settings ###
##############################
    def getTrain_Setting(self):
        ## sets
        sets = [
            [7,8,1000000,30,4],
            [151,64,1000000,500,10]
            ]
        item = 1
        n_episode = sets[item][0]
        batch_size = sets[item][1]
        capacity = sets[item][2]
        max_steps = sets[item][3]
        episodes_before_train = sets[item][4]
        return capacity,batch_size,n_episode,max_steps,episodes_before_train


'''
    def rotate_agent(self, agent_id, new_rot):
        self.env.agents[agent_id].update_agent_rot(new_rot)
    self.rotate_agent('agent-0', 'DOWN')
# basic moves every agent should do
BASE_ACTIONS = {0: 'MOVE_LEFT',  # Move left
                1: 'MOVE_RIGHT',  # Move right
                2: 'MOVE_UP',  # Move up
                3: 'MOVE_DOWN',  # Move down
                4: 'STAY',  # don't move
                5: 'TURN_CLOCKWISE',  # Rotate counter clockwise
                6: 'TURN_COUNTERCLOCKWISE'}  # Rotate clockwise
HARVEST_ACTIONS.update({7: 'FIRE'})
'''
'''
    '1': [159, 67, 255],  # Purple
    '2': [2, 81, 154],  # Blue
    '3': [204, 0, 204],  # Magenta
    '4': [216, 30, 54],  # Red
    '5': [254, 151, 0],  # Orange
    '6': [100, 255, 255],  # Cyan
    '7': [99, 99, 255],  # Lavender
    '8': [250, 204, 255],  # Pink
    '9': [238, 223, 16]}  # Yellow
'''