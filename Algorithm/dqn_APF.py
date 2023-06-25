import random
import gym
import math
import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from tqdm import tqdm
from Env import AirportEnv_APF
from Algorithm.memory_APF import ReplayMemory
from Algorithm.model_APF import Net


# 超参数
BATCH_SIZE = 32  # mini-batch大小
REPLAY_MEMORY_SIZE = 100000  # batch总大小
EPISODE = 600  # episode个数
LR = 0.001  # 学习率
FINAL_EPSILON = 0.1  # 最终e贪婪
INITIAL_EPSILON = 1  # 起始e贪婪
GAMMA = 0.99  # 奖励递减参数
TARGET_REPLACE_FREQ = 1000  # Q 目标网络的更新频率
NOOP_MAX = 30  # 初始化时无动作最大数
VALUE_SPACE = 1


#
# # 环境
# ENV_NAME = 'PongNoFrameskip-v4'
# env = MyEnv(ENV_NAME, skip=4)   ## 10.28 add my env
# N_ACTIONS = env.action_space_number

class DQN():
    def __init__(self, args):
        SAMPLE_SIZE = 4
        self.args = args
        self.evaluate_net = Net(args.value_space)
        self.target_net = Net(args.value_space)
        self.optimizer = torch.optim.RMSprop(self.evaluate_net.parameters(), lr=args.lr, alpha=0.95,
                                             eps=0.01)  ## 10.24 fix alpha and eps
        # self.optimizer=torch.optim.Adam(self.evaluate_net.parameters(), lr=args.lr)
        self.loss_func = torch.nn.MSELoss()
        self.replay_memory = ReplayMemory(args.replay_memory_size, SAMPLE_SIZE)
        self.learn_step_counter = 0
        self.evaluate_net.cuda()
        self.target_net.cuda()

    def initialize(self,point,Fires, Exits):
        force_vector=self.APF(point,Fires,Exits)
        state = torch.tensor(np.array([point[3], point[4], force_vector[0], force_vector[1]]))
        return  state

    def APF(self, c_s , fires, exits):
        alpha=0.1
        exit_distance = np.sqrt((exits[0][3] - c_s[3]) ** 2 + (exits[0][4] - c_s[4]) ** 2)
        #calculate attractive force
        fatt=np.array([0,0],dtype=float)
        for i in exits:
            a=i[3] - c_s[3]
            fatt[0]+=alpha*(i[3] - c_s[3] )
            fatt[1]+=alpha*(i[4] - c_s[4] )
            cexit_distance=np.sqrt((i[3] - c_s[3])**2+ (i[4] - c_s[4] )**2)
            if cexit_distance<exit_distance:
                exit_distance=cexit_distance
        #calculate replusive force
        beta=0.2
        frep=np.array([0,0],dtype=float)
        for i in fires:
            try:
                a=(i[3] - c_s[3])
                b=(i[4] - c_s[4])
                fire_distance=np.sqrt(a**2+b**2)
                if a==0:
                    a=0.1
                if b==0:
                    b=0.1
                frep[0] += -beta * (1 / a) * ((1 / a ** 2))
                frep[1] += -beta * (1 / b) * ((1 / b ** 2))
            except ZeroDivisionError:
                print('')
        force=fatt+frep
        one_force = [force[0] / np.sqrt(force[0] ** 2 + force[1] ** 2), force[1] / np.sqrt(force[0] ** 2 + force[1] ** 2)]
        TEST=one_force[0]**2+one_force[1]**2

        '''
        this part aim to change the state,we do not use APF
        '''


        return [exit_distance,fire_distance]
        # return one_force

    def calculate_state(self,c_s,n_s,force_vector):
        x = n_s[3] - c_s[3]
        y = n_s[3] - c_s[4]
        orientation_vector =np.array( [x / np.sqrt(x ** 2 + y ** 2),y / np.sqrt(x** 2 + y ** 2)])
        force_vector=np.array(force_vector)
        cos_sim = force_vector.dot(orientation_vector) / np.linalg.norm(force_vector) * np.linalg.norm(orientation_vector)
        return orientation_vector,cos_sim

    def select_action(self, c_s ,s_list, Fires, Exits,epsilon):


        if random.random() >= epsilon:
            # orientation_vector=self.calculate_state(c_s,s_list[0],force_vector)
            try:
                force_vector = self.APF(s_list[0], Fires, Exits)
            except IndexError:
                print('what happened')
            s_ = torch.tensor(np.array([s_list[0][3], s_list[0][4], force_vector[0], force_vector[1]]))
            s_ = s_.to(torch.float32).cuda()
            max_q = self.evaluate_net.forward(s_).cpu().data.numpy()
            choice = s_list[0]
            for s in s_list:
                # orientation_vector=[s_[2]-c_s[2],s_[3]-c_s[3]]
                # orientation_vector=self.calculate_state(c_s,s_,force_vector)
                force_vector = self.APF(s, Fires, Exits)
                state = torch.tensor(np.array([s[3], s[4], force_vector[0], force_vector[1]]))
                state = state.to(torch.float32).cuda()
                q_eval = self.evaluate_net.forward(state).cpu().data.numpy()
                if q_eval[0] >= max_q[0]:
                    max_q = q_eval
                    choice = s
                    s_= torch.tensor(np.array([s[3], s[4], force_vector[0], force_vector[1]]))
        else:
            try:
                number = random.sample(range(0, len(s_list)), 1)

                # orientation_vector,cos_sim = self.calculate_state(c_s, s_list[number[0]], force_vector)
                # state = torch.tensor(np.array([s_list[0][3], s_list[0][4], force_vector[0], force_vector[1]]))
                # state = state.to(torch.float32).cuda()
                choice=s_list[number[0]]
                s_force_vector=self.APF(choice,Fires,Exits)
                s_ = torch.tensor(np.array([choice[3], choice[4], s_force_vector[0], s_force_vector[1]]))
                s_= s_.to(torch.float32).cuda()
                max_q = self.evaluate_net.forward(s_).cpu().data.numpy()
            except ValueError:
                print('What happened')
        return max_q, choice, s_

    def test_select_action(self, c_s, s_list, Fires, Exits, epsilon):
        # orientation_vector=self.calculate_state(c_s,s_list[0],force_vector)
        try:
            force_vector = self.APF(s_list[0], Fires, Exits)
        except IndexError:
            print('what happened')
        s_ = torch.tensor(np.array([s_list[0][3], s_list[0][4], force_vector[0], force_vector[1]]))
        s_ = s_.to(torch.float32).cuda()
        max_q = self.evaluate_net.forward(s_).cpu().data.numpy()
        choice = s_list[0]
        for s in s_list:
            # orientation_vector=[s_[2]-c_s[2],s_[3]-c_s[3]]
            # orientation_vector=self.calculate_state(c_s,s_,force_vector)
            force_vector = self.APF(s, Fires, Exits)
            state = torch.tensor(np.array([s[3], s[4], force_vector[0], force_vector[1]]))
            state = state.to(torch.float32).cuda()
            q_eval = self.evaluate_net.forward(state).cpu().data.numpy()
            if q_eval[0] >= max_q[0]:
                max_q = q_eval
                choice = s
                s_= torch.tensor(np.array([s[3], s[4], force_vector[0], force_vector[1]]))

        return max_q, choice, s_

    # for state in s_list:
    #      if random.random() > epsilon:
    #          orientation_vector=0
    #          s=0
    #          q_eval = self.evaluate_net.forward(s)
    #          s_value = q_eval[0].max(0)[1].cpu().data.numpy() ## 10.21 to cpu
    #      else:
    #          action = np.asarray(random.randrange(VALUE_SPACE))
    #  return state

    def store_transition(self, s, a, r, s_):
        self.replay_memory.store(s, a, r, s_)

    def learn(self, ):

        if self.learn_step_counter % self.args.update_frequency == 0:
            self.target_net.load_state_dict(self.evaluate_net.state_dict())
            print("----------Network update----------")



        self.learn_step_counter += 1

        s_s, a_s, r_s, s__s = self.replay_memory.sample(self.args.batch_size)

        q_eval = self.evaluate_net(s_s)

        q_next = self.target_net(s__s).detach()

        # try just one network
        # q_next = self.evaluate_net(s__s).detach()
        q_target = r_s + GAMMA * q_next.view(self.args.batch_size, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        return loss

# def main():
#     dqn = DQN()
#     writer = SummaryWriter()
#
#     step_counter = 1
#     learning_flag = True
#     EPSILON = INITIAL_EPSILON
#
#     for episode in tqdm(range(EPISODE)):
#         s = env.noop_reset(NOOP_MAX)
#         episode_reward = 0
#         while True:
#             # env.render()
#             a = dqn.select_action(s, EPSILON)
#             s_, r, done, info = env.step_skip(a)
#
#             dqn.store_transition(s, a, r, s_)
#             episode_reward += r
#
#             if dqn.replay_memory.memory_counter > REPLAY_MEMORY_SIZE:
#                 EPSILON = FINAL_EPSILON
#                 if learning_flag:
#                     print("Start learning...")
#                     learning_flag = False
#                 loss = dqn.learn()
#                 writer.add_scalar('Train/loss', loss, step_counter)
#             else:
#                 EPSILON -= (INITIAL_EPSILON - FINAL_EPSILON)/REPLAY_MEMORY_SIZE
#             if done:
#                 break
#             s = s_
#             step_counter += 1
#         writer.add_scalar('Train/reward', episode_reward, episode + 1)
#
#     torch.save(dqn.evaluate_net.state_dict(), ENV_NAME + '.pkl')
#     writer.close()
#
# if __name__ == '__main__':
#     main()
