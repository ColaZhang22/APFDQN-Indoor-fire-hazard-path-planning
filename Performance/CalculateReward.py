import matplotlib.pyplot as plt
import pandas as pd
import torch
import numpy as np
import argparse
from Env.AirportEnv_v2 import Env
from Algorithm.dqn_v2 import DQN
import torch

import argparse
import os
def DrawReward(dqn):
    # state,
    fp='./Env/'
    Airport=Env(fp)
    sum_reward=0
    num=50
    Existlist = Airport.setExit(3)
    step_counter = 1
    learning_flag = True
    EPSILON=0.05
    for n in range(num):
        Firelist = Airport.setFire(2)
        episode_reward = 0
        print("Start new game")
        Airport.Begin()
        s = dqn.initialize(Airport.Curstate, Airport.Fires, Airport.Exits)
        while True:
            s_list = Airport.nextlist(Airport.Curstate)
            # APF DQN
            q_value, a, s_ = dqn.select_action(Airport.Curstate, s_list, Airport.Fires, Airport.Exits, EPSILON)
            r, done = Airport.step(s, a, s_, s_list)
            # dqn.store_transition(s, a, r, s_)
            s = s_
            episode_reward -= r
            if done:
                    break
        sum_reward-=episode_reward
    print('calculate finish')
    return sum_reward/num

        # plt.show()
# if __name__ == '__main__':
    # # -----------------------------------------------------------------------------
    # parser = argparse.ArgumentParser(description='Airport Emergency Exit Plan base on PyTorch')
    #
    # # hyperparameter
    # parser.add_argument('--envname', type=str, default='Airport')
    # parser.add_argument('--envpath', type=str, default='../Env/')
    # parser.add_argument('--save_modelpath', type=str, default='../Model/')
    # parser.add_argument('--batch_size', type=int, default=64)
    # parser.add_argument('--replay_memory_size', type=int, default=1000)
    # parser.add_argument('--episode', type=int, default=10000)
    # parser.add_argument('--lr', type=float, default=0.00025)
    # parser.add_argument('--final_epsilon', type=float, default=0.1)
    # parser.add_argument('--initial_epsilon', type=float, default=1)
    # parser.add_argument('--gamma', type=float, default=0.99)
    # parser.add_argument('--update_frequency', type=int, default=100)
    # parser.add_argument('--value_space', type=int, default=1)
    #
    # args = parser.parse_args()
    #
    #
    # file_names=os.listdir('../Model/')
    #
    # # load Alogrithm
    # dqn = DQN(args)
    #
    # # load Env
    # Airport = Env(args.envpath)
    # Existlist = Airport.setExit(3)
    #
    # step_counter = 1
    # learning_flag = True
    #
    # # parameter
    # EPSILON = args.initial_epsilon
    # EPISODE = args.episode
    # REPLAY_MEMORY_SIZE = args.replay_memory_size
    # FINAL_EPSILON = args.final_epsilon
    # INITIAL_EPSILON = args.initial_epsilon
    #
    # for i in file_names:
    #
    #     file_path='./Model/'+i
    #     # file_path = './Model/Airport4000.pt'
    #     DrawCB_path(Airport,dqn,file_path)
