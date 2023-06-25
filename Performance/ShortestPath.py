import matplotlib.pyplot as plt
import pandas as pd
import torch
import numpy as np
import argparse
from Env.AirportEnv_v2 import Env
from Algorithm.dqn_v2 import DQN
import torch
import networkx as nx

import argparse
import os

def DFS(test_env,fp,dqn):
    # state,
    Infodf = pd.read_excel(fp + 'Env.xlsx')
    Envdata = Infodf.values

    Airport=Env(fp)
    Existlist = Airport.setExit(3)
    Firelist = Airport.setFire(2)
    Airport.Begin()

    Airport.Fires=test_env.Fires
    Airport.Exits=test_env.Exits
    Airport.Curstate=test_env.Curstate

    # evaluate_net=torch.load(filenames)
    # dqn.evaluate_net = evaluate_net
    # for n in range(10):
    # Airport.D = []
    Track=[]
    # W_Track=[]
    # Airport.History=[]
    # Airport.Curstate=Envdata[29]
    Track.append(Airport.Curstate[1])
    s = dqn.initialize(Airport.Curstate, Airport.Fires, Airport.Exits)
    EPSILON=0
    episode_reward=0
    count=0
    s_list = Airport.test_nextlist(Airport.Curstate, Track)
    min_f_d = float("inf")
    # W_Track.append([Airport.Curstate[1], s_list])
    while True:
        # count+=1
        f_d = Airport.calc_fire_d(Airport.Curstate)
        if f_d<min_f_d:
            min_f_d=f_d
        # W_Track.append([Airport.Curstate[1], s_list])
        try:
            while len(s_list)==0:
                #save current history
                Airport.backup(Track)
                s_list = Airport.test_nextlist(Airport.Curstate, Track)
        except IndexError:
            print("No way can escape, sorry")
            return Track, min_f_d, count
        else:
            # a,s_=dqn.select_action(s,s_list,EPSILON)
            q_value, a, s_ = dqn.test_select_action(Airport.Curstate, s_list, Airport.Fires, Airport.Exits, EPSILON)
            r, done = Airport.test_step(s, a, s_, s_list)
            Track.append(a[1])
            s_list = Airport.test_nextlist(Airport.Curstate,Track)
            s = s_
            episode_reward += r
            if done:
                count+=1
                return Track, min_f_d, count



def Draw_path(fp, Track):
    #Begin draw
    Infodf = pd.read_excel(fp + 'Env.xlsx')
    Envdata = Infodf.values

    #Draw Line
    Tx = []
    Ty = []
    for j in Track:
        for k in Envdata:
            if j==k[1]:
                Tx.append(k[3])
                Ty.append(k[4])

    #Draw state point
    Spacex = []
    Spacey = []
    Spacez = []
    Doorx = []
    Doory = []
    Doorz = []
    for i in Envdata:
        if i[2] == 'Space':
            Spacex.append(i[3])
            Spacey.append(i[4])
            force_vector = dqn.APF(i, Airport.Fires, Airport.Exits)
            state = torch.tensor(np.array([force_vector[0], force_vector[1]]))
            state = state.to(torch.float32).cuda()
            state_value = dqn.target_net(state).cpu().data.numpy()
            Spacez.append(state_value[0])
        if i[2] == 'Door':
            Doorx.append(i[3])
            Doory.append(i[4])
            force_vector = dqn.APF(i, Airport.Fires, Airport.Exits)
            state = torch.tensor(np.array([force_vector[0], force_vector[1]]))
            state = state.to(torch.float32).cuda()
            state_value = dqn.target_net(state).cpu().data.numpy()
            Doorz.append(state_value[0])
    #Draw Exit and Fire point
    Exitx = []
    Exity = []
    for i in Envdata:
        if i[8]==1:
            Exitx.append(i[3])
            Exity.append(i[4])
    Firex = []
    Firey = []
    for i in Airport.Fires:
        Firex.append(i[3])
        Firey.append(i[4])

    '''
    ------START DRAW--------
    '''
    plt.figure(figsize=(6, 6))
    #draw edge
    G = nx.Graph()
    pos = {}
    color_map = []
    edge_map = []

    for i in Envdata:
        pos[i[1]] = [i[3], i[4]]
        G.add_node(i[1])
        if pd.isna(i[6])!=True:
            for j in i[6].split(','):
                G.add_edge(i[1], int(j), weight=1)
                edge_map.extend(['grey'])
        if  pd.isna(i[7])!=True:
            for j in i[7].split(','):
                G.add_edge(i[1], int(j), weight=1)
                edge_map.extend(['grey'])

    nx.draw_networkx_edges(G, pos, width=1, edge_color=edge_map)

    plt.scatter(Spacex, Spacey, marker='o', c=Spacez, edgecolors=['none'], s=30, label='LST', cmap='GnBu')
    plt.scatter(Doorx, Doory, marker='_', c=Doorz, s=30, label='LST', cmap='GnBu')
    plt.colorbar(shrink=1, orientation='vertical', extend='both', pad=0.015, aspect=30, label='frequency')
    plt.scatter(Exitx, Exity, c='g', marker='s', s=50)
    plt.scatter(Firex,Firey,c='r',marker='s',s=80)
    plt.plot(Tx, Ty, c='r')

    fsp='./ValuePic/'
    modelname=dqn.learn_step_counter
    name = 'Model_'+str(modelname)+'_Sample_'+str(n)+'.jpg'
    print('Draw',name,':Finish!!')

    file=fsp+name
    plt.savefig(file, dpi=200)
    plt.close()

    # plt.show()
if __name__ == '__main__':
    # -----------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='Airport Emergency Exit Plan base on PyTorch')

    # hyperparameter
    parser.add_argument('--envname', type=str, default='Airport')
    parser.add_argument('--envpath', type=str, default='../Env/')
    parser.add_argument('--save_modelpath', type=str, default='../Model/')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--replay_memory_size', type=int, default=1000)
    parser.add_argument('--episode', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=0.00025)
    parser.add_argument('--final_epsilon', type=float, default=0.1)
    parser.add_argument('--initial_epsilon', type=float, default=1)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--update_frequency', type=int, default=100)
    parser.add_argument('--value_space', type=int, default=1)

    args = parser.parse_args()


    file_names=os.listdir('../Model/')

    # load Alogrithm
    dqn = DQN(args)

    # load Env
    Airport = Env(args.envpath)
    Existlist = Airport.setExit(3)
    Firelist = Airport.setFire(2)
    Airport.Begin()


    step_counter = 1
    learning_flag = True

    # parameter
    EPSILON = args.initial_epsilon
    EPISODE = args.episode
    REPLAY_MEMORY_SIZE = args.replay_memory_size
    FINAL_EPSILON = args.final_epsilon
    INITIAL_EPSILON = args.initial_epsilon

    for i in file_names:

        file_path='../Model/'+i
        # file_path = './Model/Airport4000.pt'
        A,B,C=DFS(Airport,args.envpath,dqn)
