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


def Astar(test_env,fp):
    Openlist=[]
    Clostlist=[]
    Track = []

    Airport=Env(fp)
    Existlist = Airport.setExit(3)
    Firelist = Airport.setFire(2)
    Airport.Begin()

    Airport.Fires = test_env.Fires
    Airport.Exits = test_env.Exits
    Airport.Curstate = test_env.Curstate

    Track.append(Airport.Curstate[1])
    s=Airport.Curstate
    steps=0
    min_f_d=float("inf")
    for j in range(1000):
        f_d = Airport.calc_fire_d(Airport.Curstate)
        if f_d<min_f_d:
            min_f_d=f_d
        s_list = Airport.test_nextlist(s,Track)
        while True:
            if  len(s_list) == 0:
                Airport.backup(Track)
                s_list = Airport.test_nextlist(Airport.Curstate, Track)
            else:
                H = np.sqrt((Airport.Exits[0][3] - s_list[0][3]) ** 2 + (Airport.Exits[0][4] - s_list[0][4]) ** 2)
                for i in s_list:
                    #启发式函数，图的距离
                    for k in Airport.Exits:
                        Distance=np.sqrt((k[3]-i[3])**2+(k[4]-i[4])**2)
                        if Distance<=H:
                            H=Distance
                            NextPoint=i
                break

        Track.append(NextPoint[1])
        s=NextPoint
        done=Airport.astar_step(NextPoint)
        steps+=1
        if done:
            return Track,min_f_d


if __name__ == '__main__':

    fp_path='../Env/'
    for i in range(100):
        Path,min_f_d=Astar(fp_path)
        print(Path)
        print(len(Path))