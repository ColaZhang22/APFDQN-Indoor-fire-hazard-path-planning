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

def C_AFP(test_env,fp):
    KP = 5.0  # 引力势场增益
    ETA = 100.0  # 斥力势场增益

    def calc_attractive_potential(s,exits):
        amin = float("inf")
        for i in exits:
            d = np.hypot(s[3] - i[3], s[4] - i[4])
            if amin >= d:
                amin = d
                aid = i
        return 0.5 * KP * (amin)

    def calc_repulsive_potential(s, fires):
        # search nearest obstacle
        dmin = float("inf")
        for i in fires:
            d = np.hypot(s[3] - i[3], s[3] - i[4])
            if dmin >= d:
                dmin = d

        return 0.5 * ETA * (1.0 / dmin) ** 2

    def calc_potential_field(exits,fires, s):
        ug = calc_attractive_potential(s,exits)
        uo = calc_repulsive_potential(s,fires)
        uf = ug + uo
        return uf

    Track = []
    Airport = Env(fp)
    Existlist = Airport.setExit(3)
    Firelist = Airport.setFire(2)
    Airport.Begin()

    Airport.Fires = test_env.Fires
    Airport.Exits = test_env.Exits
    Airport.Curstate = test_env.Curstate

    Track.append(Airport.Curstate[1])
    s = Airport.Curstate
    steps = 0
    min_f_d = float("inf")
    for i in range(1000):
        f_d = Airport.calc_fire_d(Airport.Curstate)
        if f_d < min_f_d:
            min_f_d = f_d
        s_list = Airport.test_nextlist(Airport.Curstate, Track)
        while True:
            if  len(s_list) == 0:
                Airport.backup(Track)
                s_list = Airport.test_nextlist(Airport.Curstate, Track)
            else:
                for j in s_list:
                    max_field=float("-inf")
                    field =calc_potential_field(Airport.Exits,Airport.Fires,j)
                    if field>max_field:
                        max_field = field
                        next = j
                break
        done = Airport.astar_step(next)
        s = next
        Track.append(next[1])
        steps += 1
        if done:
            return Track,min_f_d

if __name__ == '__main__':
    fp_path = '../Env/'
    for i in range(100):
        path,min_f_d=C_AFP(fp_path)
        print(path)
        print(len(path))