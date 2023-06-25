import pandas as pd
import random
import numpy as np
import math

class Env:
    def __init__(self,fp):
        # read xlsx file
        Doordf = pd.read_excel(fp + 'Ifcdoor.xlsx')
        Spacedf = pd.read_excel(fp + 'Ifcspace.xlsx')
        Infodf=pd.read_excel(fp+'Env.xlsx')
        self.name="AirportEnv"
        self.state_space=4
        self.Spacedata=Spacedf.values
        self.Doordata=Doordf.values
        self.Envdata=Infodf.values
        self.Timestep=1
        print('-----------Initialization Start-----------')

        return


    #this function return a Exit list contain number of exit
    def setExit(self,num):
        Exitlist=[]
        for i in self.Envdata:
            #i[8] represent whether this space is available to be an Exit
            if i[8]==1:
                Exitlist.append(i[0])

        #get index of random
        ExitNo=random.sample(range(0,len(Exitlist)),num)
        ExitNo=[0,1]
        self.Exits = []
        for i in ExitNo:
            self.Exits.append(self.Envdata[Exitlist[i]-1])

        print('Env has',num,'Exits which been set in',self.Exits)
        return self.Exits

    #this function return a Firepoint list contain several number of point
    def setFire(self,num):
        self.Fires=[]
        count=0
        token=True
        while count<num:
            # get index of fire
            self.Firepoint = random.sample(range(1, len(self.Envdata)), 1)
            self.Firepoint = [37]
            for i in self.Exits:
                if  self.Firepoint[0] == i[0]:
                    token=False
                    break
            if token==True:
                count+=1
                self.Envdata[self.Firepoint[0]-1][9]=1
                self.Fires.append(self.Envdata[self.Firepoint[0]-1])

        print('Env has', num, 'Fires which been set in', self.Fires)
        return  self.Fires

    def Begin(self):
        self.Stepcounter=0
        for i in self.Envdata:
            i[9]=0
        while True:
            token = True
            # get index of fire
            self.Birthpoint = random.sample(range(0, len(self.Envdata)), 1)
            for i in self.Exits:
                if  self.Birthpoint[0] == i[1]:
                    token=False
                    break
            for i in self.Fires:
                if self.Birthpoint[0] == i[1]:
                    token = False
                    break

            if token==True:
                print('Game begin!!! Agent has been set in', self.Birthpoint[0]-1)
                self.Curstate=self.Envdata[self.Birthpoint[0]-1]
                break


        return self.Curstate

    def nextlist(self,s):

        nextlist=[]
        nextnumber=[]

        #get number of next list
        if pd.isna(s[6]):
            pass
        else:
            spacedege=s[6]
            for i in spacedege.split(','):
                nextnumber.append(int(i))
        if pd.isna(s[7]):
            pass
        else:
            dooredge = s[7]
            for i in dooredge.split(','):
                nextnumber.append(int(i))

        #get specific info about list
        for i in nextnumber:
            for j in self.Envdata:
                if i == j[1] and j[9]!=1:
                    nextlist.append(j)


                # #Add DoorEdge
                # a=i[4][1:-1]
                # if a!='': #no data
                #     try: #one data
                #         nextlist.append(int(a))
                #     except ValueError:
                #         for j in i[4][1:-1].split(','): #multi data
                #             nextlist.append(int(j))
                #
                # #Add SpaceEdge
                # b=i[5][1:-1]
                # if b != '':  # no data
                #     try:  # one data
                #         nextlist.append(int(b))
                #     except ValueError:
                #         for j in i[5][1:-1].split(','):  # multi data
                #             nextlist.append(int(j))
                # break

        return  nextlist

    def step(self,s_):

        Overcheck=False
        # Environment Change
        self.spread()

        #calculate reward funciton

        #compare s&s_ distance
        SDistance=0
        for i in self.Exits:
            Distance=np.sqrt(np.square(self.Curstate[3]-i[3])+np.square(self.Curstate[4]-i[4]))
            SDistance+=Distance
        SDistance = SDistance / len(self.Exits)

        S_Distance=0
        for i in self.Exits:
            Distance=np.sqrt(np.square(s_[3]-i[3])+np.square(s_[4]-i[4]))
            S_Distance+=Distance
        S_Distance = S_Distance / len(self.Exits)




        # check if game done
        for i in self.Exits:
            if i[1] == s_[1]:
                Overcheck = True
            pass

        # If agent cost too many timesteps to
        if self.Stepcounter >= 30:
            Overcheck = True
            print("Over step!")

        if Overcheck == True:
            Reward = 50
        else:
            Reward = (SDistance - S_Distance - self.Stepcounter) / 10

        # check if no choice
        # print(len(self.nextlist(s_)))
        if len(self.nextlist(s_)) == 0:
            Overcheck = True
            Reward = -10
            print("No Choice & Dead!")




        #set new s
        self.Curstate=s_
        self.Stepcounter+=1
        return Reward,Overcheck

    def spread(self):
        '''
        :param Model:Model represent what kind of Spread Model we adapt
        :return:
        This funciton trys to simulate fire hazard spread model， now we do not consider influence of wind
        '''
        spread_list=[]
        radius=1.2*self.Stepcounter
        for i in self.Fires:
            for j in self.Envdata:
                if j[9]!=1:
                    Distance = np.sqrt(np.square(j[3] - i[3]) + np.square(j[4] - i[4]))
                    if Distance<radius:
                        j[9] = 1
                        spread_list.append(j[0])
        # for i in spread_list:
        #     self.Fires.append(self.Envdata[i-1])
        # print("Env Fire Spread")

    # def begin(self,n1,n2):
    #     self.setExit(n1)
    #     self.setFirePoint(n2)
    #     initial_coordinate=np.array()
    #     return initial_coordinate

