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
        self.History = []
        self.step_history = []
        self.D=[]
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
        ExitNo=[0,1,2]
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
            self.Firepoint = random.randint(0, len(self.Envdata))
            for i in self.Exits:
                if  self.Firepoint == i[0]:
                    token=False
                    break
            if token==True:
                count+=1
                self.Envdata[self.Firepoint-1][9]=1
                self.Fires.append(self.Envdata[self.Firepoint-1])
            token=True


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

        self.D.append(self.Curstate)
        return self.Curstate

    def nextlist(self,s):

        nextlist=[]
        nextnumber=[]

        #get number of next list
        if pd.isna(s[6]):
            pass
        else:
            spacedege=s[6]
            try:
                for i in spacedege.split(','):
                    nextnumber.append(int(i))
            except AttributeError:
                print('a')
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

    def test_nextlist(self, s, Track):

        nextlist = []
        self.nextnumber = []

        # get number of next list
        if pd.isna(s[6]):
            pass
        else:
            spacedege = s[6]
            for i in spacedege.split(','):
                self.nextnumber.append(int(i))

        if pd.isna(s[7]):
            pass
        else:
            dooredge = s[7]
            for i in dooredge.split(','):
                self.nextnumber.append(int(i))

        self.nrlist=[]
        for i in self.nextnumber:
            # check repeat
            Repeat_toke = True
            for k in Track:
                if i == k:
                    Repeat_toke = False
                    break
            # back
            Back_toke = True
            for k in self.History:
                if i == k:
                    Back_toke = False
                    break
            if Back_toke == True and Repeat_toke == True:
                self.nrlist.append(i)

        # get specific info about list
        for i in self.nrlist:
            for j in self.Envdata:
                    if i == j[1] :
                            nextlist.append(j)

        self.D.append([s[1],self.nrlist,nextlist])
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

        return nextlist

    def backup(self,Track):

        #History
        # self.History=[]
        self.History.append(Track[-1])

        #delete the last
        Track.pop()

        for i in self.Envdata:
        #reset the state

            if i[1]==Track[-1]:
                self.Curstate=i

        self.Stepcounter  -=1


    def step(self, a, s_,s_list):

        Overcheck=False
        # Environment Change
        self.spread()

        #calculate reward funciton

        #compare s&s_ distance
        SDistance=0
        for i in self.Exits:
            Distance=np.sqrt((self.Curstate[3]-i[3])**2+(self.Curstate[4]-i[4])**2)
            SDistance+=Distance
        SDistance = SDistance / len(self.Exits)

        S_Distance=0
        for i in self.Exits:
            Distance=np.sqrt((a[3]-i[3])**2+(a[4]-i[4])**2)
            S_Distance+=Distance
        S_Distance = S_Distance / len(self.Exits)

        FDistance = 0
        for i in self.Fires:
            Distance = np.sqrt((self.Curstate[3] - i[3])**2 + (self.Curstate[4] - i[4])**2)
            FDistance += Distance
        FDistance = FDistance / len(self.Fires)

        F_Distance = 0
        for i in self.Fires:
            Distance = np.sqrt((a[3] - i[3])**2 + (a[4] - i[4])**2)
            F_Distance += Distance
        F_Distance = F_Distance / len(self.Fires)



        # check if game done
        for i in self.Exits:
            if i[1] == a[1]:
                Overcheck = True
            pass

        if Overcheck == True:
            Reward = 20
        else:
            Reward = -((SDistance - S_Distance) * 1.2 - (FDistance - F_Distance) * 0.8 )*1.5

        # If agent cost too many timesteps to
        # if self.Stepcounter >= 30 and Overcheck == False:
        #     Overcheck = True
        #     Reward=-20
        #     print("Over step!")

        # check if no choice
        # print(len(self.nextlist(s_)))
        if len(self.nextlist(a)) == 0:
            Overcheck = True
            Reward = -20
            print("No Choice & Dead!")




        #set new s
        self.Curstate=a
        self.Stepcounter+=1
        return Reward,Overcheck

    def spread(self):
        '''
        :param Model:Model represent what kind of Spread Model we adapt
        :return:
        This funciton trys to simulate fire hazard spread model， now we do not consider influence of wind
        '''
        self.spread_list=[]
        radius=1.2*self.Stepcounter
        for i in self.Fires:
            for j in self.Envdata:
                if j[9]!=1:
                    Distance = np.sqrt(np.square(j[3] - i[3]) + np.square(j[4] - i[4]))
                    if Distance<radius:
                        j[9] = 1
                        self.spread_list.append(j[0])
        # for i in spread_list:
        #     self.Fires.append(self.Envdata[i-1])
        # print("Env Fire Spread")

    # def begin(self,n1,n2):
    #     self.setExit(n1)
    #     self.setFirePoint(n2)
    #     initial_coordinate=np.array()
    #     return initial_coordinate

