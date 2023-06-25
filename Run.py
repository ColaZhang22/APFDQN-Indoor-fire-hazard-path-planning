from Env.AirportEnv_v2 import Env
from Algorithm.dqn_v2 import DQN
from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
from tqdm import tqdm
import torch
from Algorithm.memory_APF import ReplayMemory
from Algorithm.model_APF import Net
from Algorithm.State_Img import DrawCB_statevalue
from Algorithm.Reward_Img import Draw_reward,Draw_distance
from Performance.ShortestPath import DFS,Draw_path
from ComparedAlgorithm.Astar import Astar
from ComparedAlgorithm.AFP import C_AFP
from ComparedAlgorithm.Comparison import compare
import argparse
import os

def main(args):

    #load Alogrithm
    dqn = DQN(args)

    #load Env
    Airport = Env(args.envpath)
    Existlist = Airport.setExit(args.exit_num)

    #load logger
    writer = SummaryWriter(log_dir='./runs/APF-DQN/{}_FIRE_{}_EXIT_APF_DQN_0'.format(args.fire_num,args.exit_num))
    writer_Astar = SummaryWriter(log_dir='./runs/APF-DQN/{}_FIRE_{}_EXIT_Astar_0'.format(args.fire_num, args.exit_num))
    writer_APF = SummaryWriter(log_dir='./runs/APF-DQN/{}_FIRE_{}_EXIT_APF_0'.format(args.fire_num, args.exit_num))

    step_counter = 1
    learning_flag = True

    #parameter
    EPSILON = args.initial_epsilon
    EPISODE = args.episode
    REPLAY_MEMORY_SIZE=args.replay_memory_size
    FINAL_EPSILON=args.final_epsilon
    INITIAL_EPSILON=args.initial_epsilon
    # A1, A2, A3 = [], [], []
    # S1, S2, S3 = [], [], []

    #
    REWARD_LIST=[]
    EPISODE_LIST=[]

    for episode in tqdm(range(EPISODE)):
        Firelist = Airport.setFire(2)
        episode_reward = 0
        print("Start new game")
        Airport.Begin()
        s = dqn.initialize(Airport.Curstate, Airport.Fires, Airport.Exits)
        while True:

            # env.render()
            s_list=Airport.nextlist(Airport.Curstate)
            #DQN
            # a,s_=dqn.select_action(s,s_list,EPSILON)
            #APF DQN
            q_value, a , s_ = dqn.select_action(Airport.Curstate, s_list, Airport.Fires, Airport.Exits, EPSILON)
            r, done = Airport.step(s, a, s_, s_list)
            dqn.store_transition(s, a, r, s_)
            s = s_
            episode_reward += r

            if dqn.replay_memory.memory_counter > REPLAY_MEMORY_SIZE:
                EPSILON = FINAL_EPSILON
                if learning_flag:
                    print("---------------Start learning---------------")
                    learning_flag = False


            if done:
                print("\nGame Over, Timestep consume:", Airport.Stepcounter,'Episode Rewards:',episode_reward,'EPSILON:',EPSILON)
                # EPISODE_LIST.append(episode)
                # REWARD_LIST.append(episode_reward/Airport.Stepcounter)
                EPSILON -= (INITIAL_EPSILON - FINAL_EPSILON) / REPLAY_MEMORY_SIZE
                # EPSILON = args.initial_epsilon
                break
            step_counter += 1


        # if learning_flag == False:
        #     Model_AV_Reward = DrawReward(dqn)
        #     EPISODE_LIST.append(episode)
        #     REWARD_LIST.append(Model_AV_Reward)
        #
        # if dqn.learn_step_counter % 100 == 0:
        #     Draw_reward(EPISODE_LIST, REWARD_LIST, dqn.learn_step_counter, EPSILON)


        # if learning_flag == False:
        if learning_flag == False and dqn.learn_step_counter % 100 == 0:
            a1,a2,a3,s1,s2,s3=compare(args.envpath,dqn)
            writer_Astar.add_scalar('D2F',a1,global_step=dqn.learn_step_counter)
            writer_APF.add_scalar('D2F',a2,global_step=dqn.learn_step_counter)
            writer.add_scalar('D2F', a3, global_step=dqn.learn_step_counter)
            writer_Astar.add_scalar('STEP', s1, global_step=dqn.learn_step_counter)
            writer_APF.add_scalar('STEP', s2, global_step=dqn.learn_step_counter)
            writer.add_scalar('STEP', s3, global_step=dqn.learn_step_counter)
            # Draw_reward(EPISODE_LIST, REWARD_LIST, dqn.learn_step_counter, EPSILON)

        if learning_flag == False:
            # EPSILON = args.initial_epsilon
            loss = dqn.learn()
            # print("Episode:", episode, " loss=", loss,"Memory Account:",dqn.replay_memory.memory_counter)
            writer.add_scalar('Train/loss', loss, step_counter)

        if episode % 1000==0:
            file=args.save_modelpath+args.envname+str(episode)+'.pt'
            torch.save(dqn.target_net, file)
            print('Model has been saved')

    # Draw_distance(A1,A2,A3,'./DistancePic/Episode_Distance_' + 'Diff_Al' + '.jpg')
    # Draw_distance(S1,S2,S3,'./DistancePic/Episode_Step_' + 'Diff_Al' + '.jpg')
    # print("Plot-Distance+Step finish!")
    #
    # for i in file_names:
    #
    #     file_path=args.save_modelpath+i
    #     # file_path = './Model/Airport4000.pt'
    #     DrawCB_path(Airport,dqn,file_path)
    #
    # writer.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Airport Emergency Exit Plan base on PyTorch')

    #hyperparameter
    parser.add_argument('--envname',type=str,default='Airport')
    parser.add_argument('--envpath',type=str,default='./Env/')
    parser.add_argument('--save_modelpath', type=str, default='./Model/')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--replay_memory_size',type=int, default=1000)
    parser.add_argument('--episode',type=int,default=5001)
    parser.add_argument('--lr',type=float,default=0.00025)
    parser.add_argument('--final_epsilon',type=float,default=0.1)
    parser.add_argument('--initial_epsilon',type=float,default=10)
    parser.add_argument('--gamma',type=float,default=0.99)
    parser.add_argument('--update_frequency',type=int,default=100)
    parser.add_argument('--value_space',type=int, default=1)
    parser.add_argument('--exit_num',type=int,default=3)
    parser.add_argument('--fire_num', type=int, default=2)
    args = parser.parse_args()

    main(args)




