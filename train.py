from os import name

from numpy.core.fromnumeric import shape
from tensorflow.python.ops import ragged
from maze_env import Maze
from RL_brain import DeepQNetwork, EDGE
import numpy as np

def Rotate(mat):
    mshape=mat.shape
    re=np.zeros((mshape[1],mshape[0]))
    for i in range(mshape[1]):
        re[i]=mat[:,mshape[1]-i-1]
    return re

def Reverse(mat):
    mshape=mat.shape
    re=np.zeros(mshape)
    for i in range(mshape[0]):
        re[i]=mat[mshape[1]-i-1]
    return re

def Store_Transition(RL,s,r):
    mat=s.reshape((EDGE,EDGE))
    for x in range(1):
        for y in range(1):
            s=mat.reshape((EDGE*EDGE))
            RL.Store_Transition(s,r)
            mat=Rotate(mat)
        mat=Reverse(mat)

def run_maze():
    red_score=0
    blue_score=0
    for episode in range(50000):
        # initial observation
        observation = env.reset()
        turn=env.turn
        pace=0
        boards=[]
        scores=[]
        while True:
            # fresh env
            env.render()
            turn=env.turn
            observation=observation*turn
            action,action_value,action_obs = RL.choose_action(observation)
            scores.append(action_value)
            boards.append(action_obs)
            observation_, done = env.step(turn,action)
            # break while loop when end of this episode
            name={1:"Red",-1:"Blue"}
            print("{0} steps at ({1},{2}) , the value is {3}".format(name[turn],(int)(action/EDGE)+1,(action%EDGE)+1,action_value))
            pace+=1
            if done:
                env.render()
                if turn==1:
                    print("Red score!")
                    red_score+=1
                else:
                    print("BLue score!")
                    blue_score+=1
                print("red:{0} blue:{1}".format(red_score,blue_score))
                break
            #last_observation=observation
            turn=env.turn
            observation = env.Observation()
            if(pace>=EDGE*EDGE):
                break
        reward=1
        if(pace>=EDGE*EDGE):
            reward=0
        for i in range(pace):
            board=boards[i]
            if(i==pace-1):
                Store_Transition(RL,board,reward)
            elif i==pace-2:
                Store_Transition(RL,board,-0.9*reward)
            else:
                Store_Transition(RL,board,-0.9*scores[i+1])
        RL.learn()
        if episode>0 and episode%10==0:
            RL.save()

    # end of game
    print('game over')
    env.destroy()


if __name__ == "__main__":
    # maze game
    env = Maze()
    RL=DeepQNetwork("brain",
                        env.n_features,
                        learning_rate=0.001,
                        memory_size=200,
                    )
    RL.load()
    env.after(100, run_maze)
    env.mainloop()
    RL.save()
    RL.plot_cost()