#!/usr/bin/python
# -*- coding: UTF-8 -*-
 
import numpy as np
import time
import sys

from numpy.core.shape_base import block
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

    
UNIT = 30   # pixels
MAZE_H = 10  # grid height
MAZE_W = 10  # grid width

class Maze(tk.Tk, object):
    def __init__(self,play=False):
        super(Maze, self).__init__()
        self.n_features = MAZE_H*MAZE_W
        self.title('Chess')
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_H * UNIT))
        self.state=[[0 for j in range(MAZE_W)] for i in range(MAZE_H)]
        self._build_maze(play)
        self.turn=1
        self.debug=False
        self.gameover=False
        
    def callback(self,event):
        #self.turn=-self.turn
        j,i=(int)(event.x/UNIT),(int)(event.y/UNIT)
        _,done=self.step(self.turn,action=i*MAZE_W+j)
        if done:
            self.reset()

    def Debug(self,event):
        self.debug=not self.debug
        print(self.debug)

    def _build_maze(self,play=False):
        self.canvas = tk.Canvas(self, bg='white',
                           height=MAZE_H * UNIT,
                           width=MAZE_W * UNIT)
        #create lines
        for i in range(1,MAZE_H):
            x0=0
            x1=MAZE_W*UNIT
            y0=i*UNIT
            y1=y0
            self.canvas.create_line(x0,y0,x1,y1)
        for i in range(1,MAZE_W):
            x0=i*UNIT
            x1=x0
            y0=0
            y1=MAZE_H*UNIT
            self.canvas.create_line(x0,y0,x1,y1)
        self.rect=[[self.canvas.create_rectangle(j*UNIT+1,i*UNIT+1,(j+1)*UNIT-1,(i+1)*UNIT-1,fill="white") for j in range(MAZE_W)] for i in range(MAZE_H)]
        #self.canvas.bind("<Button -1>",self.callback)
        if play:
            self.canvas.bind("<Button -1>",self.callback)
        else:
            self.canvas.bind("<Button -1>",self.Debug)
        self.debug=play
        self.canvas.pack()

    def reset(self):
        self.update()
        self.turn=1
        #time.sleep(0.1)
        self.canvas.delete(self.rect)
        self.state=[[0 for j in range(MAZE_W)] for i in range(MAZE_H)]
        self.gameover=False
        return self.Observation()
        
    def check1(self,i,j):
        if MAZE_W-1-j+1<5:
            return False
        player=self.state[i][j]
        if player==0:
            return False
        for p in range(5):
            if(self.state[i][j+p]!=player):
                return False
        return True

    def check2(self,i,j):
        if MAZE_H-1-i+1<5:
            return False
        player=self.state[i][j]
        if player==0:
            return False
        for p in range(5):
            if(self.state[i+p][j]!=player):
                return False
        return True

    def check3(self,i,j):
        if MAZE_H-1-i+1<5 or MAZE_H-1-j+1<5:
            return False
        player=self.state[i][j]
        if player==0:
            return False
        for p in range(5):
            if(self.state[i+p][j+p]!=player):
                return False
        return True
    
    def check4(self,i,j):
        if MAZE_H-1-i+1<5 or j<4:
            return False
        player=self.state[i][j]
        if player==0:
            return False
        for p in range(5):
            if(self.state[i+p][j-p]!=player):
                return False
        return True
    '''
    def Observation(self):
        npa=np.array(self.state)
        re=to_categorical(npa,num_classes=3)
        return re
    '''
    def Observation(self):
        re=np.array(self.state).reshape((self.n_features))
        return re

    def step(self,player,action):
        i=(int)(action/MAZE_W)
        j=action-i*MAZE_W
        res=self.Observation()
        if self.state[i][j]!=0:
            return res,False
        self.state[i][j]=player
        self.turn=-self.turn
        res=self.Observation()
        for i in range(MAZE_H):
            for j in range(MAZE_W):
                if self.state[i][j]==0:
                    continue
                elif self.check1(i,j) or self.check2(i,j) or self.check3(i,j) or self.check4(i,j):
                    self.gameover=True
                    return res,True
        return res,False

    def render(self):
        if(self.debug):
            time.sleep(0.3)
        for i in range(MAZE_H):
            for j in range(MAZE_W):
                dict={0:"white",1:"red",-1:"blue"}
                self.canvas.delete(self.rect[i][j])
                if self.state[i][j]==0:
                    continue
                self.rect[i][j]=self.canvas.create_rectangle(j*UNIT+1,i*UNIT+1,(j+1)*UNIT-1,(i+1)*UNIT-1,fill=dict[self.state[i][j]])
        #self.canvas.pack()
        self.update()
'''
if __name__=="__main__":
    env=Maze()
    #env.step(player=1,action=50)
    
    while True:
        env.render()
        #print(env.turn)
        #print("fuck pps")
    env.mainloop()
'''