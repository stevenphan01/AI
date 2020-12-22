
# transform.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
# 
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018

"""
This file contains the transform function that converts the robot arm map
to the maze.
"""
import copy
from arm import Arm
from maze import Maze
from search import *
from geometry import *
from const import *
from util import *

def transformToMaze(arm, goals, obstacles, window, granularity):
    limits = arm.getArmLimit()
    angles = arm.getArmAngle()
    if len(limits) == 1:
        alphaRange = limits[0]
        betaRange = gammaRange = (0,0)
        alpha_init = angles[0]
        beta_init = gamma_init = 0
    elif len(limits) == 2:
        alphaRange, betaRange = limits[0], limits[1]
        gammaRange = (0,0)
        alpha_init, beta_init = angles[0], angles[1]
        gamma_init = 0
    else:
        alphaRange, betaRange, gammaRange = limits[0], limits[1], limits[2]
        alpha_init, beta_init, gamma_init = angles[0], angles[1], angles[2]
    
    m = int((alphaRange[1]-alphaRange[0])/(granularity)) + 1
    n = int((betaRange[1]-betaRange[0])/(granularity)) + 1
    o = int((gammaRange[1]-gammaRange[0])/(granularity)) + 1
    maze = [[[SPACE_CHAR for z in range(o)] for col in range(n)] for row in range(m)]
    a = 0
    alpha = alphaRange[0]
    alphaMax = alphaRange[1]
    while alpha <= alphaMax:
        b = 0
        beta = betaRange[0]
        betaMax = betaRange[1]
        while beta <= betaMax: 
            g = 0 
            gamma = gammaRange[0]
            gammaMax = gammaRange[1]
            while gamma <= gammaMax:
                arm.setArmAngle((alpha, beta, gamma))
                if isArmWithinWindow(arm.getArmPos(), window) == False or doesArmTouchObjects(arm.getArmPosDist(), obstacles):
                    maze[a][b][g] = WALL_CHAR 
                elif doesArmTipTouchGoals(arm.getArmPos()[-1][-1], goals):
                    maze[a][b][g] = OBJECTIVE_CHAR
                else:
                    maze[a][b][g] = SPACE_CHAR
                g += 1
                gamma += granularity
            b += 1
            beta += granularity
        a += 1
        alpha += granularity
    (start_x, start_y, start_z) = angleToIdx([alpha_init, beta_init, gamma_init], [alphaRange[0], betaRange[0], gammaRange[0]], granularity)
    maze[start_x][start_y][start_z] = START_CHAR
    return Maze(maze, [alphaRange[0], betaRange[0], gammaRange[0]], granularity)