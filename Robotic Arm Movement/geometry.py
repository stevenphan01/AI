# geometry.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018

"""
This file contains geometry functions that relate with Part1 in MP2.
"""

import math
import numpy as np
from const import *

def computeCoordinate(start, length, angle):
    return (start[0] + length*(math.cos(math.radians(angle))), start[1] - length*(math.sin(math.radians(angle))))

def doesArmTouchObjects(armPosDist, objects, isGoal=False):
    for arm in armPosDist:
        v1 = np.array(arm[0])
        v2 = np.array(arm[1])
        v = v2 - v1
        padding = arm[2]
        for obj in objects:
            r = obj[2]
            if isGoal == False:
                r += padding
            point = np.array([obj[0], obj[1]])
            u = v1 - point 
            a = np.dot(v,v)
            b = 2*np.dot(v,u)
            c = np.dot(u,u) - r**2 
            d = (b**2) - (4*a*c)
            if d < 0:
                continue 
            d = math.sqrt(d)
            t1 = (-b - d)/(2*a)
            t2 = (-b + d)/(2*a)
            if (t1 >= 0 and t1 <= 1) or (t2 >= 0 and t2 <= 1):
                return True 
    return False 

def inRange(p1, p2):
    (x1, y1) = p1 
    (x2, y2) = (p2[0], p2[1])
    r = p2[2]
    if math.pow(x2-x1,2) + math.pow(y2-y1,2) <= math.pow(r,2):
        return True 
    return False

def doesArmTipTouchGoals(armEnd, goals):
    for goal in goals: 
        if inRange(armEnd, goal):
            return True 
    return False

def isArmWithinWindow(armPos, window):
    width = window[0]
    height = window[1]
    for arm in armPos:
        start = arm[0]
        end = arm[1]
        if start[0] < 0 or end[0] < 0 or start[0] > width or end[0] > width:
            return False
        if start[1] < 0 or end[1] < 0 or start[1] > height or end[1] > height:
            return False     
    return True


if __name__ == '__main__':
    computeCoordinateParameters = [((150, 190),100,20), ((150, 190),100,40), ((150, 190),100,60), ((150, 190),100,160)]
    resultComputeCoordinate = [(243, 156), (226, 126), (200, 104), (57, 156)]
    testRestuls = [computeCoordinate(start, length, angle) for start, length, angle in computeCoordinateParameters]
    assert testRestuls == resultComputeCoordinate

    testArmPosDists = [((100,100), (135, 110), 4), ((135, 110), (150, 150), 5)]
    testObstacles = [[(120, 100, 5)], [(110, 110, 20)], [(160, 160, 5)], [(130, 105, 10)]]
    resultDoesArmTouchObjects = [
        True, True, False, True, False, True, False, True,
        False, True, False, True, False, False, False, True
    ]

    testResults = []
    for testArmPosDist in testArmPosDists:
        for testObstacle in testObstacles:
            testResults.append(doesArmTouchObjects([testArmPosDist], testObstacle))
            # print(testArmPosDist)
            # print(doesArmTouchObjects([testArmPosDist], testObstacle))

    print("\n")
    for testArmPosDist in testArmPosDists:
        for testObstacle in testObstacles:
            testResults.append(doesArmTouchObjects([testArmPosDist], testObstacle, isGoal=True))
            # print(testArmPosDist)
            # print(doesArmTouchObjects([testArmPosDist], testObstacle, isGoal=True))

    assert resultDoesArmTouchObjects == testResults

    testArmEnds = [(100, 100), (95, 95), (90, 90)]
    testGoal = [(100, 100, 10)]
    resultDoesArmTouchGoals = [True, True, False]

    testResults = [doesArmTickTouchGoals(testArmEnd, testGoal) for testArmEnd in testArmEnds]
    assert resultDoesArmTouchGoals == testResults

    testArmPoss = [((100,100), (135, 110)), ((135, 110), (150, 150))]
    testWindows = [(160, 130), (130, 170), (200, 200)]
    resultIsArmWithinWindow = [True, False, True, False, False, True]
    testResults = []
    for testArmPos in testArmPoss:
        for testWindow in testWindows:
            testResults.append(isArmWithinWindow([testArmPos], testWindow))
    assert resultIsArmWithinWindow == testResults

    print("Test passed\n")
