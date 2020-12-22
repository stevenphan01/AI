# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018

"""
This file contains search functions.
"""
# Search should return the path and the number of states explored.
# The path should be a list of tuples in the form (alpha, beta, gamma) that correspond
# to the positions of the path taken by your search algorithm.
# Number of states explored should be a number.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,astar)
# You may need to slight change your previous search functions in MP1 since this is 3-d maze

from collections import deque
from heapq import heappop, heappush

def search(maze, searchMethod):
    return {
        "bfs": bfs,
    }.get(searchMethod, [])(maze)

def constructPath(prev, start, end):
    path = []
    curr = end
    while 1:
        path.append(curr)
        if curr == start:
            path.reverse()
            return path
        curr = prev[curr]
    return []

def bfs(maze):
    start = maze.getStart()    
    frontier = []
    frontier.append(start)
    visited = {}
    visited[start] = True
    prev = {}
    while len(frontier) != 0:
        curr = frontier.pop(0)
        if (maze.isObjective(curr[0], curr[1], curr[2])):
            return constructPath(prev, start, curr)  
        for neighbor in maze.getNeighbors(curr[0], curr[1], curr[2]):
            if neighbor not in visited.keys():
                visited[neighbor] = False
            if visited[neighbor] == False:
                visited[neighbor] = True                    
                frontier.append(neighbor)
                prev[neighbor] = curr
    return None
