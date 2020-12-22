# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Michael Abir (abir2@illinois.edu) on 08/28/2018

"""
This is the main entry point for MP1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# Search should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi,fast)

import heapq

def search(maze, searchMethod):
    return {
        "bfs": bfs,
        "astar": astar,
        "astar_corner": astar_corner,
        "astar_multi": astar_multi,
        "fast": fast,
    }.get(searchMethod)(maze)


def bfs(maze):
    start = maze.getStart()    
    end = maze.getObjectives()[0]
    
    frontier = []
    frontier.append(start)
    
    visited = {}
    visited[start] = True
    
    prev = {}
    while len(frontier) != 0:
        curr = frontier.pop(0)
        if curr == end:
            return constructPath(prev, start, end)  
        for neighbor in maze.getNeighbors(curr[0], curr[1]):
            if neighbor not in visited.keys():
                visited[neighbor] = False
            if visited[neighbor] == False:
                visited[neighbor] = True                    
                frontier.append(neighbor)
                prev[neighbor] = curr
    return []

def astar(maze):
    start = maze.getStart()
    end = maze.getObjectives()[0]
    
    g = {}
    g[start] = 0
    
    f = {}
    f[start] = g[start] + manhattan(start, end)
    
    frontier = []
    heapq.heappush(frontier, (f[start], start))
    
    prev = {}
    while len(frontier) != 0:
        curr = heapq.heappop(frontier)
        if curr[1] == end:
            return constructPath(prev, start, end)
        for neighbor in maze.getNeighbors(curr[1][0], curr[1][1]):
            if neighbor not in g.keys():
                g[neighbor] = float('inf')
            g_score = g[curr[1]] + 1
            if g_score < g[neighbor]:
                g[neighbor] = g_score
                f[neighbor] = g[neighbor] + manhattan(neighbor, end)
                prev[neighbor] = curr[1]
                heapq.heappush(frontier, (f[neighbor], neighbor))
    return []

def astar_corner(maze):
    start = (getMST(maze.getStart(), maze.getObjectives()), 0, maze.getStart(), tuple())
    objectives = maze.getObjectives()
    g = {}
    g[maze.getStart(), tuple()] = 0
    h = {}
    h[maze.getStart(), tuple()] = start[0]
    frontier = []
    #idx                            0                   1              2                 3
    #state representation: (fscore of current, gscore from previous, (x,y), dots found so far in a tuple)
    heapq.heappush(frontier, (h[maze.getStart(),tuple()], 0, maze.getStart(), tuple()))
    prev = {}
    prev[start] = None
    while len(frontier) != 0:      
        curr = heapq.heappop(frontier)
        curr_found = curr[3][:] #make a copy so we don't modify the current state's list of dots found
        remaining = objectives[:]
        #curr node is a dot, so add it to the dots found tuple, and remove it from the "remaining" tuple
        temp = list(curr[3]) 
        for found in temp:
            remaining.remove(found)
        if curr[2] in remaining:
            temp.append(curr[2])
            remaining.remove(curr[2])
        temp = tuple(temp)
        #found 4 dots construct the path
        if len(temp) == len(objectives):
            return constructPath_Multi(prev, start, curr)
        #otherwise, consider the neighbors 
        for neighbor in maze.getNeighbors(curr[2][0], curr[2][1]):
            #calculator the fscore for all the neighbors
            neighbor_key = (neighbor, temp)
            if neighbor_key not in h.keys():
                hscore = getMST(neighbor, remaining)
                h[neighbor_key] = hscore
            else:
                hscore = h[neighbor_key]
            gscore = curr[1] + 1
            fscore = gscore + hscore            
            if neighbor_key not in g.keys() or fscore < g[neighbor_key] + h[neighbor_key]:
                g[neighbor_key] = gscore
                h[neighbor_key] = hscore
                prev[(fscore, gscore, neighbor, temp)] = (curr[0], curr[1], curr[2], curr_found)
                heapq.heappush(frontier, (fscore, gscore, neighbor, temp))
    return []


def astar_multi(maze):    
    astar_dict = {}
    objectives = maze.getObjectives()
    g = {}
    g[maze.getStart(), tuple()] = 0
    h = {}
    h[maze.getStart(), tuple()] = getMST_Multi(maze, astar_dict, maze.getStart(), maze.getObjectives()) #+ nearestDot(maze, astar_dict, maze.getStart(), maze.getObjectives())[1]
    frontier = []
    start = (h[maze.getStart(), tuple()], 0, maze.getStart(), tuple())
    
    #idx                            0                    1               2                 3
    #state representation:   (fscore of current,     gscore current,   (x,y), dots found so far in a tuple)
    heapq.heappush(frontier, (h[maze.getStart(),tuple()], 0, maze.getStart(), tuple()))
    prev = {}
    prev[start] = None
    while len(frontier) != 0:
        curr = heapq.heappop(frontier)           
        curr_found = curr[3][:] #make a copy so we don't modify the current state's list of dots found
        remaining = objectives[:]
        #curr node is a dot, so add it to the dots found tuple, and remove it from the "remaining" tuple
        new_found = list(curr[3]) 
        for found in new_found:
            remaining.remove(found)
        if curr[2] in remaining:
            new_found.append(curr[2])
            remaining.remove(curr[2]) 
        #found 4 dots construct the path
        if len(new_found) == len(objectives):
            #print(astar_dict)
            return constructPath_Multi(prev, start, curr)
        new_found = tuple(new_found)        
        #otherwise, consider the neighbors 
        for neighbor in maze.getNeighbors(curr[2][0], curr[2][1]):
            #calculator the fscore for all the neighbors
            neighbor_key = (neighbor, new_found)
            if neighbor_key not in h.keys():
                hscore = getMST_Multi(maze, astar_dict, neighbor, remaining) #+ nearestDot(maze, astar_dict, neighbor, remaining)[1]
                h[neighbor_key] = hscore
            else:
                hscore = h[neighbor_key]
            gscore = curr[1] + 1
            fscore = gscore + hscore 
            if [neighbor] == remaining:
                prev[(fscore, gscore, neighbor, new_found)] = (curr[0], curr[1], curr[2], curr_found)
                return constructPath_Multi(prev, start, (fscore, gscore, neighbor, new_found))
            if neighbor_key not in g.keys() or fscore < g[neighbor_key] + h[neighbor_key]:
                g[neighbor_key] = gscore
                h[neighbor_key] = hscore
                prev[(fscore, gscore, neighbor, new_found)] = (curr[0], curr[1], curr[2], curr_found)
                heapq.heappush(frontier, (fscore, gscore, neighbor, new_found))
    return []

def fast(maze):
    astar_dict = {}
    objectives = maze.getObjectives()
    g = {}
    g[maze.getStart(), tuple()] = 0
    h = {}
    h[maze.getStart(), tuple()] = getMST_Fast(maze, astar_dict, maze.getStart(), maze.getObjectives()) #+ nearestDot(maze, astar_dict, maze.getStart(), maze.getObjectives())[1]
    frontier = []
    start = (h[maze.getStart(), tuple()], 0, maze.getStart(), tuple())
    
    #idx                            0                    1               2                 3
    #state representation:   (fscore of current,     gscore current,   (x,y), dots found so far in a tuple)
    heapq.heappush(frontier, (h[maze.getStart(),tuple()], 0, maze.getStart(), tuple()))
    prev = {}
    prev[start] = None
    while len(frontier) != 0:
        curr = heapq.heappop(frontier)           
        curr_found = curr[3][:] #make a copy so we don't modify the current state's list of dots found
        remaining = objectives[:]
        #curr node is a dot, so add it to the dots found tuple, and remove it from the "remaining" tuple
        new_found = list(curr[3]) 
        for found in new_found:
            remaining.remove(found)
        if curr[2] in remaining:
            new_found.append(curr[2])
            remaining.remove(curr[2]) 
        #found 4 dots construct the path
        if len(new_found) == len(objectives):
            #print(astar_dict)
            return constructPath_Multi(prev, start, curr)
        new_found = tuple(new_found)        
        #otherwise, consider the neighbors 
        for neighbor in maze.getNeighbors(curr[2][0], curr[2][1]):
            #calculator the fscore for all the neighbors
            neighbor_key = (neighbor, new_found)
            if neighbor_key not in h.keys():
                hscore = getMST_Fast(maze, astar_dict, neighbor, remaining) #+ nearestDot(maze, astar_dict, neighbor, remaining)[1]
                h[neighbor_key] = hscore
            else:
                hscore = h[neighbor_key]
            gscore = curr[1] + 1
            fscore = gscore + hscore 
            if [neighbor] == remaining:
                prev[(fscore, gscore, neighbor, new_found)] = (curr[0], curr[1], curr[2], curr_found)
                return constructPath_Multi(prev, start, (fscore, gscore, neighbor, new_found))
            if neighbor_key not in g.keys():
                g[neighbor_key] = gscore
                h[neighbor_key] = hscore
                prev[(fscore, gscore, neighbor, new_found)] = (curr[0], curr[1], curr[2], curr_found)
                heapq.heappush(frontier, (fscore, gscore, neighbor, new_found))
    return []    

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

def constructPath_Multi(prev, start, end):
    path = []
    curr = end
    while 1:
        path.append(curr[2])
        if curr == start:
            path.reverse()
            return path
        curr = prev[curr]
    return []

def constructPath_Fast(prev, start, end):
    path = []
    curr = end
    while 1:
        path.append(curr)
        if curr == start:
            path.reverse()
            return path
        curr = prev[curr]
    return []

def astar_helper(maze, start, end):
    g = {}
    g[start] = 0
    
    f = {}
    f[start] = g[start] + manhattan(start, end)
    
    frontier = []
    heapq.heappush(frontier, (f[start], manhattan(start, end), start))
    
    prev = {}
    prev[start] = None
    while len(frontier) != 0:
        curr = heapq.heappop(frontier)
        if curr[2] == end:
            return len(constructPath(prev, start, end))
        for neighbor in maze.getNeighbors(curr[2][0], curr[2][1]):
            if neighbor not in g.keys():
                g[neighbor] = float('inf')
            g_score = g[curr[2]] + 1
            if g_score < g[neighbor]:
                g[neighbor] = g_score
                f[neighbor] = g[neighbor] + manhattan(neighbor, end)
                prev[neighbor] = curr[2]
                heapq.heappush(frontier, (f[neighbor], manhattan(neighbor, end), neighbor))
    return 0    

def getMST_Fast(maze, astar_dict, curr, remaining):
    U = remaining[:]
    V = []
    (dot, cost) = nearestDot(maze, astar_dict, curr, remaining)
    frontier = []
    heapq.heappush(frontier, (0, dot))
    S = cost
    
    while len(frontier) != 0:
        curr = heapq.heappop(frontier)
        if curr[1] not in V:
            S += curr[0]
            if len(U) == 1:
                break
            U.remove(curr[1])
            V.append(curr[1])
            for u in U:
                if (curr[1], u) not in astar_dict.keys() or (u, curr[1]) not in astar_dict.keys():
                    astar_dict[(curr[1], u)] = astar_helper(maze, curr[1], u)
                    astar_dict[(u, curr[1])] = astar_dict[(curr[1], u)]
                cost = astar_dict[(curr[1], u)]
                heapq.heappush(frontier, (cost, u))
    return S
    
def getMST_Multi(maze, astar_dict, curr, remaining):
    U = remaining[:]
    V = [curr]    
    if V == U:
        return 0
    
    V = []
    closest = nearestDot(maze, astar_dict, curr, U)[0]
    cost = nearestDot(maze, astar_dict, curr, U)[1]
    V = [closest]
    flag = 0

    S = cost - 1
    
    if V == U:
        return 0
    
    while 0 < len(U):
        minCost = float('inf')
        for v in V:
            for u in U:
                if u != v:
                    if (v,u) not in astar_dict or (u,v) not in astar_dict:
                        val = astar_helper(maze, v, u)
                        astar_dict[(v,u)] = val
                        astar_dict[(u,v)] = val
                    currCost = astar_dict[(u,v)]
                    if currCost < minCost:
                        flag = 1
                        minCost = currCost
                        minNeighbor = u
        if flag:
            V.append(minNeighbor)
            U.remove(minNeighbor)
            flag = 0
            S = S + minCost - 1
    return S  


def getMST(curr, remaining):
    U = remaining[:]
    V = [curr]    
    if V == U:
        return 0
    
    V = []
    closest = nearestDot1(curr, U)[0]
    cost = nearestDot1(curr, U)[1]
    V = [closest]
    flag = 0
    
    S = 0
    S += cost
    if V == U:
        return 0
    while len(V) <= len(U):
        minCost = float('inf')
        for v in V:
            for u in U:
                if u != v:
                    currCost = manhattan(v, u)
                    if currCost < minCost:
                        flag = 1
                        minCost = currCost
                        minNeighbor = u
        if flag:
            V.append(minNeighbor)
            flag = 0
            S += minCost
    return S

def nearestDot1(curr, remaining):
    minVal = float('inf')
    minDot = tuple()
    for dot in remaining:
        if manhattan(curr, dot) < minVal:
            minVal = manhattan(curr, dot)
            minDot = dot
    return (minDot, minVal)

def nearestDot(maze, astar_dict, curr, remaining):
    minVal = float('inf')
    minDot = tuple()
    for dot in remaining:
        if (curr, dot) not in astar_dict.keys():
            currVal = astar_helper(maze, curr, dot)
            astar_dict[(curr, dot)] = currVal
            astar_dict[(dot, curr)] = currVal 
        currVal = astar_dict[(curr, dot)]
        if currVal < minVal:
            minVal = currVal
            minDot = dot
    return (minDot, minVal)

def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])