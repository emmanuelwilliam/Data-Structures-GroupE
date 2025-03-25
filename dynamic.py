"""
    The dynamic programming approach is the most suitable
    appproach for a range of less than 20 cities as its exact and able to traverse 
    through all the cities with a time complexity of O(2^n x n^2)
"""

import numpy as np

def held_karp_tsp(dist_matrix):
    n = len(dist_matrix)
    
    # This represents the number of cities
    # DP table initialization
    # dp[mask][i] = min cost to reach city i visiting all cities in 'mask'

    dp =[[float('inf')]*n for _ in range(1<<n)]
    dp[1][0] = 0 #Base case: starting at city 0 (mask 0000001)

    #Parent table to reconstruct the path
    parent = [[-1]*n for _ in range(1<<n)]

    """
    -mask loops over all subset of cities
    -i represents the last visited city
    -mask and (1<<i) checks if city i is visited
    -for every visited city i, we check for all 
    possible cities j from which i could have been reached
    -We update the dp table and track the parent city
    """
    #Populate DP table
    #This computes the minimum cost for using DP
    for mask in range (1, 1<<n): #All subsets
        for i in range(n): #All possible ending cities
            if not (mask & (1<<i)):
                #Checks if city i is in subset
                continue


            for j in range(n): #All possible previous cities
                if i==j or not (mask & (1<<j)):# j must be in subset
                    continue

                #Update DP if path through j is better
                new_cost = dp[mask ^ (1<<i)][j] + dist_matrix[j][i]
                if new_cost < dp[mask][i]:
                    dp[mask][i] = new_cost
                    parent[mask][i] = j

    #Find optimal return path to city 0
    final_mask = (1<<n)-1 #All cities visited (mask 1111111)
    min_cost = float('inf')
    last_city = -1

    for i in range (1,n): # Try all possible last cities before returning
        current_cost = dp[final_mask][i] + dist_matrix[i][0]
        if current_cost < min_cost:
            min_cost = current_cost
            last_city = i 

    #Reconstruct path
    path = [0] #Start with city 0
    mask = final_mask
    current_city = last_city

    while len(path) < n:#Backtrack using parent table
        path.append(current_city)
        new_mask = mask ^ (1 << current_city)
        current_city = parent[mask | (1 << current_city)][current_city]
        mask = new_mask

    path.append(0)  # Complete the cycle
    path.reverse()  # Start -> ... -> End -> Start
    return path, min_cost

# Example distance matrix (must be symmetric)
dist_matrix = [
    [0,10,15,20,25,30,35],
    [10,0,12,18,22,28,32],
    [15,12,0,10,15,20,25],
    [20,18,10,0,8,12,18],
    [25,22,15,8,0,6,10],
    [30,28,20,12,6,0,5],
    [35,32,25,18,10,5,0]
]


optimal_path,total_distance = held_karp_tsp(dist_matrix)

print("Optimal path:",optimal_path) # [0,1,2,3,4,5,6,0]
print("Total distance:",total_distance) # 78


# Small Test
small_matrix = [
    [0,10,15],
    [10,0,12],
    [15,12,0]
]
print(held_karp_tsp(small_matrix)) # ([0,1,2,0], 37)