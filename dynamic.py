import numpy as np

def tsp_dp(distances):
    """
    Solves the Traveling Salesman Problem using Dynamic Programming (Held-Karp algorithm)
    
    Args:
        distances: 2D list representing the distance matrix between cities
        
    Returns:
        tuple: (minimum_distance, optimal_path_indices)
    """
    
    n = len(distances)  # Number of cities
    
    # DP table initialization:
    # dp[mask][u] = minimum cost to visit all cities in 'mask' ending at city 'u'
    # mask is a bitmask representing visited cities (e.g., 0b101 means cities 0 and 2 visited)
    dp = [[float('inf')] * n for _ in range(1 << n)]
    
    # Base case: starting at city 0 (C1) with only city 0 visited (mask = 0b0000001)
    dp[1][0] = 0  
    
    # Parent table to reconstruct the optimal path
    # parent[mask][u] = city visited before u in this optimal path
    parent = [[-1] * n for _ in range(1 << n)]
    
    # Main DP computation:
    # Iterate through all possible subsets of cities (represented by bitmask)
    for mask in range(1 << n):
        
        # For each city in the current subset
        for u in range(n):
            if dp[mask][u] == float('inf'):
                continue  # Skip if this state hasn't been reached yet
                
            # Try extending the path to all unvisited cities
            for v in range(n):
                if not (mask & (1 << v)):  # If city v is not in the subset
                    new_mask = mask | (1 << v)  # Create new subset including v
                    
                    # Update DP table if we found a cheaper path to v
                    if dp[new_mask][v] > dp[mask][u] + distances[u][v]:
                        dp[new_mask][v] = dp[mask][u] + distances[u][v]
                        parent[new_mask][v] = u  # Record predecessor
    
    # Find the optimal return path to the starting city (C1)
    final_mask = (1 << n) - 1  # Bitmask with all cities visited (e.g., 0b1111111 for n=7)
    min_cost = float('inf')
    last_city = -1
    
    # Check all possible final cities before returning to C1
    for u in range(1, n):
        total_cost = dp[final_mask][u] + distances[u][0]
        if total_cost < min_cost:
            min_cost = total_cost
            last_city = u
    
    # Reconstruct the optimal path by backtracking through parent pointers
    path = []
    mask = final_mask
    current = last_city
    
    # Backtrack from last city to first
    while current != -1:
        path.append(current)
        new_mask = mask ^ (1 << current)  # Remove current city from mask
        current = parent[mask][current]   # Move to predecessor
        mask = new_mask
    
    path.append(0)  # Complete the cycle by returning to start (C1)
    path.reverse()  # Put in proper order (start to end to start)
    
    return min_cost, path

# Distance matrix for the given problem (∞ represents unavailable direct paths)
INF = float('inf')
distances = [
    # C1   C2   C3   C4   C5   C6   C7
    [  0,  12,  10, INF, INF, INF,  12],  # C1
    [ 12,   0,   8,  12, INF, INF, INF],  # C2
    [ 10,   8,   0,  11,   3, INF,   9],  # C3
    [INF,  12,  11,   0,  11,  10, INF],  # C4
    [INF, INF,   3,  11,   0,   6,   7],  # C5
    [INF, INF, INF,  10,   6,   0,   9],  # C6
    [ 12, INF,   9, INF,   7,   9,   0]   # C7
]

# Solve the TSP
min_distance, path_indices = tsp_dp(distances)

# Convert indices to city names for readable output
city_names = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
optimal_path = [city_names[i] for i in path_indices]

# Print results
print(f"Optimal Path: {' → '.join(optimal_path)}")
print(f"Total Distance: {min_distance}")

# Small test case verification
test_matrix = [
    [0, 10, 15],
    [10, 0, 12],
    [15, 12, 0]
]
test_dist, test_path = tsp_dp(test_matrix)
print(f"\nTest Case: {test_path} with distance {test_dist}")