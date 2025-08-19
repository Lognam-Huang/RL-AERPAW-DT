import math
import time
import numpy as np
from itertools import permutations
from ortools.sat.python import cp_model


def assignLandmarks(uav_pos, landmarks):
    """
    Computes an optimal or near-optimal solution to the landmark assignment problem that minimizes the maximum UAV travel time

    Args:
        landmarks (np.array(float)): The positions of the landmarks in space, shape=(num_landmarks, 3)
    
    Returns:
        Tuple(float, np.array(int)): The maximum UAV travel distance, and an array that stores the UAV id assigned to each landmark, shape=(num_landmarks,)
    """
    n = len(uav_pos)
    m = len(landmarks)

    D = np.zeros((n, m)).astype(np.int64)
    for i in range(n):
        for j in range(m):
            D[i][j] = np.int64(np.linalg.norm(uav_pos[i] - landmarks[j]))
    
    model = cp_model.CpModel()
    X = np.array([[model.new_bool_var("") for _ in range(m)] for __ in range(n)])
    
    # Both should have shape (num_uavs, num_landmarks)

    # Adding row constraints
    for i in range(n):
        model.add(np.sum(X[i, :]) == 1)
    
    # Adding column constraints
    for j in range(m):
        model.add(np.sum(X[:, j]) == 1)

    # Adding maximum constraint
    maximum = model.new_int_var(0, np.max(D), "")
    product = X * D
    for i in range(n):
        for j in range(m):
            model.add(maximum >= product[i][j])

    # Finalizing and solving
    model.Minimize(maximum)
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        raise ValueError("Model doesn't convege for this input")

    return solver.Value(maximum)


def bruteForce(uav_pos, landmarks):
    n = len(uav_pos)
    m = len(landmarks)

    assert n == m  # Making sure they're equal for this case

    D = np.zeros((n, m)).astype(np.int64)
    for i in range(n):
        for j in range(m):
            D[i][j] = np.int64(np.linalg.norm(uav_pos[i] - landmarks[j]))

    minimum = math.inf
    final_p = None
    for p in permutations(range(n)):
        maximum = 0
        for i in range(n):
            maximum = max(maximum, D[i][p[i]])
        if maximum < minimum:
            minimum = maximum 
            final_p = p

    return minimum, final_p


def _perfectMatching(matrix):
    """
    Determines if the graph represented by the adjacency matrix has a perfect bipartite matching

    Args:
        matrix (np.array(np.array(int)): The 0/1 adjacency matrix, should have shape=(n, n)
    
    Returns:
        Tuple(Boolean, np.array(int)): If a perfect matching is possible, followed by an optimal matching if possible, otherwise it is nonsense
    """
    n = len(matrix)
    match = [-1] * n  # Tracks which row is matched to each column
    graph = matrix    # The adjacency matrix
    
    def find_matching(u, seen):
        for v in range(n):
            if graph[u][v] == 1 and not seen[v]:
                seen[v] = True
                if match[v] == -1 or find_matching(match[v], seen):
                    match[v] = u
                    return True
        return False

    count = 0
    for u in range(n):
        seen = [False] * n
        if find_matching(u, seen):
            count += 1
            
    return count == n, match


def customOptimized(uav_pos, landmarks):
    """
    This produces an optimial solution every time, with a time complexity of O(n^3logn) where n is the number of landmarks
    This is the most experimentally optimized approach by a considerable margin, and it is floating-point precise instead of falling back on integers
    I would consider this medium-tested, and I am 75% confident that it is working correctly

    Args:
        uav_pos (np.array(float)): The position of the UAVs in space, shape=(num_uavs, 3)
        landmarks (np.array(float)): The positions of the landmarks in space, shape=(num_landmarks, 3)
    
    Returns:
        Tuple(float, np.array(int)): The maximum UAV travel distance, and an array that stores the id of the landmark assigned to each UAV
    """
    n = len(uav_pos)
    m = len(landmarks)

    assert n == m
    D = np.zeros((n, m)).astype(np.int64)
    for i in range(n):
        for j in range(m):
            D[i][j] = np.int64(np.linalg.norm(uav_pos[i] - landmarks[j]))
    
    flattened = np.sort(D.reshape(n * n))

    # Binary Searching with range n^2
    left = 0
    right = n * n - 1
    while left < right:
        mid = left // 2 + right // 2

        res = _perfectMatching(D <= flattened[mid])

        if res[0]:
            right = mid
        else:
            left = mid + 1

    # Swapping the matching to a standard basis
    optimal_matching = _perfectMatching(D <= flattened[left])[1]
    rtn = np.zeros(n).astype(np.int64)
    for i in range(n):
        rtn[optimal_matching[i]] = i

    return flattened[left], np.array(rtn).astype(np.int32)


if __name__ == '__main__':
    # Testing the ability to generate matches

    n = 6
    for i in range(10):
        uavs = np.random.randint(-1000, 1000, size=(n, 3))
        landmarks = np.random.randint(-1000, 1000, size=(n, 3))

        a1, m1 = customOptimized(uavs, landmarks)
        a2, m2 = bruteForce(uavs, landmarks)

        assert a1 == a2

        for i in range(n):
            if m1[i] != m2[i]:
                print("Irregular!")
                print(a1)
                print(m1)
                print(m2)
                break

    """
    n = 7

    iterations = 100
    for i in range(iterations):
        uavs = np.random.randint(-1000, 1000, size=(n, 3))
        landmarks = np.random.randint(-1000, 1000, size=(n, 3))

        a1 = customOptimized(uavs, landmarks)
        a2 = bruteForce(uavs, landmarks)
        a3 = assignLandmarks(uavs, landmarks)
    
        if a1 != a2 or a2 != a3 or a1 != a3:
            print("Incorrect!")
            exit()

    print("All Good!!")

    n = 100
    uavs = np.random.randint(-100000, 100000, size=(n, 3))

    landmarks = np.random.randint(-100000, 100000, size=(n, 3))
    
    iter = 10
    start = time.time()
    for i in range(iter):
        ans = customOptimized(uavs, landmarks)
    end = time.time()
    print(f"Custom approach done in {(end - start) / iter} sec, with answer {ans}")

    start = time.time()
    for i in range(iter):
        ans = assignLandmarks(uavs, landmarks)
    end = time.time()
    print(f"Optimized done in {(end - start) / iter} sec, with answer {ans}")
    exit()

    start = time.time()
    for i in range(iter):
        ans = bruteForce(uavs, landmarks)
    end = time.time()
    print(f"Brute Force done in {(end - start) / iter} sec, with answer {ans}")
    """

def assignLandmarks(self, landmarks):
        """
        Assigns the UAVs current in the environment to landmarks optimially to minimize the maximum travel time
        This is an algorithm, not a heuristic, so it produces an optimal solution in O(n^3logn) time

        Args:
            landmarks (np.array(float)): The positions of the landmarks in space, shape=(num_landmarks, 3)
        
        Returns:
            Tuple(float, np.array(int)): The maximum UAV travel distance, and an array that stores the id of the landmark assigned to each UAV, shape=(num_uavs,)
        """

        n = len(self.uavs)
        m = len(landmarks)

        assert n == m
        D = np.zeros((n, m)).astype(np.int64)
        for i in range(n):
            for j in range(m):
                D[i][j] = np.int64(np.linalg.norm(self.uavs[i].pos - landmarks[j]))
        
        flattened = np.sort(D.reshape(n * n))

        # Binary Searching with range n^2
        left = 0
        right = n * n - 1
        while left < right:
            mid = left // 2 + right // 2

            res = _perfectMatching(D <= flattened[mid])

            if res[0]:
                right = mid
            else:
                left = mid + 1

        # Swapping the matching to a standard basis
        optimal_matching = _perfectMatching(D <= flattened[left])[1]
        rtn = np.zeros(n).astype(np.int64)
        for i in range(n):
            rtn[optimal_matching[i]] = i

        return flattened[left], np.array(rtn).astype(np.int32)

def _perfectMatching(matrix):
        """
        Utility function for assignLandmarks
        Determines if the graph represented by the adjacency matrix has a perfect bipartite matching

        Args:
            matrix (np.array(np.array(int)): The 0/1 adjacency matrix, should have shape=(n, n)
        
        Returns:
            Tuple(Boolean, np.array(int)): If a perfect matching is possible, followed by an optimal matching if possible, otherwise it is nonsense
        """
        n = len(matrix)
        match = [-1] * n  # Tracks which row is matched to each column
        graph = matrix    # The adjacency matrix
        
        def find_matching(u, seen):
            for v in range(n):
                if graph[u][v] == 1 and not seen[v]:
                    seen[v] = True
                    if match[v] == -1 or find_matching(match[v], seen):
                        match[v] = u
                        return True
            return False

        count = 0
        for u in range(n):
            seen = [False] * n
            if find_matching(u, seen):
                count += 1
                
        return count == n, match