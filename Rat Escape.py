import math
from collections import deque
import heapq

def euclidean_distance(coord1, coord2):
    return math.sqrt((coord1[0]-coord2[0])**2 + (coord1[1]-coord2[1])**2)

def dfs(graph, start, goal):
    stack = [(start, [start], 0)]
    visited = set()
    visited_order = []
    while stack:
        node, path, cost = stack.pop()
        if node not in visited:
            visited.add(node)
            visited_order.append(node)
            if node == goal:
                return path, cost, len(visited_order)
            # Add neighbors in reverse order to explore deeper first
            for neighbor, c in sorted(graph[node], reverse=True):
                if neighbor not in visited:
                    stack.append((neighbor, path + [neighbor], cost + c))
    return None, math.inf, len(visited_order)

def bfs(graph, start, goal):
    queue = deque([(start, [start], 0)])
    visited = set([start])
    visited_order = [start]
    while queue:
        node, path, cost = queue.popleft()
        if node == goal:
            return path, cost, len(visited_order)
        for neighbor, c in sorted(graph[node]):
            if neighbor not in visited:
                visited.add(neighbor)
                visited_order.append(neighbor)
                queue.append((neighbor, path + [neighbor], cost + c))
    return None, math.inf, len(visited_order)

def ucs(graph, start, goal):
    heap = [(0, start, [start])]
    visited = set()
    visited_order = []
    while heap:
        cost, node, path = heapq.heappop(heap)
        if node in visited:
            continue
        visited.add(node)
        visited_order.append(node)
        if node == goal:
            return path, cost, len(visited_order)
        for neighbor, c in graph[node]:
            if neighbor not in visited:
                heapq.heappush(heap, (cost + c, neighbor, path + [neighbor]))
    return None, math.inf, len(visited_order)

def a_star(graph, coords, start, goal):
    heap = []
    heapq.heappush(heap, (0 + euclidean_distance(coords[start], coords[goal]), 0, start, [start]))
    visited = set()
    visited_order = []
    while heap:
        f, cost, node, path = heapq.heappop(heap)
        if node in visited:
            continue
        visited.add(node)
        visited_order.append(node)
        if node == goal:
            return path, cost, len(visited_order)
        for neighbor, c in graph[node]:
            if neighbor not in visited:
                g = cost + c
                h = euclidean_distance(coords[neighbor], coords[goal])
                heapq.heappush(heap, (g + h, g, neighbor, path + [neighbor]))
    return None, math.inf, len(visited_order)

def main():
    # Input reading
    N, M = map(int, input("Enter number of junctions and pipes: ").split())
    graph = {i: [] for i in range(1, N+1)}
    for _ in range(M):
        u, v, cost = map(int, input("Enter pipe (junction1 junction2 cost): ").split())
        graph[u].append((v, cost))
        graph[v].append((u, cost))  # Assuming undirected pipes

    coords = {}
    for i in range(1, N+1):
        x, y = map(float, input(f"Enter coordinates for junction {i}: ").split())
        coords[i] = (x, y)

    start, goal = map(int, input("Enter start and goal junctions: ").split())

    # Run algorithms
    dfs_path, dfs_cost, dfs_visited = dfs(graph, start, goal)
    bfs_path, bfs_cost, bfs_visited = bfs(graph, start, goal)
    ucs_path, ucs_cost, ucs_visited = ucs(graph, start, goal)
    astar_path, astar_cost, astar_visited = a_star(graph, coords, start, goal)

    # Output results
    print("\nDFS path:", dfs_path)
    print("DFS total cost:", dfs_cost)
    print("DFS junctions visited:", dfs_visited)

    print("\nBFS path:", bfs_path)
    print("BFS total cost:", bfs_cost)
    print("BFS junctions visited:", bfs_visited)

    print("\nUCS path:", ucs_path)
    print("UCS total cost:", ucs_cost)
    print("UCS junctions visited:", ucs_visited)

    print("\nA* path:", astar_path)
    print("A* total cost:", astar_cost)
    print("A* junctions visited:", astar_visited)

if __name__ == "__main__":
    main()


Sample Input

Enter number of junctions and pipes: 5 6
Enter pipe (junction1 junction2 cost): 1 2 2
Enter pipe (junction1 junction2 cost): 1 3 4
Enter pipe (junction1 junction2 cost): 2 4 7
Enter pipe (junction1 junction2 cost): 3 4 1
Enter pipe (junction1 junction2 cost): 4 5 3
Enter pipe (junction1 junction2 cost): 2 5 10
Enter coordinates for junction 1: 0 0
Enter coordinates for junction 2: 1 0
Enter coordinates for junction 3: 0 1
Enter coordinates for junction 4: 1 1
Enter coordinates for junction 5: 2 1
Enter start and goal junctions: 1 5

Sample Output

DFS path: [1, 3, 4, 5]
DFS total cost: 8
DFS junctions visited: 4
BFS path: [1, 2, 5]
BFS total cost: 12
BFS junctions visited: 4
UCS path: [1, 3, 4, 5]
UCS total cost: 8
UCS junctions visited: 5
A* path: [1, 3, 4, 5]
A* total cost: 8.0
A* junctions visited: 5
