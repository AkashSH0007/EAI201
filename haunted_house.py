import heapq
import math

def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def euclidean(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def diagonal(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

def reconstruct_path(parent, start, goal):
    path = []
    cur = goal
    while cur in parent and cur != start:
        path.append(cur)
        cur = parent[cur]
    if cur == start:
        path.append(start)
        return path[::-1]
    else:
        print("No valid path found from start to goal.")
        return []

def greedy_best_first_search(grid, start, goal, heuristic):
    rows, cols = len(grid), len(grid[0])
    visited = set()
    pq = [(heuristic(start, goal), start)]
    parent = {start: None}
    nodes_explored = 0

    while pq:
        _, current = heapq.heappop(pq)
        nodes_explored += 1
        if current == goal:
            break
        visited.add(current)
        x, y = current
        for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]:
            nx, ny = x + dx, y + dy
            neighbor = (nx, ny)
            if (0 <= nx < rows and 0 <= ny < cols and
                grid[nx][ny] == 0 and neighbor not in visited and neighbor not in parent):
                parent[neighbor] = current
                heapq.heappush(pq, (heuristic(neighbor, goal), neighbor))
    path = reconstruct_path(parent, start, goal)
    return path, nodes_explored

def a_star_search(grid, start, goal, heuristic):
    rows, cols = len(grid), len(grid[0])
    open_set = [(heuristic(start, goal), 0, start)]
    visited = set()
    g_score = {start: 0}
    parent = {start: None}
    nodes_explored = 0

    while open_set:
        _, cost, current = heapq.heappop(open_set)
        nodes_explored += 1
        if current == goal:
            break
        visited.add(current)
        x, y = current
        for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]:
            nx, ny = x + dx, y + dy
            neighbor = (nx, ny)
            if (0 <= nx < rows and 0 <= ny < cols and
                grid[nx][ny] == 0 and neighbor not in visited):
                tentative_g = g_score[current] + 1
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    priority = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (priority, tentative_g, neighbor))
                    parent[neighbor] = current
    path = reconstruct_path(parent, start, goal)
    return path, nodes_explored

def print_grid_with_path(grid, path):
    display = []
    for i, row in enumerate(grid):
        display_row = []
        for j, cell in enumerate(row):
            if (i, j) in path:
                display_row.append('*')
            elif cell == 1:
                display_row.append('#')
            else:
                display_row.append('.')
        display.append(' '.join(display_row))
    print('\n'.join(display))

grid = [
    [0, 0, 0, 1, 0],
    [1, 1, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0]
]
start = (0, 0)
goal = (4, 4)

print("Running Greedy Best-First Search (Manhattan)")
gbfs_path, gbfs_nodes = greedy_best_first_search(grid, start, goal, manhattan)
print("GBFS Path:", gbfs_path)
print("GBFS Path Length:", len(gbfs_path))
print("GBFS Nodes Explored:", gbfs_nodes)
print_grid_with_path(grid, gbfs_path)

print("\nRunning A* Search (Euclidean)")
astar_path, astar_nodes = a_star_search(grid, start, goal, euclidean)
print("A* Path:", astar_path)
print("A* Path Length:", len(astar_path))
print("A* Nodes Explored:", astar_nodes)
print_grid_with_path(grid, astar_path)

print("\nRunning A* Search (Diagonal)")
astar_diag_path, astar_diag_nodes = a_star_search(grid, start, goal, diagonal)
print("A* Diagonal Path:", astar_diag_path)
print("A* Diagonal Path Length:", len(astar_diag_path))
print("A* Diagonal Nodes Explored:", astar_diag_nodes)
print_grid_with_path(grid, astar_diag_path)



"""
sample output:-
Running Greedy Best-First Search (Manhattan)
GBFS Path: [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (3, 2), (4, 2), (4, 3), (4, 4)]
GBFS Path Length: 9
GBFS Nodes Explored: 9
* * * # .
# # * # .
. . * # .
# # * # .
. . * * *
"""
