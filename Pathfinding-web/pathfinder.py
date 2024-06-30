import json
import os
import time
import numpy as np
import heapq
import tkinter as tk
from tkinter.filedialog import askopenfilename
from scipy.interpolate import griddata

tk.Tk().withdraw()
import warnings

warnings.filterwarnings("ignore")


class Node:
    def __init__(self, position, g=0, h=0):
        self.position = position
        self.g = g
        self.h = h
        self.f = g + h
        self.parent = None

    def __lt__(self, other):
        return self.f < other.f


class InteractiveBIMPathfinder:
    def __init__(self, grids, grid_size, floors, bbox):
        self.grids = grids
        self.bbox = bbox
        self.floors = floors
        self.grid_size = grid_size
        self.start = None
        self.goals = []
        self.grid_stairs = None
        self.current_floor = 0
        self.speed = 1  # Default speed

        self.path = None
        self.pathlength = None
        self.minimize_cost = True
        self.algorithm = 'A*'
        self.animated = True
        self.fps = 1
        self.heuristic_style = 'Min'
        self.heuristic_resolution = 10
        self.allow_diagonal = True
        self.wall_buffer = 0
        self.buffered_grids = None

    def load_grid_data(self, filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        grids = [np.array(grid) for grid in data['grids']]
        return grids, data['bbox'], data['floors'], data['grid_size']

    def set_algorithm(self, label):
        self.algorithm = label

    def set_minimize(self, label):
        self.minimize_cost = (label == 'Cost')

    def toggle_animation(self, label):
        self.animated = not self.animated

    def set_heuristic_style(self, label):
        self.heuristic_style = label.lower()

    def toggle_diagonal(self, label):
        self.allow_diagonal = not self.allow_diagonal

    def update_speed(self, val):
        self.speed = int(val)

    def update_fps(self, val):
        self.fps = float(val)

    def update_buffer(self, val):
        self.wall_buffer = val
        self.apply_wall_buffer()

    def apply_wall_buffer(self):
        self.buffered_grids = []
        for floor in self.grids:
            buffered_floor = floor.copy()
            wall_mask = (floor == 'wall')

            buffer_distance = int(self.wall_buffer / self.grid_size)

            for _ in range(buffer_distance):
                wall_mask = self.expand_mask(wall_mask)
            rows, cols = wall_mask.shape
            for i in range(rows):
                for j in range(cols):
                    if wall_mask[i, j] and floor[i,j] not in ['wall', 'door', 'stair']:
                        buffered_floor[i,j] = 'walla'
            self.buffered_grids.append(buffered_floor)
            #print(buffered_floor[0:10, 0:3])

    def expand_mask(self, mask):
        expanded = mask.copy()
        rows, cols = mask.shape
        for i in range(rows):
            for j in range(cols):
                if mask[i, j]:
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if 0 <= i + di < rows and 0 <= j + dj < cols:
                                expanded[i + di, j + dj] = True
        return expanded

    def grid_to_numeric(self, grid):
        element_types = ['empty', 'wall', 'door', 'stair', 'floor', 'walla']
        numeric_grid = np.zeros_like(grid, dtype=int)
        for i, element_type in enumerate(element_types):
            numeric_grid[grid == element_type] = i
        return numeric_grid

    def toggle_heuristic(self, event):
        self.show_heuristic = not self.show_heuristic

    def prev_floor(self, event):
        if self.current_floor > 0:
            self.current_floor -= 1

    def next_floor(self, event):
        if self.current_floor < len(self.grids) - 1:
            self.current_floor += 1

    def set_start_mode(self, event):
        self.mode = 'start'

    def set_goal_mode(self, event):
        self.mode = 'goal'

    def identify_exits(self, event):
        exits = set()
        for floor_index, floor in enumerate(self.grids):
            rows, cols = floor.shape
            for i in range(rows):
                for j in range(cols):
                    if floor[i, j] == 'door':
                        if self.is_exit(floor, i, j):
                            exits.add((i, j, floor_index))

        # Filter exits to keep only one per BIM door
        filtered_exits = self.filter_exits(exits)

        # Add filtered exits as goals
        for exit in filtered_exits:
            if exit not in self.goals:
                self.goals.append(exit)

    def is_exit(self, floor, x, y):
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        rows, cols = floor.shape

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            while 0 <= nx < rows and 0 <= ny < cols:
                if floor[nx, ny] in ['wall', 'door']:
                    break
                if nx == 0 or nx == rows - 1 or ny == 0 or ny == cols - 1:
                    return True
                nx, ny = nx + dx, ny + dy

        return False

    def filter_exits(self, exits):
        filtered = set()
        for exit in exits:
            if not any(self.are_connected_by_doors(exit, existing) for existing in filtered):
                filtered.add(exit)
        return filtered

    def are_connected_by_doors(self, pos1, pos2):
        if pos1[2] != pos2[2]:  # Different floors
            return False

        floor = self.grids[pos1[2]]
        visited = set()
        queue = [pos1[:2]]

        while queue:
            x, y = queue.pop(0)
            if (x, y) == pos2[:2]:
                return True

            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if (nx, ny) not in visited and floor[nx, ny] == 'door':
                    visited.add((nx, ny))
                    queue.append((nx, ny))

        return False

    def find_all_stairs(self):
        grid_stairs = []
        #print(self.grids)
        for z in range(len(self.grids)):
            for i in range(self.grids[z].shape[0]):
                for j in range(self.grids[z].shape[1]):
                    #print(str(z), " ", str(i), " ", str(j))
                    if self.grids[z][i, j] == 'stair':
                        grid_stairs.append([z, i, j])
        self.grid_stairs = grid_stairs

    # find nearest stairs leading to the goal floor
    def find_nearest_stairs(self, position, goal_position):
        if not self.grid_stairs:
            self.find_all_stairs()
        x, y, z = position
        #print("z: " + str(z))
        xg, yx, zg = goal_position
        min_distance = float('inf')
        nearest_stairs = None
        for stair in self.grid_stairs:
            z2 = stair[0]
            #print("z2: " + str(z2))
            i = stair[1]
            j = stair[2]
            if z == z2 and self.grids[zg][i, j] == 'stair' and self.grids[z][i, j] == 'stair':
                distance = np.sqrt((x - i) ** 2 + (y - j) ** 2)
                if distance < min_distance:
                    min_distance = distance
                    nearest_stairs = (i, j, z)
        return nearest_stairs

    def heuristic(self, a, position_b):
        if not self.goals:
            return 0  # Return 0 if there are no goals

        # Calculate heuristic for each goal
        goal_heuristics = []
        for b in self.goals:
            dx = abs(b[0] - a[0])
            dy = abs(b[1] - a[1])
            dz = abs(b[2] - a[2])

            if self.minimize_cost:
                h = np.sqrt(dx ** 2 + dy ** 2) * self.grid_size
                if dz > 0:
                    nearest_stairs = self.find_nearest_stairs(a, b)
                    if nearest_stairs:
                        h += (np.sqrt((nearest_stairs[0] - a[0]) ** 2 + (nearest_stairs[1] - a[1]) ** 2) +
                              np.sqrt(
                                  (b[0] - nearest_stairs[0]) ** 2 + (b[1] - nearest_stairs[1]) ** 2)) * self.grid_size
                        h += dz * 3 * self.grid_size  # Increased floor change penalty
                    else:
                        h = float('inf')
                if self.buffered_grids[a[2]][a[0], a[1]] == 'walla':
                    h += 10 * self.grid_size  # Add a cost for wall-adjacent cells
            else:
                # For distance minimization, use 3D Euclidean distance
                h = np.sqrt(dx ** 2 + dy ** 2 + (dz * 3) ** 2) * self.grid_size
                if self.buffered_grids[a[2]][a[0], a[1]] == 'walla':
                    h += 10 * self.grid_size  # Add a cost for wall-adjacent cells even in distance mode

            goal_heuristics.append(h)

        if self.heuristic_style == 'sum':
            return sum(goal_heuristics)
        else:
            return min(goal_heuristics)

    def calculate_sparse_heuristic(self):
        if self.goals is None:
            return None

        floor_shape = self.grids[self.current_floor].shape
        x = np.linspace(0, floor_shape[0] - 1, self.heuristic_resolution)
        y = np.linspace(0, floor_shape[1] - 1, self.heuristic_resolution)
        xx, yy = np.meshgrid(x, y)

        heuristic_values = []
        for i in range(self.heuristic_resolution):
            for j in range(self.heuristic_resolution):
                x_coord = int(xx[i, j])
                y_coord = int(yy[i, j])
                if self.grids[self.current_floor][x_coord, y_coord] != 'wall':
                    h_value = self.heuristic((x_coord, y_coord, self.current_floor), self.goals)
                    heuristic_values.append((x_coord, y_coord, h_value))

        if not heuristic_values:
            return None

        x_sparse, y_sparse, z_sparse = zip(*heuristic_values)

        grid_x, grid_y = np.mgrid[0:floor_shape[0], 0:floor_shape[1]]
        heuristic_map = griddata((x_sparse, y_sparse), z_sparse, (grid_x, grid_y), method='cubic')

        # Set walls to NaN
        for i in range(floor_shape[0]):
            for j in range(floor_shape[1]):
                if self.grids[self.current_floor][i, j] == 'wall':
                    heuristic_map[i, j] = np.nan

        return heuristic_map

    def get_neighbors(self, current):
        x, y, z = current.position
        neighbors = []
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        if self.allow_diagonal:
            directions += [(1, 1), (1, -1), (-1, 1), (-1, -1)]

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.buffered_grids[z].shape[0] and 0 <= ny < self.buffered_grids[z].shape[1]:
                if self.buffered_grids[z][nx, ny] not in ['wall', 'walla']:
                    neighbors.append(Node((nx, ny, z)))

        if self.buffered_grids[z][x, y] == 'stair':
            for nz in range(len(self.buffered_grids)):
                if nz != z and self.buffered_grids[nz][x, y] == 'stair':
                    neighbors.append(Node((x, y, nz)))

        return neighbors

    def get_cost(self, current, neighbor):
        dx = abs(neighbor.position[0] - current.position[0])
        dy = abs(neighbor.position[1] - current.position[1])
        dz = abs(neighbor.position[2] - current.position[2])

        if self.minimize_cost:
            cost = self.grid_size
            if self.buffered_grids[neighbor.position[2]][neighbor.position[0], neighbor.position[1]] == 'door':
                cost += 5 * self.grid_size
            elif self.buffered_grids[neighbor.position[2]][neighbor.position[0], neighbor.position[1]] == 'stair':
                cost += 1.25 * self.grid_size
            elif self.buffered_grids[neighbor.position[2]][neighbor.position[0], neighbor.position[1]] == 'walla':
                cost += 10 * self.grid_size  # Add a cost for wall-adjacent cells

            if self.allow_diagonal and dx + dy == 2:
                cost *= 1.414  # Diagonal movement cost
        else:
            # For distance minimization, use Euclidean distance
            cost = np.sqrt(dx ** 2 + dy ** 2) * self.grid_size
            if dz > 0:
                cost += 3 * self.grid_size  # Significant penalty for changing floors
            if self.buffered_grids[neighbor.position[2]][neighbor.position[0], neighbor.position[1]] == 'walla':
                cost += 10 * self.grid_size  # Add a cost for wall-adjacent cells even in distance mode

        return cost

    def run_algorithm(self, event):
        if not self.grid_stairs:
            self.find_all_stairs()
        if not self.start or not self.goals:
            print("Please set both start and at least one goal point.")
            return
        if self.algorithm == 'A*':
            self.run_astar()

    def run_astar(self):
        fps = self.fps
        time0 = time.time()
        self.path = None
        open_list = []
        closed_set = set()
        start_node = Node(self.start)
        start_node.h = self.heuristic(self.start, None)  # Calculate initial heuristic
        start_node.f = start_node.g + start_node.h

        heapq.heappush(open_list, (start_node.f, start_node))

        progress_counter = 0
        progress_threshold = 0  #max(1, int(100 / self.speed))

        while open_list:
            current_node = heapq.heappop(open_list)[1]

            if current_node.position in self.goals:
                self.visualize_progress(closed_set, [node for _, node in open_list], None)
                self.reconstruct_path(current_node)
                self.save_path_visualization()
                return

            closed_set.add(current_node.position)

            for neighbor in self.get_neighbors(current_node):
                if neighbor.position in closed_set:
                    continue

                tentative_g = current_node.g + self.get_cost(current_node, neighbor)

                if not any(node.position == neighbor.position for _, node in open_list):
                    neighbor.g = tentative_g
                    neighbor.h = self.heuristic(neighbor.position, None)
                    neighbor.f = neighbor.g + neighbor.h
                    neighbor.parent = current_node
                    heapq.heappush(open_list, (neighbor.f, neighbor))
                else:
                    for i, (f, node) in enumerate(open_list):
                        if node.position == neighbor.position and tentative_g < node.g:
                            node.g = tentative_g
                            node.f = node.g + node.h
                            node.parent = current_node
                            heapq.heapify(open_list)
                            break

            progress_counter += 1
            if progress_threshold == 0 or progress_counter >= progress_threshold:
                if time.time() - time0 > 1.0 / fps and self.animated:
                    time0 = time.time()
                    self.visualize_progress(closed_set, [node for _, node in open_list],
                                            self.get_current_path(current_node))
                    progress_counter = 0

        print("No path found to any goal.")

    def reconstruct_path(self, node):
        path = []
        path_length = 0
        while node:
            path.append(node.position)
            if node.parent:
                path_length += self.heuristic(node.position, node.parent.position)
            node = node.parent
        self.path = path[::-1]
        self.t_pathlength.set_val(f"{path_length:.2f}")
        self.pathlength = path_length
        self.visualize_path()

    def get_current_path(self, node):
        path = []
        while node:
            path.append(node.position)
            node = node.parent
        return path[::-1]


def main():
    fn = askopenfilename(filetypes=[("json files", "*.json")])
    pathfinder = InteractiveBIMPathfinder(fn)


if __name__ == "__main__":
    main()
