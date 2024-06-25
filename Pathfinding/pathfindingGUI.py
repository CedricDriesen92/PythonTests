import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import heapq
from matplotlib.widgets import Button, Slider, TextBox, RadioButtons
import tkinter as tk
from tkinter.filedialog import askopenfilename
from scipy.interpolate import griddata

tk.Tk().withdraw()  # part of the import if you are not using other tkinter functions
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
import line_profiler_pycharm
from line_profiler_pycharm import profile
from timeit import default_timer as timer


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
    def __init__(self, filename):
        self.grids, self.bbox, self.floors, self.grid_size = self.load_grid_data(filename)
        self.start = None
        self.goal = None
        self.current_floor = 0
        self.speed = 1  # Default speed
        self.fig, self.ax = plt.subplots(figsize=(12, 9))
        self.canvas = self.fig.canvas
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.fig.canvas.get_tk_widget().master)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.path = None
        self.minimize_cost = True
        self.algorithm = 'A*'
        self.animated = True
        self.fps = 1
        self.show_heuristic = False
        self.heuristic_resolution = 20
        self.setup_plot()

    def load_grid_data(self, filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        grids = [np.array(grid) for grid in data['grids']]
        return grids, data['bbox'], data['floors'], data['grid_size']

    def setup_plot(self):
        plt.subplots_adjust(bottom=0.3)
        self.ax_prev = plt.axes([0.2, 0.15, 0.1, 0.075])
        self.ax_next = plt.axes([0.31, 0.15, 0.1, 0.075])
        self.ax_start = plt.axes([0.42, 0.15, 0.1, 0.075])
        self.ax_goal = plt.axes([0.53, 0.15, 0.1, 0.075])
        self.ax_run = plt.axes([0.64, 0.15, 0.1, 0.075])
        self.ax_speed = plt.axes([0.2, 0.05, 0.6, 0.03])
        self.ax_pathlength = plt.axes([0.9, 0.15, 0.075, 0.03])
        self.ax.set_xlim(0, self.grids[0].shape[1])
        self.ax.set_ylim(0, self.grids[0].shape[0])
        self.ax_minimize = plt.axes([0.75, 0.25, 0.2, 0.1])
        self.ax_animate = plt.axes([0.75, 0.7, 0.2, 0.1])
        self.ax_fps = plt.axes([0.75, 0.65, 0.2, 0.03])

        self.b_prev = Button(self.ax_prev, 'Previous')
        self.b_next = Button(self.ax_next, 'Next')
        self.b_start = Button(self.ax_start, 'Set Start')
        self.b_goal = Button(self.ax_goal, 'Set Goal')
        self.b_run = Button(self.ax_run, 'Run A*')
        self.s_speed = Slider(self.ax_speed, 'Delay', 1, 100, valinit=self.speed, valstep=0.1)
        self.t_pathlength = TextBox(self.ax_pathlength, 'Path Length:', initial='0')
        self.radio_minimize = RadioButtons(self.ax_minimize, ('Cost', 'Distance'))
        self.radio_animate = RadioButtons(self.ax_animate, ('Yes', 'No'))
        self.s_fps = Slider(self.ax_fps, 'fps', 0.1, 120, valinit=self.speed, valstep=0.1)

        self.b_prev.on_clicked(self.prev_floor)
        self.b_next.on_clicked(self.next_floor)
        self.b_start.on_clicked(self.set_start_mode)
        self.b_goal.on_clicked(self.set_goal_mode)
        self.b_run.on_clicked(self.run_astar)
        self.s_speed.on_changed(self.update_speed)
        self.radio_minimize.on_clicked(self.set_minimize)
        self.radio_animate.on_clicked(self.set_animate)
        self.s_fps.on_changed(self.update_fps)

        self.ax_heuristic = plt.axes([0.75, 0.35, 0.2, 0.05])
        self.b_heuristic = Button(self.ax_heuristic, 'Toggle Heuristic')
        self.b_heuristic.on_clicked(self.toggle_heuristic)

        self.mode = None
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

        self.update_plot()

    def set_algorithm(self, label):
        self.algorithm = label

    def set_minimize(self, label):
        self.minimize_cost = (label == 'Cost')

    def set_animate(self, label):
        self.animated = (label == 'Yes')

    def update_speed(self, val):
        self.speed = int(val)

    def update_fps(self, val):
        self.fps = float(val)

    def grid_to_numeric(self, grid):
        element_types = ['empty', 'wall', 'door', 'stair', 'floor']
        numeric_grid = np.zeros_like(grid, dtype=int)
        for i, element_type in enumerate(element_types):
            numeric_grid[grid == element_type] = i
        return numeric_grid

    def toggle_heuristic(self, event):
        self.show_heuristic = not self.show_heuristic
        self.update_plot()

    def calculate_floor_heuristic(self):
        if self.goal is None:
            return None

        heuristic_map = np.zeros(self.grids[self.current_floor].shape)
        for x in range(self.grids[self.current_floor].shape[0]):
            for y in range(self.grids[self.current_floor].shape[1]):
                if self.grids[self.current_floor][x, y] != 'wall':
                    heuristic_map[x, y] = self.heuristic((x, y, self.current_floor), self.goal)
                else:
                    heuristic_map[x, y] = np.nan  # Use NaN for walls to exclude them from the heatmap
        return heuristic_map

    def calculate_sparse_heuristic(self):
        if self.goal is None:
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
                    h_value = self.heuristic((x_coord, y_coord, self.current_floor), self.goal)
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

    def update_plot(self):
        self.ax.clear()
        colors = ['white', 'black', 'orange', 'red', 'lavenderblush']
        color_map = ListedColormap(colors)

        grid = self.grid_to_numeric(self.grids[self.current_floor])
        self.ax.imshow(grid.T, cmap=color_map, interpolation='nearest')

        if self.show_heuristic and self.goal:
            heuristic_map = self.calculate_sparse_heuristic()
            if heuristic_map is not None:
                heuristic_cmap = LinearSegmentedColormap.from_list("", ["blue", "green", "yellow", "red"])
                self.ax.imshow(heuristic_map.T, cmap=heuristic_cmap, alpha=0.5)
                #plt.colorbar(heatmap, ax=self.ax)
                #self.ax.axis('off')

        if self.start and self.start[2] == self.current_floor:
            self.ax.plot(self.start[0], self.start[1], 'go', markersize=10)
        if self.goal and self.goal[2] == self.current_floor:
            self.ax.plot(self.goal[0], self.goal[1], 'ro', markersize=10)

        if self.path:
            self.visualize_path()

        self.ax.set_xlim(self.ax.get_xlim())
        self.ax.set_ylim(self.ax.get_ylim())
        self.ax.set_title(f'Floor {self.current_floor + 1}')
        self.ax.axis('off')
        #self.ax.get_legend().remove()
        plt.draw()

    def prev_floor(self, event):
        if self.current_floor > 0:
            self.current_floor -= 1
            self.update_plot()

    def next_floor(self, event):
        if self.current_floor < len(self.grids) - 1:
            self.current_floor += 1
            self.update_plot()

    def set_start_mode(self, event):
        self.mode = 'start'

    def set_goal_mode(self, event):
        self.mode = 'goal'

    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        x, y = int(event.xdata), int(event.ydata)
        if self.mode == 'start':
            self.start = (x, y, self.current_floor)
            print(f"Start set to: {self.start}")
        elif self.mode == 'goal':
            self.goal = (x, y, self.current_floor)
            print(f"Goal set to: {self.goal}")
        self.mode = None
        self.update_plot()

    def find_nearest_stairs(self, position):
        x, y, z = position
        min_distance = float('inf')
        nearest_stairs = None
        for i in range(self.grids[z].shape[0]):
            for j in range(self.grids[z].shape[1]):
                if self.grids[z][i, j] == 'stair':
                    distance = np.sqrt((x - i) ** 2 + (y - j) ** 2)
                    if distance < min_distance:
                        min_distance = distance
                        nearest_stairs = (i, j, z)
        return nearest_stairs

    def heuristic(self, a, b):
        if a[2] == b[2]:  # Same floor
            return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) * self.grid_size
        else:  # Different floors
            nearest_stairs = self.find_nearest_stairs(a)
            if nearest_stairs:
                # Distance to nearest stairs + estimated distance from stairs to goal
                return (np.sqrt((nearest_stairs[0] - a[0]) ** 2 + (nearest_stairs[1] - a[1]) ** 2) +
                        np.sqrt((b[0] - nearest_stairs[0]) ** 2 + (b[1] - nearest_stairs[1]) ** 2) +
                        abs(b[2] - a[2]) * 5) * self.grid_size  # Add floor change penalty
            else:
                # If no stairs found, use a large heuristic to avoid this path
                return float('inf')

    def get_neighbors(self, current):
        x, y, z = current.position
        neighbors = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.grids[z].shape[0] and 0 <= ny < self.grids[z].shape[1]:
                if self.grids[z][nx, ny] != 'wall':
                    neighbors.append(Node((nx, ny, z)))

        if self.grids[z][x, y] == 'stair':
            for nz in range(len(self.grids)):
                if nz != z and self.grids[nz][x, y] == 'stair':
                    neighbors.append(Node((x, y, nz)))

        return neighbors

    def get_cost(self, current, neighbor):
        if self.minimize_cost:
            cost = self.grid_size
            if self.grids[neighbor.position[2]][neighbor.position[0], neighbor.position[1]] == 'door':
                cost += 5 * self.grid_size
            elif self.grids[neighbor.position[2]][neighbor.position[0], neighbor.position[1]] == 'stair':
                cost += 1.0 * self.grid_size
            return cost
        else:
            return self.heuristic(current.position, neighbor.position)

    def run_algorithm(self, event):
        if not self.start or not self.goal:
            print("Please set both start and goal points.")
            return
        if self.algorithm == 'A*':
            self.run_astar()
        elif self.algorithm == 'Dijkstra':
            self.run_dijkstra()
        elif self.algorithm == 'BFS':
            self.run_bfs()

    def run_astar(self, event):
        fps = 1
        time0 = timer()
        self.path = None
        open_list = []
        closed_set = set()
        start_node = Node(self.start)
        goal_node = Node(self.goal)

        heapq.heappush(open_list, (start_node.f, start_node))

        progress_counter = 0
        progress_threshold = max(1, int(100 / self.speed))

        while open_list:
            current_node = heapq.heappop(open_list)[1]

            if current_node.position == goal_node.position:
                self.reconstruct_path(current_node)
                return

            closed_set.add(current_node.position)

            for neighbor in self.get_neighbors(current_node):
                if neighbor.position in closed_set:
                    continue

                tentative_g = current_node.g + self.get_cost(current_node, neighbor)

                if not any(node.position == neighbor.position for _, node in open_list):
                    neighbor.g = tentative_g
                    neighbor.h = self.heuristic(neighbor.position, goal_node.position)
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
                if timer() - time0 > 1.0 / fps:
                    time0 = timer()
                    self.visualize_progress(closed_set, [node for _, node in open_list], self.get_current_path(current_node))
                    progress_counter = 0

        print("No path found.")

    @profile
    def run_astar_old(self, event):
        if not self.start or not self.goal:
            print("Please set both start and goal points.")
            return
        fps = self.fps
        time0 = timer()
        self.path = None  # Clear the previous path
        open_list = []
        closed_set = set()
        start_node = Node(self.start)
        goal_node = Node(self.goal)

        heapq.heappush(open_list, (start_node.f, start_node))

        progress_counter = 0

        progress_threshold = max(1, int(100 / self.speed))

        while open_list:
            current_node = heapq.heappop(open_list)[1]
            if current_node.position == goal_node.position:
                path = []
                path_length = 0
                while current_node:
                    path.append(current_node.position)
                    if current_node.parent:
                        path_length += self.heuristic(current_node.position, current_node.parent.position)
                    current_node = current_node.parent
                self.path = path[::-1]

                self.t_pathlength.set_val(f"{path_length:.2f}")
                self.visualize_path()
                return

            closed_set.add(current_node.position)

            for neighbor in self.get_neighbors(current_node):
                if neighbor.position in closed_set:
                    continue

                neighbor.g = current_node.g + self.grid_size
                if self.grids[neighbor.position[2]][neighbor.position[0], neighbor.position[1]] == 'door':
                    if self.minimize_cost:
                        neighbor.g += 5 * self.grid_size  # Higher cost for doors

                elif self.grids[neighbor.position[2]][neighbor.position[0], neighbor.position[1]] == 'stair':
                    if self.minimize_cost:
                        neighbor.g += 1.0 * self.grid_size  # Moderate cost for stairs

                neighbor.h = self.heuristic(neighbor.position, goal_node.position)
                neighbor.f = neighbor.g + neighbor.h
                neighbor.parent = current_node

                if not any(node.position == neighbor.position for _, node in open_list):
                    heapq.heappush(open_list, (neighbor.f, neighbor))
                else:
                    # Update existing node if this path is better
                    for i, (f, node) in enumerate(open_list):
                        if node.position == neighbor.position and neighbor.g < node.g:
                            open_list[i] = (neighbor.f, neighbor)
                            heapq.heapify(open_list)
                            break

            progress_counter += 1
            if progress_threshold == 0 or progress_counter >= progress_threshold:
                if timer() - time0 > 1.0 / fps and self.animated:
                    time0 = timer()
                    current_path = []
                    temp_node = current_node
                    current_path_length = 0
                    while temp_node:
                        current_path.append(temp_node.position)
                        if temp_node.parent:
                            current_path_length += self.heuristic(temp_node.position, temp_node.parent.position)
                        temp_node = temp_node.parent
                    current_path = current_path[::-1]
                    self.t_pathlength.set_val(f"{current_path_length:.2f}")
                    self.visualize_progress(closed_set, [node for _, node in open_list], current_path)
                    progress_counter = 0

        print("No path found.")

    @profile
    def visualize_progress(self, closed_set, open_list, current_path):
        self.ax.clear()
        colors = ['white', 'black', 'orange', 'red', 'lavenderblush']
        color_map = ListedColormap(colors)

        grid = self.grid_to_numeric(self.grids[self.current_floor])
        self.ax.imshow(grid.T, cmap=color_map, interpolation='nearest')

        # Plot closed set
        closed_on_floor = [node for node in closed_set if node[2] == self.current_floor]
        if closed_on_floor:
            closed_x, closed_y = zip(*[(node[0], node[1]) for node in closed_on_floor])
            self.ax.scatter(closed_x, closed_y, color='blue', alpha=0.1, s=20)

        # Plot open list
        open_on_floor = [node.position for node in open_list if node.position[2] == self.current_floor]
        if open_on_floor:
            open_x, open_y = zip(*[(node[0], node[1]) for node in open_on_floor])
            self.ax.scatter(open_x, open_y, color='aqua', alpha=0.4, s=20)

        # Plot current path
        current_path_on_floor = [node for node in current_path if node[2] == self.current_floor]
        if len(current_path_on_floor) > 1:
            path_x, path_y = zip(*[(node[0], node[1]) for node in current_path_on_floor])
            self.ax.plot(path_x, path_y, color='yellow', linewidth=2)

        # Plot start and goal
        if self.start and self.start[2] == self.current_floor:
            self.ax.plot(self.start[0], self.start[1], 'go', markersize=10)
        if self.goal and self.goal[2] == self.current_floor:
            self.ax.plot(self.goal[0], self.goal[1], 'ro', markersize=10)

        self.ax.set_title(f'Floor {self.current_floor + 1}')
        self.ax.axis('off')
        plt.draw()
        plt.pause(0.01)

    def visualize_path(self):
        self.ax.clear()
        colors = ['white', 'gray', 'brown', 'red', 'beige']
        color_map = ListedColormap(colors)

        grid = self.grid_to_numeric(self.grids[self.current_floor])
        self.ax.imshow(grid.T, cmap=color_map, interpolation='nearest')

        if self.path:
            path_on_floor = [node for node in self.path if node[2] == self.current_floor]
            for i in range(len(path_on_floor) - 1):
                start = path_on_floor[i]
                end = path_on_floor[i + 1]

                # Check if this segment is continuous or a jump
                if (abs(end[0] - start[0]) > 1 or abs(end[1] - start[1]) > 1):
                    # Discontinuous segment - draw in green
                    self.ax.plot([start[0], end[0]], [start[1], end[1]], color='green', linewidth=2, linestyle='--')
                else:
                    # Continuous segment - draw in blue
                    self.ax.plot([start[0], end[0]], [start[1], end[1]], color='blue', linewidth=2)

        if self.start and self.start[2] == self.current_floor:
            self.ax.plot(self.start[0], self.start[1], 'go', markersize=10)
        if self.goal and self.goal[2] == self.current_floor:
            self.ax.plot(self.goal[0], self.goal[1], 'ro', markersize=10)

        self.ax.set_title(f'Floor {self.current_floor + 1}')
        self.ax.axis('off')
        plt.draw()
        plt.pause(0.001)

    def get_current_path(self, node):
        path = []
        while node:
            path.append(node.position)
            node = node.parent
        return path[::-1]


def main():
    fn = askopenfilename(filetypes=[("json files", "*.json")])
    pathfinder = InteractiveBIMPathfinder(fn)
    plt.show()


if __name__ == "__main__":
    main()
