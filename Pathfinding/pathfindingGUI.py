import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import heapq
from matplotlib.widgets import Button


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
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.setup_plot()

    def load_grid_data(self, filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        grids = [np.array(grid) for grid in data['grids']]
        return grids, data['bbox'], data['floors'], data['grid_size']

    def setup_plot(self):
        plt.subplots_adjust(bottom=0.2)
        self.ax_prev = plt.axes([0.2, 0.05, 0.1, 0.075])
        self.ax_next = plt.axes([0.31, 0.05, 0.1, 0.075])
        self.ax_start = plt.axes([0.42, 0.05, 0.1, 0.075])
        self.ax_goal = plt.axes([0.53, 0.05, 0.1, 0.075])
        self.ax_run = plt.axes([0.64, 0.05, 0.1, 0.075])

        self.b_prev = Button(self.ax_prev, 'Previous')
        self.b_next = Button(self.ax_next, 'Next')
        self.b_start = Button(self.ax_start, 'Set Start')
        self.b_goal = Button(self.ax_goal, 'Set Goal')
        self.b_run = Button(self.ax_run, 'Run A*')

        self.b_prev.on_clicked(self.prev_floor)
        self.b_next.on_clicked(self.next_floor)
        self.b_start.on_clicked(self.set_start_mode)
        self.b_goal.on_clicked(self.set_goal_mode)
        self.b_run.on_clicked(self.run_astar)

        self.mode = None
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

        self.update_plot()

    def grid_to_numeric(self, grid):
        element_types = ['empty', 'wall', 'door', 'stair', 'floor']
        numeric_grid = np.zeros_like(grid, dtype=int)
        for i, element_type in enumerate(element_types):
            numeric_grid[grid == element_type] = i
        return numeric_grid

    def update_plot(self):
        self.ax.clear()
        colors = ['white', 'gray', 'brown', 'red', 'beige']
        color_map = ListedColormap(colors)

        grid = self.grid_to_numeric(self.grids[self.current_floor])
        self.ax.imshow(grid.T, cmap=color_map, interpolation='nearest')

        if self.start:
            self.ax.plot(self.start[0], self.start[1], 'go', markersize=10)
        if self.goal:
            self.ax.plot(self.goal[0], self.goal[1], 'ro', markersize=10)

        self.ax.set_title(f'Floor {self.current_floor + 1}')
        self.ax.axis('off')
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

    def heuristic(self, a, b):
        return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2 + (b[2] - a[2]) ** 2) * self.grid_size

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

    def run_astar(self, event):
        if not self.start or not self.goal:
            print("Please set both start and goal points.")
            return

        open_list = []
        closed_set = set()
        start_node = Node(self.start)
        goal_node = Node(self.goal)

        heapq.heappush(open_list, (start_node.f, start_node))

        while open_list:
            current_node = heapq.heappop(open_list)[1]

            if current_node.position == goal_node.position:
                path = []
                while current_node:
                    path.append(current_node.position)
                    current_node = current_node.parent
                self.visualize_path(path[::-1])
                return

            closed_set.add(current_node.position)

            for neighbor in self.get_neighbors(current_node):
                if neighbor.position in closed_set:
                    continue

                neighbor.g = current_node.g + self.grid_size
                if self.grids[neighbor.position[2]][neighbor.position[0], neighbor.position[1]] == 'door':
                    neighbor.g += 5 * self.grid_size  # Higher cost for doors
                elif self.grids[neighbor.position[2]][neighbor.position[0], neighbor.position[1]] == 'stair':
                    neighbor.g += 2 * self.grid_size  # Moderate cost for stairs

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

            self.visualize_progress(closed_set, [node for _, node in open_list])

        print("No path found.")

    def visualize_progress(self, closed_set, open_list):
        self.ax.clear()
        colors = ['white', 'gray', 'brown', 'red', 'beige']
        color_map = ListedColormap(colors)

        grid = self.grid_to_numeric(self.grids[self.current_floor])
        self.ax.imshow(grid.T, cmap=color_map, interpolation='nearest')

        # Plot closed set
        closed_on_floor = [node for node in closed_set if node[2] == self.current_floor]
        if closed_on_floor:
            closed_x, closed_y = zip(*[(node[0], node[1]) for node in closed_on_floor])
            self.ax.scatter(closed_x, closed_y, color='blue', alpha=0.5, s=50)

        # Plot open list
        open_on_floor = [node.position for node in open_list if node.position[2] == self.current_floor]
        if open_on_floor:
            open_x, open_y = zip(*[(node[0], node[1]) for node in open_on_floor])
            self.ax.scatter(open_x, open_y, color='cyan', alpha=0.5, s=50)

        # Plot start and goal
        if self.start:
            self.ax.plot(self.start[0], self.start[1], 'go', markersize=10)
        if self.goal:
            self.ax.plot(self.goal[0], self.goal[1], 'ro', markersize=10)

        self.ax.set_title(f'Floor {self.current_floor + 1}')
        self.ax.axis('off')
        plt.draw()
        plt.pause(0.1)

    def visualize_path(self, path):
        self.ax.clear()
        colors = ['white', 'gray', 'brown', 'red', 'beige']
        color_map = ListedColormap(colors)

        grid = self.grid_to_numeric(self.grids[self.current_floor])
        self.ax.imshow(grid.T, cmap=color_map, interpolation='nearest')

        path_on_floor = [node for node in path if node[2] == self.current_floor]
        if path_on_floor:
            path_x, path_y = zip(*[(node[0], node[1]) for node in path_on_floor])
            self.ax.plot(path_x, path_y, color='blue', linewidth=2)

        if self.start:
            self.ax.plot(self.start[0], self.start[1], 'go', markersize=10)
        if self.goal:
            self.ax.plot(self.goal[0], self.goal[1], 'ro', markersize=10)

        self.ax.set_title(f'Floor {self.current_floor + 1}')
        self.ax.axis('off')
        plt.draw()


def main():
    pathfinder = InteractiveBIMPathfinder('bim_grids.json')
    plt.show()


if __name__ == "__main__":
    main()