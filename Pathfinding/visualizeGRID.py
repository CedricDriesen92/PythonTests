import numpy as np
import matplotlib.pyplot as plt
import json


def load_grid_data(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    grids = [np.array(grid) for grid in data['grids']]
    return grids, data['grid_size'], data['floors']


def visualize_grids(grids, floors, grid_size):
    num_floors = len(grids)
    fig, axs = plt.subplots(num_floors, 1, figsize=(8, 5 * num_floors), squeeze=False)

    colors = {
        'wall': 'black',
        'floor': 'lavenderblush',
        'door': 'orange',
        'stair': 'red',
        'empty': 'white'
    }

    for floor_index, (ax, floor) in enumerate(zip(axs.flat, floors)):
        for element_type in ['empty', 'wall', 'stair', 'floor', 'door']:  # Order matters for visibility
            y, x = np.where(grids[floor_index] == element_type)
            ax.scatter(x, y, c=colors[element_type], marker='s', s=grid_size * 80, edgecolors='none')

        ax.set_title(f'Floor {floor_index + 1} (Elevation: {floor["elevation"]:.2f}m)')
        ax.set_aspect('equal', 'box')
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()

grids, grid_size, floors = load_grid_data('bim_grids.json')
visualize_grids(grids, floors, grid_size)