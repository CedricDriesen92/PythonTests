import ifcopenshell
import ifcopenshell.geom
import numpy as np
import matplotlib.pyplot as plt
import json
import tkinter as tk
from tkinter.filedialog import askopenfilename
tk.Tk().withdraw() # part of the import if you are not using other tkinter functions

wall_types = ['IfcWall', 'IfcWallStandardCase', 'IfcColumn', 'IfcCurtainWall', 'IfcWindow']
floor_types = ['IfcSlab', 'IfcFloor']
door_types = ['IfcDoor']
stair_types = ['IfcStair', 'IfcStairFlight']
all_types = wall_types + floor_types + door_types + stair_types


def load_ifc_file(file_path):
    return ifcopenshell.open(file_path)


def calculate_bounding_box_and_floors(ifc_file):
    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True)

    bbox = {
        'min_x': float('inf'), 'min_y': float('inf'), 'min_z': float('inf'),
        'max_x': float('-inf'), 'max_y': float('-inf'), 'max_z': float('-inf')
    }

    floor_elevations = set()

    # Process all elements to find the actual min and max Z values
    for item in ifc_file.by_type("IfcProduct"):
        if item.Representation:
            try:
                shape = ifcopenshell.geom.create_shape(settings, item)
                verts = shape.geometry.verts
                for i in range(0, len(verts), 3):
                    x, y, z = verts[i:i + 3]
                    bbox['min_x'] = min(bbox['min_x'], x)
                    bbox['min_y'] = min(bbox['min_y'], y)
                    bbox['min_z'] = min(bbox['min_z'], z)
                    bbox['max_x'] = max(bbox['max_x'], x)
                    bbox['max_y'] = max(bbox['max_y'], y)
                    bbox['max_z'] = max(bbox['max_z'], z)
            except RuntimeError:
                pass

    # Use IfcBuildingStorey for initial floor detection
    for item in ifc_file.by_type("IfcBuildingStorey"):
        elevation = item.Elevation
        if elevation is not None:
            floor_elevations.add(float(elevation))

    floor_elevations = sorted(list(floor_elevations))

    # If no floors detected or unusual elevations, create floors based on bounding box
    if not floor_elevations or min(floor_elevations) < bbox['min_z'] or max(floor_elevations) > bbox['max_z']:
        num_floors = max(1, int((bbox['max_z'] - bbox['min_z']) / 3))  # Assume 3m floor height
        floor_elevations = np.linspace(bbox['min_z'], bbox['max_z'], num_floors + 1)[:-1]

    floors = []
    for i, e in enumerate(floor_elevations):
        if i < len(floor_elevations) - 1:
            next_e = floor_elevations[i + 1]
        else:
            next_e = bbox['max_z']

        height = next_e - e
        if height > 0:  # Accept any positive height
            floors.append({'elevation': e, 'height': height})

    if not floors:
        print("Warning: No valid floors found. Creating a single floor based on bounding box.")
        floors = [{'elevation': bbox['min_z'], 'height': bbox['max_z'] - bbox['min_z']}]

    return bbox, floors


def create_faux_3d_grid(bbox, floors, grid_size=0.2):
    x_size = bbox['max_x'] - bbox['min_x']
    y_size = bbox['max_y'] - bbox['min_y']

    x_cells = int(np.ceil(x_size / grid_size)) + 10  # +2 for one extra on each side
    y_cells = int(np.ceil(y_size / grid_size)) + 10  # +2 for one extra on each side

    return [np.full((x_cells, y_cells), 'empty', dtype=object) for _ in floors]

def process_element(element, grids, bbox, floors, grid_size, total_elements, current_element):
    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True)

    print(f"Processing: {current_element}/{total_elements} elements ({current_element / total_elements * 100:.2f}% done)")
    try:
        shape = ifcopenshell.geom.create_shape(settings, element)
        verts = shape.geometry.verts
        faces = shape.geometry.faces
    except RuntimeError:
        print(f"Failed to process: {element.is_a()}")
        return

    if element.is_a() in wall_types:
        element_type = 'wall'
    elif element.is_a() in door_types:
        element_type = 'door'
    elif element.is_a() in stair_types:
        print("stair found")
        element_type = 'stair'
    elif element.is_a() in floor_types:
        element_type = 'floor'
    else:
        return  # Skip other types

    if not verts:
        print(f"Warning: No vertices found for {element.is_a()} (ID: {element.id()})")
        return

    min_z = min(verts[i+2] for i in range(0, len(verts), 3))
    max_z = max(verts[i+2] for i in range(0, len(verts), 3))

    print(f"Processing: {element.is_a()} (ID: {element.id()}) on floor(s): ", end="")
    for floor_index, floor in enumerate(floors):
        if min_z < floor['elevation'] + floor['height'] and max_z > floor['elevation']:
            print(f"{floor_index + 1}", end=" ")
            for i in range(0, len(faces), 3):
                triangle = [verts[faces[i]*3:faces[i]*3+3],
                            verts[faces[i+1]*3:faces[i+1]*3+3],
                            verts[faces[i+2]*3:faces[i+2]*3+3]]
                mark_cells(triangle, grids[floor_index], bbox, floor, grid_size, element_type)
    print()


def trim_and_pad_grids(grids, padding=1):
    trimmed_grids = []
    min_x_global = float('inf')
    max_x_global = float('-inf')
    min_y_global = float('inf')
    max_y_global = float('-inf')

    for grid in grids:
        # Find the bounds of non-empty and non-floor cells
        non_empty = np.argwhere((grid != 'empty') & (grid != 'floor'))
        if len(non_empty) == 0:
            trimmed_grids.append(grid)  # If the grid is entirely empty or floor, don't trim
            continue

        min_x, min_y = non_empty.min(axis=0)
        max_x, max_y = non_empty.max(axis=0)
        min_x_global = min(min_x_global, min_x)
        max_x_global = max(max_x_global, max_x)
        min_y_global = min(min_y_global, min_y)
        max_y_global = max(max_y_global, max_y)

    for grid in grids:
        # Trim the grid
        trimmed = grid[max(0, min_x_global - padding):min(grid.shape[0], max_x_global + padding + 1),
                  max(0, min_y_global - padding):min(grid.shape[1], max_y_global + padding + 1)]

        # Add padding if necessary
        padded = np.full((trimmed.shape[0] + 2 * padding, trimmed.shape[1] + 2 * padding), 'empty', dtype=object)
        padded[padding:-padding, padding:-padding] = trimmed

        trimmed_grids.append(padded)

    return trimmed_grids

def mark_cells(triangle, grid, bbox, floor, grid_size, element_type):
    min_x = min(p[0] for p in triangle)
    max_x = max(p[0] for p in triangle)
    min_y = min(p[1] for p in triangle)
    max_y = max(p[1] for p in triangle)
    min_z = min(p[2] for p in triangle)
    max_z = max(p[2] for p in triangle)

    if min_z < floor['elevation'] + floor['height'] and max_z > floor['elevation']:
        start_x = max(0, int((min_x - bbox['min_x']) / grid_size))
        end_x = min(grid.shape[0] - 1, int((max_x - bbox['min_x']) / grid_size))
        start_y = max(0, int((min_y - bbox['min_y']) / grid_size))
        end_y = min(grid.shape[1] - 1, int((max_y - bbox['min_y']) / grid_size))

        for x in range(start_x, end_x + 1):
            for y in range(start_y, end_y + 1):
                current = grid[x, y]
                if element_type == 'door' or (element_type == 'stair' and current != 'door') or (
                        element_type == 'wall' and current not in ['door', 'stair']):
                    grid[x, y] = element_type
                if element_type == 'floor' and current == 'empty':
                    grid[x, y] = element_type

def create_navigation_grid(ifc_file_path, grid_size=0.2):
    ifc_file = load_ifc_file(ifc_file_path)
    print("IFC file loaded...")
    bbox, floors = calculate_bounding_box_and_floors(ifc_file)
    print(f"Bounding box and floors calculated... grid size: {grid_size}")
    print(f"Number of floors: {len(floors)}")
    for i, floor in enumerate(floors):
        print(f"Floor {i + 1}: Elevation = {floor['elevation']}, Height = {floor['height']}")

    grids = create_faux_3d_grid(bbox, floors, grid_size)
    print("Empty grids created...")
    print(f"Grid dimensions: {grids[0].shape}")

    elements_all = list(ifc_file.by_type('IfcProduct'))
    elements = [element for element in elements_all if element.is_a() in all_types]
    total_elements = len(elements)

    for current_element, element in enumerate(elements, 1):
        if element.Representation:
            process_element(element, grids, bbox, floors, grid_size, total_elements, current_element)

    print("\nProcessing complete!")
    grids = trim_and_pad_grids(grids)

    x_size = grids[0].shape[0] * grid_size
    y_size = grids[0].shape[1] * grid_size
    bbox['min_x'] -= grid_size
    bbox['min_y'] -= grid_size
    bbox['max_x'] = bbox['min_x'] + x_size
    bbox['max_y'] = bbox['min_y'] + y_size

    return grids, bbox, floors


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


def export_grids(grids, bbox, floors, grid_size, filename):
    data = {
        'grids': [grid.tolist() for grid in grids],
        'bbox': bbox,
        'floors': floors,
        'grid_size': grid_size,
    }
    with open(filename, 'w') as f:
        json.dump(data, f)

def main():
    grid_size = input("Please select grid size in meters (0.3 default): ")
    if grid_size:
        grid_size = float(grid_size)
    else:
        grid_size = 0.3
    if grid_size < 0 or grid_size > 10000:
        grid_size = 0.3
        print("invalid size, 0.3 selected")
    fn = askopenfilename(filetypes=[("ifc files", "*.ifc")])
    ifc_file_path = fn
    try:
        grids, bbox, floors = create_navigation_grid(ifc_file_path, grid_size=grid_size)
        visualize_grids(grids, floors, grid_size)
        export_grids(grids, bbox, floors, grid_size, 'bim_grids.json')
        print("Grid creation, visualization, and export successful.")
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please check your IFC file and ensure it contains the necessary building elements.")

if __name__ == "__main__":
    main()