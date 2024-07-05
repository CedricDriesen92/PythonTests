import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.placement
import numpy as np
from typing import Dict, List, Tuple, Any

# Constants
WALL_TYPES = ['IfcWall', 'IfcWallStandardCase', 'IfcColumn', 'IfcCurtainWall', 'IfcWindow']
FLOOR_TYPES = ['IfcSlab', 'IfcFloor']
DOOR_TYPES = ['IfcDoor']
STAIR_TYPES = ['IfcStair', 'IfcStairFlight']
ALL_TYPES = WALL_TYPES + FLOOR_TYPES + DOOR_TYPES + STAIR_TYPES

class IFCProcessor:
    def __init__(self, file_path: str, grid_size: float = 0.2):
        self.file_path = file_path
        self.grid_size = grid_size
        self.ifc_file = None
        self.bbox = None
        self.floors = None
        self.grids = None

    def process(self) -> Dict[str, Any]:
        try:
            self.ifc_file = self.load_ifc_file()
            self.bbox, self.floors = self.calculate_bounding_box_and_floors()
            self.grids = self.create_grids()
            self.process_elements()
            self.trim_grids()

            return {
                'grids': [grid.tolist() for grid in self.grids],
                'bbox': self.bbox,
                'floors': self.floors,
                'grid_size': self.grid_size
            }
        except Exception as e:
            raise RuntimeError(f"Error processing IFC file: {str(e)}")

    def load_ifc_file(self) -> ifcopenshell.file:
        try:
            return ifcopenshell.open(self.file_path)
        except Exception as e:
            raise RuntimeError(f"Error loading IFC file: {str(e)}")

    def calculate_bounding_box_and_floors(self) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        settings = ifcopenshell.geom.settings()
        settings.set(settings.USE_WORLD_COORDS, True)

        bbox = {
            'min_x': float('inf'), 'min_y': float('inf'), 'min_z': float('inf'),
            'max_x': float('-inf'), 'max_y': float('-inf'), 'max_z': float('-inf')
        }
        floor_elevations = set()

        for product in self.ifc_file.by_type('IfcProduct'):
            if product.ObjectPlacement:
                matrix = ifcopenshell.util.placement.get_local_placement(product.ObjectPlacement)
                for i in range(3):
                    bbox['min_' + 'xyz'[i]] = min(bbox['min_' + 'xyz'[i]], matrix[i][3])
                    bbox['max_' + 'xyz'[i]] = max(bbox['max_' + 'xyz'[i]], matrix[i][3])

        for storey in self.ifc_file.by_type("IfcBuildingStorey"):
            if storey.Elevation is not None:
                floor_elevations.add(float(storey.Elevation))

        floor_elevations = sorted(list(floor_elevations))

        if not floor_elevations or min(floor_elevations) < bbox['min_z'] or max(floor_elevations) > bbox['max_z']:
            num_floors = max(1, int((bbox['max_z'] - bbox['min_z']) / 3))  # Assume 3m floor height
            floor_elevations = np.linspace(bbox['min_z'], bbox['max_z'], num_floors + 1)[:-1]

        floors = []
        for i, elevation in enumerate(floor_elevations):
            next_elevation = floor_elevations[i + 1] if i < len(floor_elevations) - 1 else bbox['max_z']
            height = next_elevation - elevation
            if height > 0:
                floors.append({'elevation': elevation, 'height': height})

        if not floors:
            floors = [{'elevation': bbox['min_z'], 'height': bbox['max_z'] - bbox['min_z']}]

        return bbox, floors

    def create_grids(self) -> List[np.ndarray]:
        x_size = self.bbox['max_x'] - self.bbox['min_x']
        y_size = self.bbox['max_y'] - self.bbox['min_y']

        x_cells = int(np.ceil(x_size / self.grid_size)) + 10
        y_cells = int(np.ceil(y_size / self.grid_size)) + 10

        return [np.full((x_cells, y_cells), 'empty', dtype=object) for _ in self.floors]

    def process_elements(self) -> None:
        settings = ifcopenshell.geom.settings()
        settings.set(settings.USE_WORLD_COORDS, True)

        elements = [element for element in self.ifc_file.by_type('IfcProduct') if element.is_a() in ALL_TYPES]

        for element in elements:
            if element.Representation:
                self.process_single_element(element, settings)

    def process_single_element(self, element: ifcopenshell.entity_instance, settings: ifcopenshell.geom.settings) -> None:
        try:
            shape = ifcopenshell.geom.create_shape(settings, element)
            verts = shape.geometry.verts
            faces = shape.geometry.faces
        except RuntimeError:
            print(f"Failed to process: {element.is_a()}")
            return

        element_type = self.get_element_type(element)
        if element_type is None:
            return

        if not verts:
            print(f"Warning: No vertices found for {element.is_a()} (ID: {element.id()})")
            return

        min_x = min(verts[i] for i in range(0, len(verts), 3))
        max_x = max(verts[i] for i in range(0, len(verts), 3))
        min_y = min(verts[i+1] for i in range(0, len(verts), 3))
        max_y = max(verts[i+1] for i in range(0, len(verts), 3))
        min_z = min(verts[i+2] for i in range(0, len(verts), 3))
        max_z = max(verts[i+2] for i in range(0, len(verts), 3))

        for floor_index, floor in enumerate(self.floors):
            if min_z < floor['elevation'] + floor['height'] - 0.5 and max_z > floor['elevation'] + 0.5:
                if element_type == 'door':
                    # Use bounding box for doors
                    self.mark_door(floor_index, min_x, min_y, max_x, max_y, floor)
                else:
                    # Use detailed geometry for other elements
                    for i in range(0, len(faces), 3):
                        triangle = [verts[faces[i]*3:faces[i]*3+3],
                                    verts[faces[i+1]*3:faces[i+1]*3+3],
                                    verts[faces[i+2]*3:faces[i+2]*3+3]]
                        self.mark_cells(triangle, self.grids[floor_index], floor, element_type)

    def mark_door(self, floor_index: int, min_x: float, min_y: float, max_x: float, max_y: float, floor: Dict[str, float]) -> None:
        start_x = max(0, int((min_x - self.bbox['min_x']) / self.grid_size))
        end_x = min(self.grids[floor_index].shape[0] - 1, int((max_x - self.bbox['min_x']) / self.grid_size))
        start_y = max(0, int((min_y - self.bbox['min_y']) / self.grid_size))
        end_y = min(self.grids[floor_index].shape[1] - 1, int((max_y - self.bbox['min_y']) / self.grid_size))

        for x in range(start_x, end_x + 1):
            for y in range(start_y, end_y + 1):
                self.grids[floor_index][x, y] = 'door'

    def get_element_type(self, element: ifcopenshell.entity_instance) -> str:
        if element.is_a() in WALL_TYPES:
            return 'wall'
        elif element.is_a() in DOOR_TYPES:
            return 'door'
        elif element.is_a() in STAIR_TYPES:
            return 'stair'
        elif element.is_a() in FLOOR_TYPES:
            return 'floor'
        else:
            return None

    def mark_cells(self, triangle: List[List[float]], grid: np.ndarray, floor: Dict[str, float], element_type: str) -> None:
        min_x = min(p[0] for p in triangle)
        max_x = max(p[0] for p in triangle)
        min_y = min(p[1] for p in triangle)
        max_y = max(p[1] for p in triangle)
        min_z = min(p[2] for p in triangle)
        max_z = max(p[2] for p in triangle)

        if min_z < floor['elevation'] + floor['height'] and max_z > floor['elevation']:
            start_x = max(0, int((min_x - self.bbox['min_x']) / self.grid_size))
            end_x = min(grid.shape[0] - 1, int((max_x - self.bbox['min_x']) / self.grid_size))
            start_y = max(0, int((min_y - self.bbox['min_y']) / self.grid_size))
            end_y = min(grid.shape[1] - 1, int((max_y - self.bbox['min_y']) / self.grid_size))

            for x in range(start_x, end_x + 1):
                for y in range(start_y, end_y + 1):
                    current = grid[x, y]
                    if element_type == 'door' or (element_type == 'stair' and current != 'door') or (
                            element_type == 'wall' and current not in ['door', 'stair']):
                        grid[x, y] = element_type
                    if element_type == 'floor' and current == 'empty':
                        grid[x, y] = element_type

    def trim_grids(self, padding: int = 1) -> None:
        trimmed_grids = []
        min_x_global = float('inf')
        max_x_global = float('-inf')
        min_y_global = float('inf')
        max_y_global = float('-inf')

        for grid in self.grids:
            non_empty = np.argwhere((grid != 'empty') & (grid != 'floor'))
            if len(non_empty) == 0:
                trimmed_grids.append(grid)
                continue

            min_x, min_y = non_empty.min(axis=0)
            max_x, max_y = non_empty.max(axis=0)
            min_x_global = min(min_x_global, min_x)
            max_x_global = max(max_x_global, max_x)
            min_y_global = min(min_y_global, min_y)
            max_y_global = max(max_y_global, max_y)

        for grid in self.grids:
            trimmed = grid[max(0, min_x_global - padding):min(grid.shape[0], max_x_global + padding + 1),
                      max(0, min_y_global - padding):min(grid.shape[1], max_y_global + padding + 1)]

            padded = np.full((trimmed.shape[0] + 2 * padding, trimmed.shape[1] + 2 * padding), 'empty', dtype=object)
            padded[padding:-padding, padding:-padding] = trimmed

            trimmed_grids.append(padded)

        self.grids = trimmed_grids

        x_size = self.grids[0].shape[0] * self.grid_size
        y_size = self.grids[0].shape[1] * self.grid_size
        self.bbox['min_x'] -= self.grid_size
        self.bbox['min_y'] -= self.grid_size
        self.bbox['max_x'] = self.bbox['min_x'] + x_size
        self.bbox['max_y'] = self.bbox['min_y'] + y_size

def process_ifc_file(file_path: str, grid_size: float = 0.2) -> Dict[str, Any]:
    processor = IFCProcessor(file_path, grid_size)
    return processor.process()