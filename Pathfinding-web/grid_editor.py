import numpy as np


class InteractiveGridEditor:
    def __init__(self, grids, grid_size, floors, bbox):
        self.grids = grids
        self.grid_size = grid_size
        self.floors = floors
        self.bbox = bbox
        self.current_floor = 0

    def draw(self, floor, row, col, element_type):
        """
        Draw an element on the grid.

        :param floor: The floor number
        :param row: The row index
        :param col: The column index
        :param element_type: The type of element to draw ('wall', 'stair', 'door', 'floor', 'empty')
        """
        if 0 <= floor < len(self.grids) and 0 <= row < self.grids[floor].shape[0] and 0 <= col < \
                self.grids[floor].shape[1]:
            self.grids[floor][row, col] = element_type
        else:
            raise ValueError("Invalid grid coordinates")

    def draw_wall(self, floor, row, col):
        """Draw a wall at the specified location."""
        self.draw(floor, row, col, 'wall')

    def draw_stair(self, floor, row, col):
        """Draw a stair at the specified location."""
        self.draw(floor, row, col, 'stair')

    def draw_door(self, floor, row, col):
        """Draw a door at the specified location."""
        self.draw(floor, row, col, 'door')

    def draw_floor(self, floor, row, col):
        """Draw a floor at the specified location."""
        self.draw(floor, row, col, 'floor')

    def draw_empty(self, floor, row, col):
        """Draw an empty space at the specified location."""
        self.draw(floor, row, col, 'empty')

    def edit_grid(self, edits):
        """
        Apply multiple edits to the grid.

        :param edits: A list of dicts, each containing 'floor', 'row', 'col', and 'element_type'
        :return: The updated grids
        """
        for edit in edits:
            self.draw(edit['floor'], edit['row'], edit['col'], edit['element_type'])
        return self.grids

    def get_grid(self, floor):
        """Get the grid for a specific floor."""
        if 0 <= floor < len(self.grids):
            return self.grids[floor]
        else:
            raise ValueError("Invalid floor number")

    def get_all_grids(self):
        """Get all grids."""
        return self.grids

    def set_current_floor(self, floor):
        """Set the current floor."""
        if 0 <= floor < len(self.grids):
            self.current_floor = floor
        else:
            raise ValueError("Invalid floor number")

    def get_current_floor(self):
        """Get the current floor number."""
        return self.current_floor

    def get_grid_info(self):
        """Get information about the grid."""
        return {
            'num_floors': len(self.grids),
            'grid_size': self.grid_size,
            'floors': self.floors,
            'bbox': self.bbox
        }

    def clear_floor(self, floor):
        """Clear all elements on a specific floor, setting them to 'empty'."""
        if 0 <= floor < len(self.grids):
            self.grids[floor].fill('empty')
        else:
            raise ValueError("Invalid floor number")

    def flood_fill(self, floor, row, col, target_element, replacement_element):
        """
        Perform a flood fill operation starting from a specific point.

        :param floor: The floor number
        :param row: The starting row
        :param col: The starting column
        :param target_element: The element type to be replaced
        :param replacement_element: The new element type
        """
        if (0 <= floor < len(self.grids) and
                0 <= row < self.grids[floor].shape[0] and
                0 <= col < self.grids[floor].shape[1] and
                self.grids[floor][row, col] == target_element and
                target_element != replacement_element):
            self.grids[floor][row, col] = replacement_element

            # Recursively fill adjacent cells
            self.flood_fill(floor, row + 1, col, target_element, replacement_element)
            self.flood_fill(floor, row - 1, col, target_element, replacement_element)
            self.flood_fill(floor, row, col + 1, target_element, replacement_element)
            self.flood_fill(floor, row, col - 1, target_element, replacement_element)

    def validate_grid(self):
        """
        Validate the grid to ensure all elements are valid.

        :return: A list of errors, if any
        """
        valid_elements = {'wall', 'stair', 'door', 'floor', 'empty'}
        errors = []

        for floor in range(len(self.grids)):
            for row in range(self.grids[floor].shape[0]):
                for col in range(self.grids[floor].shape[1]):
                    element = self.grids[floor][row, col]
                    if element not in valid_elements:
                        errors.append(f"Invalid element '{element}' at floor {floor}, row {row}, col {col}")

        return errors if errors else None

    def resize_grid(self, new_rows, new_cols):
        """
        Resize all floor grids to the specified dimensions.

        :param new_rows: New number of rows
        :param new_cols: New number of columns
        """
        for floor in range(len(self.grids)):
            old_grid = self.grids[floor]
            new_grid = np.full((new_rows, new_cols), 'empty', dtype=object)

            # Copy the old grid into the new grid
            rows_to_copy = min(old_grid.shape[0], new_rows)
            cols_to_copy = min(old_grid.shape[1], new_cols)
            new_grid[:rows_to_copy, :cols_to_copy] = old_grid[:rows_to_copy, :cols_to_copy]

            self.grids[floor] = new_grid

        # Update bbox
        self.bbox['max_x'] = self.bbox['min_x'] + new_cols * self.grid_size
        self.bbox['max_y'] = self.bbox['min_y'] + new_rows * self.grid_size

    def add_floor(self):
        """Add a new floor to the grid."""
        new_floor = np.full_like(self.grids[0], 'empty', dtype=object)
        self.grids.append(new_floor)
        self.floors.append({'elevation': self.floors[-1]['elevation'] + self.floors[-1]['height'],
                            'height': self.floors[-1]['height']})

    def remove_floor(self, floor):
        """Remove a specific floor from the grid."""
        if 0 < floor < len(self.grids):  # Prevent removing the ground floor
            del self.grids[floor]
            del self.floors[floor]
        else:
            raise ValueError("Cannot remove the specified floor")

# You can add more methods here as needed for your specific requirements