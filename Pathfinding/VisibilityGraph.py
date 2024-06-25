import ifcopenshell
import ifcopenshell.geom
import numpy as np
from scipy.spatial import Delaunay
import networkx as nx
import matplotlib.pyplot as plt


def load_ifc_file(file_path):
    return ifcopenshell.open(file_path)


def extract_geometry(ifc_file):
    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True)

    floors = {}
    stairs = []
    elevators = []
    walls = {}
    doors = {}

    for storey in ifc_file.by_type("IfcBuildingStorey"):
        elevation = float(storey.Elevation)
        floors[elevation] = {"vertices": [], "walls": [], "doors": []}
        walls[elevation] = []
        doors[elevation] = []

    for product in ifc_file.by_type("IfcProduct"):
        if product.is_a() in ["IfcWall", "IfcColumn", "IfcDoor", "IfcStair", "IfcTransportElement"]:
            try:
                shape = ifcopenshell.geom.create_shape(settings, product)
                verts = shape.geometry.verts
                z_values = [verts[i + 2] for i in range(0, len(verts), 3)]
                min_z, max_z = min(z_values), max(z_values)

                for elevation in floors.keys():
                    if min_z <= elevation <= max_z:
                        floor_verts = [(verts[i], verts[i + 1]) for i in range(0, len(verts), 3)]
                        floors[elevation]["vertices"].extend(floor_verts)

                        if product.is_a() in ["IfcWall", "IfcColumn"]:
                            walls[elevation].append(floor_verts)
                        elif product.is_a() == "IfcDoor":
                            door_line = (floor_verts[0], floor_verts[-1])  # Start and end points of the door
                            doors[elevation].append(door_line)
                            floors[elevation]["doors"].append(door_line)
                if product.is_a() == "IfcStair":
                    stairs.append((min_z, max_z, np.mean(verts[::3]), np.mean(verts[1::3])))
                elif product.is_a() == "IfcTransportElement" and product.PredefinedType == "ELEVATOR":
                    elevators.append((min_z, max_z, np.mean(verts[::3]), np.mean(verts[1::3])))

            except RuntimeError:
                pass

    return floors, stairs, elevators, walls, doors


def create_weighted_visibility_graph(vertices, doors, walls, ax=None):
    vertices_array = np.array(vertices)
    tri = Delaunay(vertices_array)
    G = nx.Graph()

    if ax:
        # Draw walls
        for wall in walls:
            wall_array = np.array(wall)
            ax.plot(wall_array[:, 0], wall_array[:, 1], 'k-', linewidth=2)

        # Draw doors
        for door in doors:
            door_array = np.array(door)
            ax.plot(door_array[:, 0], door_array[:, 1], 'g-', linewidth=2)

        ax.triplot(vertices_array[:, 0], vertices_array[:, 1], tri.simplices, 'r-', alpha=0.2)
        ax.plot(vertices_array[:, 0], vertices_array[:, 1], 'bo', markersize=2)

    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                v1, v2 = tuple(vertices[simplex[i]]), tuple(vertices[simplex[j]])
                if is_visible(v1, v2, vertices):
                    weight = calculate_edge_weight(v1, v2, doors)
                    G.add_edge(v1, v2, weight=weight)
                    if ax:
                        ax.plot([v1[0], v2[0]], [v1[1], v2[1]], 'g-', alpha=0.5)
                        plt.pause(0.01)

    return G


def is_visible(p1, p2, obstacles):
    p1, p2 = tuple(p1), tuple(p2)
    for obstacle in obstacles:
        obstacle = tuple(obstacle)
        if obstacle != p1 and obstacle != p2:
            if intersects(p1, p2, obstacle, obstacle):
                return False
    return True

def ccw(A, B, C):
    (Ax, Ay), (Bx, By), (Cx, Cy) = A, B, C
    return ((Cy - Ay) * (Bx - Ax)) > ((By - Ay) * (Cx - Ax))


def intersects(p1, p2, p3, p4):
    # If p3 and p4 are the same (representing a point), duplicate it
    if p3 == p4:
        p4 = (p3[0] + 1e-6, p3[1] + 1e-6)  # Slightly offset to avoid degenerate cases

    # Ensure all points are tuples
    p1, p2, p3, p4 = map(lambda p: p[0] if isinstance(p[0], tuple) else p, [p1, p2, p3, p4])

    return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)


def calculate_edge_weight(v1, v2, doors):
    v1, v2 = tuple(v1), tuple(v2)
    distance = np.linalg.norm(np.array(v1) - np.array(v2))
    door_cost = 10  # Additional cost for passing through a door

    for door in doors:
        # If door is a tuple of points (e.g., start and end of a door line)
        if isinstance(door[0], tuple):
            if intersects(v1, v2, door[0], door[1]):
                return distance + door_cost
        else:  # If door is a single point
            if intersects(v1, v2, door, door):
                return distance + door_cost

    return distance


def connect_floors(G_multi, floors, stairs, elevators, ax=None):
    floor_elevations = sorted(floors.keys())

    for i in range(len(floor_elevations) - 1):
        lower_floor = floor_elevations[i]
        upper_floor = floor_elevations[i + 1]

        for stair in stairs:
            if stair[0] <= lower_floor and stair[1] >= upper_floor:
                lower_node = min(G_multi.nodes(floor=lower_floor),
                                 key=lambda n: ((n[0] - stair[2]) ** 2 + (n[1] - stair[3]) ** 2))
                upper_node = min(G_multi.nodes(floor=upper_floor),
                                 key=lambda n: ((n[0] - stair[2]) ** 2 + (n[1] - stair[3]) ** 2))
                G_multi.add_edge(lower_node, upper_node, weight=20)
                if ax:
                    ax.plot([lower_node[0], upper_node[0]], [lower_node[1], upper_node[1]], 'm-', linewidth=2)
                    plt.pause(0.1)

        for elevator in elevators:
            if elevator[0] <= lower_floor and elevator[1] >= upper_floor:
                lower_node = min(G_multi.nodes(floor=lower_floor),
                                 key=lambda n: ((n[0] - elevator[2]) ** 2 + (n[1] - elevator[3]) ** 2))
                upper_node = min(G_multi.nodes(floor=upper_floor),
                                 key=lambda n: ((n[0] - elevator[2]) ** 2 + (n[1] - elevator[3]) ** 2))
                G_multi.add_edge(lower_node, upper_node, weight=15)
                if ax:
                    ax.plot([lower_node[0], upper_node[0]], [lower_node[1], upper_node[1]], 'c-', linewidth=2)
                    plt.pause(0.1)


def create_multi_floor_graph(floors, stairs, elevators, walls, doors):
    G_multi = nx.Graph()
    fig, axs = plt.subplots(1, len(floors), figsize=(8 * len(floors), 8), squeeze=False)

    for i, (elevation, floor_data) in enumerate(floors.items()):
        ax = axs[0, i]
        ax.set_title(f"Floor at elevation {elevation}")
        G_floor = create_weighted_visibility_graph(floor_data["vertices"], doors[elevation], walls[elevation], ax)
        for node in G_floor.nodes():
            G_multi.add_node((node[0], node[1], elevation), floor=elevation)
        for u, v, data in G_floor.edges(data=True):
            G_multi.add_edge((u[0], u[1], elevation), (v[0], v[1], elevation), weight=data['weight'])

    connect_floors(G_multi, floors, stairs, elevators, ax)

    plt.tight_layout()
    plt.show()

    return G_multi


def find_path(G, start, end, ax=None):
    path = nx.astar_path(G, start, end, weight='weight',
                         heuristic=lambda u, v: np.linalg.norm(np.array(u) - np.array(v)))

    if ax:
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            ax.plot([u[0], v[0]], [u[1], v[1]], 'r-', linewidth=2)
            plt.pause(0.1)

    return path


def visualize_multi_floor_graph_and_path(G, path, floors, walls, doors):
    fig, axs = plt.subplots(1, len(floors), figsize=(6 * len(floors), 6), squeeze=False)

    for i, (elevation, floor_data) in enumerate(floors.items()):
        ax = axs[0, i]
        floor_nodes = [node for node in G.nodes() if G.nodes[node]['floor'] == elevation]
        floor_edges = [edge for edge in G.edges() if G.nodes[edge[0]]['floor'] == elevation]

        pos = {node: node[:2] for node in floor_nodes}
        nx.draw_networkx_edges(G.subgraph(floor_nodes), pos, ax=ax, alpha=0.2)
        nx.draw_networkx_nodes(G.subgraph(floor_nodes), pos, node_size=20, node_color='b', ax=ax)

        floor_path = [node for node in path if G.nodes[node]['floor'] == elevation]
        if len(floor_path) > 1:
            path_edges = list(zip(floor_path, floor_path[1:]))
            nx.draw_networkx_edges(G.subgraph(floor_path), pos, edgelist=path_edges, edge_color='r', width=2, ax=ax)

        ax.set_title(f"Floor at elevation {elevation}")
        ax.set_aspect('equal')
        ax.axis('off')

        visualize_floor(ax, floor_data, walls, doors)

    plt.tight_layout()
    plt.show()


def visualize_floor(ax, floor_data, walls, doors):
    # Plot walls
    for wall in walls:
        wall_array = np.array(wall)
        ax.plot(wall_array[:, 0], wall_array[:, 1], 'k-', linewidth=2, label='Walls')

    # Plot doors
    for door in doors:
        door_array = np.array(door)
        ax.plot(door_array[:, 0], door_array[:, 1], 'r-', linewidth=2, label='Doors')

    # Plot vertices
    vertices = np.array(floor_data["vertices"])
    ax.scatter(vertices[:, 0], vertices[:, 1], c='b', s=10, label='Vertices')

    # Set aspect ratio to equal for a proper 2D view
    ax.set_aspect('equal', 'box')

    # Remove duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    # Adjust limits to show all elements
    ax.autoscale()

def main():
    ifc_file = load_ifc_file("Duplex.ifc")
    floors, stairs, elevators, walls, doors = extract_geometry(ifc_file)
    G_multi = create_multi_floor_graph(floors, stairs, elevators, walls, doors)

    start_floor = min(floors.keys())
    end_floor = max(floors.keys())
    start = (*floors[start_floor]["vertices"][0], start_floor)
    end = (*floors[end_floor]["vertices"][-1], end_floor)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title("Pathfinding Visualization")
    path = find_path(G_multi, start, end, ax)
    plt.show()

    visualize_multi_floor_graph_and_path(G_multi, path, floors, walls, doors)

    total_cost = sum(G_multi[path[i]][path[i + 1]]['weight'] for i in range(len(path) - 1))
    print(f"Total path cost: {total_cost:.2f}")


if __name__ == "__main__":
    main()