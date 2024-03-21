import numpy as np
from scipy.spatial import ConvexHull

def compute_polyhedron_metrics(polyhedron_points, central_atom_coord, hull):
    """
    Computes various metrics related to a given polyhedron defined by its vertices.
    """
    center_of_mass = np.mean(polyhedron_points[hull.vertices, :], axis=0)
    vector_to_center_of_mass = center_of_mass - central_atom_coord
    distance_to_center = np.linalg.norm(vector_to_center_of_mass)

    edges = set()
    for simplex in hull.simplices:
        for i in range(-1, len(simplex)-1):
            if (simplex[i], simplex[i+1]) not in edges and (simplex[i+1], simplex[i]) not in edges:
                edges.add((simplex[i], simplex[i+1]))

    number_of_edges = len(edges)
    number_of_faces = len(hull.simplices)
    number_of_vertices = len(polyhedron_points)

    face_centers = np.mean(polyhedron_points[hull.simplices], axis=1)
    distances_to_faces = np.linalg.norm(face_centers - central_atom_coord, axis=1)
    shortest_distance_to_face = np.min(distances_to_faces)

    edge_centers = np.array([(polyhedron_points[edge[0]] + polyhedron_points[edge[1]]) / 2 for edge in edges])
    distances_to_edges = np.linalg.norm(edge_centers - central_atom_coord, axis=1)
    shortest_distance_to_edge = np.min(distances_to_edges)

    radius_of_inscribed_sphere = shortest_distance_to_face
    volume_of_inscribed_sphere = 4 / 3 * np.pi * radius_of_inscribed_sphere**3

    packing_efficiency = volume_of_inscribed_sphere / hull.volume

    return {
        "Volume_of_polyhedron": hull.volume,
        "distance_to_center": distance_to_center,
        "number_of_vertices": number_of_vertices,
        "number_of_edges": number_of_edges,
        "number_of_faces": number_of_faces,
        "shortest_distance_to_face": shortest_distance_to_face,
        "shortest_distance_to_edge": shortest_distance_to_edge,
        "volume_of_inscribed_sphere": volume_of_inscribed_sphere,
        "packing_efficiency": packing_efficiency
    }
