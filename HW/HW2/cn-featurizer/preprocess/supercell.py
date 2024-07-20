import numpy as np
from preprocess.cif_parser import *
from util.unit import *
from collections import defaultdict
from collections import Counter

def calculate_distance(point1, point2, cell_lengths, angles):
    """
    Calculates the Euclidean distance between two points using the given cell lengths and angles.
    """
    delta_x1, delta_y1, delta_z1, label1 = list(map(float, point1[:-1])) + [point1[-1]]
    delta_x2, delta_y2, delta_z2, label2 = list(map(float, point2[:-1])) + [point2[-1]]

    result = (
        (cell_lengths[0] * (delta_x1 - delta_x2))**2 +
        (cell_lengths[1] * (delta_y1 - delta_y2))**2 +
        (cell_lengths[2] * (delta_z1 - delta_z2))**2 +
        2 * cell_lengths[1] * cell_lengths[2] * np.cos(angles[0]) * (delta_y1 - delta_y2) * (delta_z1 - delta_z2) +
        2 * cell_lengths[2] * cell_lengths[0] * np.cos(angles[1]) * (delta_z1 - delta_z2) * (delta_x1 - delta_x2) +
        2 * cell_lengths[0] * cell_lengths[1] * np.cos(angles[2]) * (delta_x1 - delta_x2) * (delta_y1 - delta_y2)
    )

    distance = np.sqrt(result)

    return distance, label1, label2


def get_coords_list(block, loop_values):
    """
    Computes the new coordinates after applying symmetry operations to the initial coordinates.
    """
    
    loop_length = len(loop_values[0])
    coords_list = []
    for i in range(loop_length):
        atom_site_x, atom_site_y, atom_site_z = remove_string_braket(loop_values[4][i]), remove_string_braket(loop_values[5][i]), remove_string_braket(loop_values[6][i])
        atom_site_label = loop_values[0][i]
        atom_type_symbol = loop_values[1][i]

        all_coords = get_coords_after_sym_op(block, float(atom_site_x), float(atom_site_y), float(atom_site_z), atom_site_label)
        coords_list.append(all_coords)

    return coords_list


def get_coords_after_sym_op(block, atom_site_fract_x, atom_site_fract_y, atom_site_fract_z, atom_site_label):
    """
    Generates a list of coordinates for each atom site in the block.
    """
    all_coords = set()
    symmetry_operation_loop = None

    # Find appropriate loop based on whichever is present in the block
    if block.find_loop("_space_group_symop_operation_xyz"):
        symmetry_operation_loop = block.find_loop("_space_group_symop_operation_xyz")
    elif block.find_loop("_symmetry_equiv_pos_as_xyz"):
        symmetry_operation_loop = block.find_loop("_symmetry_equiv_pos_as_xyz")

    if not symmetry_operation_loop:
        print("Neither '_symmetry_equiv_pos_as_xyz' nor '_space_group_symop_operation_xyz' were found in the block.")
        return []

    for operation in symmetry_operation_loop:
        operation = operation.replace("'", "")
        try:
            op = gemmi.Op(operation)
            new_x, new_y, new_z = op.apply_to_xyz([atom_site_fract_x, atom_site_fract_y, atom_site_fract_z])
            new_x = round(new_x, 5)
            new_y = round(new_y, 5)
            new_z = round(new_z, 5)

            all_coords.add((new_x, new_y, new_z, atom_site_label))

        except RuntimeError as e:
            print(f"Skipping operation '{operation}': {str(e)}")
            raise RuntimeError("An error occurred while processing symmetry operation") from e

    return list(all_coords)


def get_points_and_labels(all_coords_list, loop_values):
    """
    Process coordinates and loop values to extract points, labels, and atom types.
    """
    all_points = []
    unique_labels = []
    unique_atoms_tuple = []
    for i, all_coords in enumerate(all_coords_list):
        points = np.array([list(map(float, coord[:-1])) for coord in all_coords])
        atom_site_label = loop_values[0][i]
        atom_site_type = loop_values[1][i]
        unique_labels.append(atom_site_label)
        unique_atoms_tuple.append(atom_site_type)
        all_points.extend(shift_and_append_points(points, atom_site_label))

        if get_atom_type(atom_site_label) != atom_site_type:
            raise RuntimeError("Different elements found in atom site and label")
    
    # print("all_points:", all_points)
    # print("unique_labels:", unique_labels)
    # print("unique_atoms_tuple:", unique_atoms_tuple)

    return list(set(all_points)), unique_labels, unique_atoms_tuple


def shift_and_append_points(points, atom_site_label):
    """
    Shift and duplicate points to create a 2 by 2 by 2 supercell.
    """
    shifts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
                        [-1, 0, 0], [0, -1, 0], [-1, -1, 0], [0, 0, -1], [1, 0, -1], [0, -1, -1], [-1, -1, -1]])
    shifted_points = points[:, None, :] + shifts[None, :, :]
    all_points = []
    for point_group in shifted_points:
        for point in point_group:
            new_point = (*np.round(point,5), atom_site_label)
            all_points.append(new_point)

    return all_points


def get_atomic_pair_list(flattened_points, cell_lengths, angles):
    """
    Calculate atomic distances and properties between pairs of points.
    The distance must be greater than 0.1 Å
    """
        
    atomic_info_list = []
    seen_pairs = set()  # This set will track pairs that we've already processed

    for i, point1 in enumerate(flattened_points):
        distances_from_point_i = []

        for j, point2 in enumerate(flattened_points):
            if i != j:
                pair = tuple(sorted([i, j]))  # Sort the pair so (i, j) is treated as equivalent to (j, i)
                if pair not in seen_pairs:  # Check if we've already processed this pair
                    distance, label1, label2 = calculate_distance(point1, point2, cell_lengths, angles)
                    if abs(distance) > 0.1:  # Add the pair if the distance is greater than 0.1 Å
                        distances_from_point_i.append({
                            'point_pair': (i + 1, j + 1),
                            'labels': (label1, label2),
                            'coordinates': (point1[:3], point2[:3]),  # include coordinates
                            'distance': np.round(distance, 5)
                        })
                        seen_pairs.add(pair)  # Add the pair to the set of seen pairs

        distances_from_point_i.sort(key=lambda x: x['distance'])
        atomic_info_list.extend(distances_from_point_i)

    return atomic_info_list


def exceeds_atom_count_limit(all_points, max_atoms_count):
    """
    Checks if the number of unique atomic positions after applying symmetry operations 
    exceeds the specified atom count limit.
    """
    return len(all_points) > max_atoms_count


def swap_labels_and_coordinates(pair, label):
    # If label is not the first one, swap labels, coordinates, pair
    if pair['labels'][0] != label:
        pair['labels'] = pair['labels'][::-1]
        pair['coordinates'] = pair['coordinates'][::-1]
        pair['point_pair'] = pair['point_pair'][::-1]

    return pair


def filter_and_swap_pairs(atomic_pair_list, label):
    """
    Filter and swap pairs to to ensure the label exists in either in the ith or jth atom in each pair.
    """
    filtered_atomic_pair_list = [pair for pair in atomic_pair_list if label in pair['labels']]

    for j in range(len(filtered_atomic_pair_list)):
        filtered_atomic_pair_list[j] = swap_labels_and_coordinates(filtered_atomic_pair_list[j], label)
    
    return filtered_atomic_pair_list


def get_unique_distances(filtered_atomic_pair_list, label):
    """
    Get unique atomic distances based on a given label.
    """
    unique_distances = set()

    for pair in filtered_atomic_pair_list:
        if label in pair['labels']:
            unique_distances.add(round(pair['distance'], 2))
    
    # print("Unique distances", unique_distances)
            
    return unique_distances


def find_most_common_distances_and_points(filtered_atomic_pair_list, label, unique_distances):
    """
    Determines the common points among lists of points associated with various distances.
    """
    num_distances = len(unique_distances)
    max_distance_counts = [0] * num_distances
    points_with_most_min_distances = [[] for _ in range(num_distances)]

    counted_pairs = set()
    point_label_counts = {i: {} for i in range(num_distances)}

    for pair in filtered_atomic_pair_list:
        if pair['labels'][0] == label:
            sorted_pair = ((pair['point_pair'][0], pair['labels'][0]), (pair['point_pair'][1], pair['labels'][1]))

            if sorted_pair not in counted_pairs:
                for idx, dist in enumerate(unique_distances):
                    if rounded_distance(pair['distance']) == rounded_distance(dist):
                        for point, point_label in sorted_pair:
                            if point_label == label:
                                new_count = point_label_counts[idx].get((point, point_label), 0) + 1
                                point_label_counts[idx][(point, point_label)] = new_count

                                if new_count > max_distance_counts[idx]:
                                    max_distance_counts[idx] = new_count
                                    points_with_most_min_distances[idx] = [(point, point_label)]
                                elif new_count == max_distance_counts[idx]:
                                    points_with_most_min_distances[idx].append((point, point_label))

                counted_pairs.add(sorted_pair)
    
    return max_distance_counts, points_with_most_min_distances


def find_common_points(points_with_most_min_distances):
    """
    Finds the common points among lists of points associated with various distances.
    """
    set_of_min_distances = [set(points) for points in points_with_most_min_distances]
    # print("set_of_min_distances", set_of_min_distances)
    common_points = set.intersection(*set_of_min_distances)
    return common_points


def get_atom_pair_info_dict(unique_atom_labels, atomic_pairs):
    """
    1. Filters the atomic pairs to those that match the label.
    2. Extracts the unique distances associated with that label.
    3. Finds the most common distances and points associated with those distances.
    4. Identifies the point that appears most frequently across the top six shortest distances.
    5. Collects and sorts atomic pairs related to the point found in step 4.
    6. Constructs and returns a dictionary containing this collected information for each atom label.
    """
    atom_pairs_info_dict = {}

    for idx, atom_label in enumerate(unique_atom_labels):

        atom_pairs_filtered = filter_and_swap_pairs(atomic_pairs, atom_label)
        unique_distances = get_unique_distances(atom_pairs_filtered, atom_label)

        top_six_distances = sorted(unique_distances)[:6]
        # print(atom_label, top_six_distances)

        most_common_distance_counts, points_with_common_distances = find_most_common_distances_and_points(
            atom_pairs_filtered, atom_label, top_six_distances
        )

        shared_points = None
        for i in range(6, 0, -1):
            shared_points = find_common_points(points_with_common_distances[:i])
            if shared_points:
                # print(f"The point with the largest occurrences of the {i} shortest distances are found")
                break
            # else:
            #     # print(f"The point with the largest occurrences of the {i} shortest distances are NOT found")

        point_with_most_common_distances = list(shared_points)[0]

        count_top_two_common_distances = most_common_distance_counts[:2]

        atom_pairs_related_to_point = [
            (pair['point_pair'], pair['labels'], pair['distance'], pair['coordinates'])
            for pair in atom_pairs_filtered
            if pair['point_pair'][0] == point_with_most_common_distances[0] or pair['point_pair'][1] == point_with_most_common_distances[0]
        ]

        atom_pairs_related_to_point_sorted = sorted(atom_pairs_related_to_point, key=lambda x: x[2])

        atom_pairs_info_dict.setdefault(atom_label, {
            "pair_info": atom_pairs_related_to_point_sorted,
            "shortest_distances": top_six_distances[:2],
            "shortest_distances_count": count_top_two_common_distances
        })



    return atom_pairs_info_dict


def process_cell_data(CIF_block):
    """
    Processes the CIF block data to retrieve cell dimensions and angles.
    """
    # Extract cell dimensions and angles from CIF block
    cell_lengths_angles = get_unit_cell_lengths_angles(CIF_block)
    cell_length_a, cell_length_b, cell_length_c, alpha_deg, beta_deg, gamma_deg = cell_lengths_angles
    
    # Convert angles from degrees to radians
    alpha_rad, beta_rad, gamma_rad = get_radians_from_degrees([alpha_deg, beta_deg, gamma_deg])

    # Store angles in radians and cell lengths in a list
    cell_angles_rad = [alpha_rad, beta_rad, gamma_rad]
    cell_lengths = [cell_length_a, cell_length_b, cell_length_c]

    return cell_lengths, cell_angles_rad


def get_atomic_pair_list_dict(all_points, unique_labels, cell_lengths, cell_angles_rad):
    """
    Extracts and processes atomic data from the CIF block.
    """

    # Calculate atomic pairs based on cell lengths and angles
    atomic_pair_list = get_atomic_pair_list(all_points, cell_lengths, cell_angles_rad)
    
    # Create a dictionary containing information about atom pairs
    atom_pair_info_dict = get_atom_pair_info_dict(unique_labels, atomic_pair_list)

    return atomic_pair_list, atom_pair_info_dict


def swap_pairs(pair_info, counts, label, dist_type, most_common_point):
    """
    Swaps pairs based on the most commont point as the ith atom
    """

    swapped_pair_info = []

    for pair in pair_info[:counts[label][dist_type]]:
        ith_point = pair[0][0]
        if ith_point != most_common_point:
            # Swap ith and jth in the pair if the ith label does not match the most common label
            swapped_pair = ((pair[0][1], pair[0][0]), (pair[1][1], pair[1][0]), pair[2], (pair[3][1], pair[3][0]))
            swapped_pair_info.append(swapped_pair)
        else:
            swapped_pair_info.append(pair)
    
    return swapped_pair_info
