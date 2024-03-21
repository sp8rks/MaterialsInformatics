from preprocess.cif_parser import get_atom_type
from util.unit import *
from collections import Counter
from scipy.spatial import ConvexHull
import pandas as pd
from featurizer.polyhedron import *
import featurizer.coordinate_number_dataframe as cn_dataframe
import preprocess.supercell as supercell

def update_difference_lists(dist, dist_sums, prev_dist_sums, diff_lists, first_iteration):
    """
    Updates the difference between consecutive distance sums, except during the first iteration.
    """
    if not first_iteration:
        for key, data in dist_sums.items():
            diff_lists[key].append(dist_sums[key] - prev_dist_sums[key])          
    
    return dist_sums

def get_counts_before_max_difference(diff_lists):
    """
    Determines the number of iterations before the maximum difference is reached for each key.
    """
    counts = {}
    for key in diff_lists:
        max_value = max(diff_lists[key])
        count = diff_lists[key].index(max_value) + 1
        counts[key] = count
    return counts

def get_atom_type_prefix(atom_type, A, B, R, M, X):
    """
    Maps an atom type to its corresponding prefix.
    """
    atom_type_map = {
        R: 'R', M: 'M', X: 'X',
        A: 'A', B: 'B'
    }
    
    return atom_type_map.get(atom_type, '')


def get_combined_prefix(first_label, second_label, A, B, R, M, X):
    """
    Combines the prefixes of two atom labels.
    """

    first_prefix = get_atom_type_prefix(get_atom_type(first_label), A, B, R, M, X)
    second_prefix = get_atom_type_prefix(get_atom_type(second_label),A, B, R, M, X)

    return f"{first_prefix}_{second_prefix}"


def get_pair_info_and_radius_sums(label, data_dict, rad_sum, A, B, R, M, X):
    """
    Retrieves pair information and sum of radii for a given atom label.
    """
    pair_info = data_dict["pair_info"][:19]  # First 20 points
    shortest_dist = pair_info[0][2]
    pairs_and_radius_sums = []

    for pair in pair_info:
        second_pair_label = pair[1][1]
        prefix = get_combined_prefix(label, second_pair_label, A, B, R, M, X)

        pairs_and_radius_sums.append({
            'pair': pair,
            'CIF_rad_sum': rad_sum['CIF_rad_sum'].get(prefix, 0.0),
            'CIF_rad_refined_sum': rad_sum['CIF_rad_refined_sum'].get(prefix, 0.0),
            'Pauling_rad_sum': rad_sum['Pauling_rad_sum'].get(prefix, 0.0)
        })

    return shortest_dist, pairs_and_radius_sums


def calculate_diff_counts_per_label(unique_labels, output_dict, rad_sum, A, B, R, M, X):
    """
    Calculates difference counts per atom label.
    """
    counts = {}

    for label in unique_labels:
        diff_lists = {
            "dist_by_shortest_dist": [],
            "dist_by_CIF_rad_sum": [],
            "dist_by_CIF_rad_refined_sum": [],
            "dist_by_Pauling_rad_sum": []
        }
        prev_dist_sums = {}
        first_iteration = True

        if label in output_dict:
            shortest_dist, pairs_and_radius_sums = get_pair_info_and_radius_sums(label, output_dict[label], rad_sum, A, B, R, M, X)

            for pair_info in pairs_and_radius_sums:
                dist_sums = {
                    'dist_by_shortest_dist': pair_info['pair'][2] / shortest_dist,
                    'dist_by_CIF_rad_sum': pair_info['pair'][2] / pair_info['CIF_rad_sum'],
                    'dist_by_CIF_rad_refined_sum': pair_info['pair'][2] / pair_info['CIF_rad_refined_sum'],
                    'dist_by_Pauling_rad_sum': pair_info['pair'][2] / pair_info['Pauling_rad_sum']
                }

                prev_dist_sums = update_difference_lists(pair_info['pair'][2], dist_sums, prev_dist_sums, diff_lists, first_iteration)

                first_iteration = False

            counts[label] = get_counts_before_max_difference(diff_lists)

    return counts


def get_pair_info_for_label(label, atom_pair_info_dict):
    """
    Returns atomic pair information for a given label.
    """
    return atom_pair_info_dict[label]["pair_info"] if label in atom_pair_info_dict else None


def find_most_common_point(pair_info, counts, label, dist_type):
    """
    Determines the most common point based on pair information.
    """
    ith_point_number = [pair[0][0] for pair in pair_info[:counts[label][dist_type]]]

    return Counter(ith_point_number).most_common(1)[0][0]


def convert_to_cartesian(pair, cell_lengths, cell_angles_rad):
    """
    Converts atomic pair coordinates from fractional to cartesian.
    """
    cart_coords_ith = get_cartesian_from_fractional(pair[3][0], cell_lengths, cell_angles_rad)
    cart_coords_jth = get_cartesian_from_fractional(pair[3][1], cell_lengths, cell_angles_rad)

    return cart_coords_ith, cart_coords_jth


def process_atom_pairs(isBinary, pair_info, cell_lengths, cell_angles_rad, most_common_point, A, B, R, M, X, counts, label, dist_type):
    """
    Processes atomic pairs to gather information like polyhedron points, central atom coordinates, 
    and counts of atom types in coordination.
    """
    polyhedron_points = []
    central_atom_coord = None
    central_atom_label = ''
    jth_atoms_labels = []

    atom_counts = Counter()

    for pair in supercell.swap_pairs(pair_info, counts, label, dist_type, most_common_point):
        cart_coords_ith, cart_coords_jth = convert_to_cartesian(pair, cell_lengths, cell_angles_rad)
        polyhedron_points.append(cart_coords_jth)
        central_atom_coord = cart_coords_ith
        central_atom_label = pair[1][0]
        jth_atoms_labels.append(pair[1][1])
        
        if isBinary:
            if pair[1][1].startswith(A):
                atom_counts['A'] += 1
            elif pair[1][1].startswith(B):
                atom_counts['B'] += 1
        else:
            if pair[1][1].startswith(R):
                atom_counts['R'] += 1
            elif pair[1][1].startswith(M):
                atom_counts['M'] += 1
            elif pair[1][1].startswith(X):
                atom_counts['X'] += 1

    if isBinary:
        return polyhedron_points, central_atom_coord, central_atom_label, jth_atoms_labels, atom_counts['A'], atom_counts['B']
    else:
        return polyhedron_points, central_atom_coord, central_atom_label, jth_atoms_labels, atom_counts['R'], atom_counts['M'], atom_counts['X']


def process_labels(label, atom_pair_info_dict, counts, cell_lengths, cell_angles_rad, atom_labels, rad_sum, CIF_id, formula_string, isTernary=False):
    """
    Processes the given label to extract relevant data and generate a DataFrame.
    """
    
    pair_info = get_pair_info_for_label(label, atom_pair_info_dict)
    
    dfs = []  # This will store dataframes for each dist_type
    
    if pair_info and label in counts:
        for dist_type in counts[label]:
            most_common_point = find_most_common_point(pair_info, counts, label, dist_type)

            pair_params = {
                'isBinary': not isTernary,
                'pair_info': pair_info,
                'cell_lengths': cell_lengths,
                'cell_angles_rad': cell_angles_rad,
                'most_common_point': most_common_point
            }

            result = process_atom_pairs(
                **pair_params, 
                **atom_labels, 
                counts=counts, 
                label=label, 
                dist_type=dist_type
            )

            if isTernary:
                polyhedron_points, central_atom_coord, central_atom_label, jth_atoms_labels, R_atom_count_in_CN, M_atom_count_in_CN, X_atom_count_in_CN = result
                atom_counts = [R_atom_count_in_CN, M_atom_count_in_CN, X_atom_count_in_CN]
                atoms = ['R', 'M', 'X']
            else:
                polyhedron_points, central_atom_coord, central_atom_label, jth_atoms_labels, A_atom_count_in_CN, B_atom_count_in_CN = result
                atom_counts = [A_atom_count_in_CN, B_atom_count_in_CN]
                atoms = ['A', 'B']

            if len(jth_atoms_labels) < 4:
                print(f"\nSkipping... {central_atom_label} has lower C.N. {len(jth_atoms_labels)}\n")
                continue

            try:
                hull = ConvexHull(polyhedron_points)
            except:
                print(f"\nError in determining polyhedron for {central_atom_label}.\n")
                continue
            polyhedron_points = np.array(polyhedron_points)
            polyhedron_metrics = compute_polyhedron_metrics(polyhedron_points, central_atom_coord, hull)

            df = cn_dataframe.get_coordniate_number_df(polyhedron_metrics, atom_counts, atoms, formula_string, label, dist_type, CIF_id)
            dfs.append(df)
    
    if not dfs == []:
        return pd.concat(dfs, ignore_index=True).round(5)

    return pd.DataFrame()
