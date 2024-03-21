import pandas as pd
import featurizer.distance as distance
import preprocess.optimize as optimize
import util.data as db
import preprocess.cif_parser as cif_parser
import featurizer.coordinate_number as cn_featurizer
import os
import time

def get_coordinate_number_binary_df(isBinary,
                                    coordinate_number_binary_df,
                                    unique_atoms_tuple,
                                    unique_labels,
                                    atomic_pair_list,
                                    atom_pair_info_dict,
                                    CIF_data,
                                    radii_data):
    
    CIF_id, cell_lengths, cell_angles_rad, CIF_loop_values, formula_string = CIF_data

    A, B = unique_atoms_tuple[0][0], unique_atoms_tuple[1][0]
    R = M = X = ''

    atoms = [A, B]
    atoms_for_radii = [unique_atoms_tuple[0][0], unique_atoms_tuple[1][0]]  # [A, B]
    atom_radii = db.get_atom_radii(atoms_for_radii, radii_data)
    A_CIF_rad, A_Pauling_rad = atom_radii[A]["CIF"], atom_radii[A]["Pauling"]
    B_CIF_rad, B_Pauling_rad = atom_radii[B]["CIF"], atom_radii[B]["Pauling"]

    shortest_AA, shortest_BB, shortest_AB = distance.find_shortest_pair_distances(isBinary, unique_atoms_tuple, atomic_pair_list)
    shortest_distances_pair = {"AA": shortest_AA, "BB": shortest_BB, "AB": shortest_AB}
    A_CIF_rad_refined, B_CIF_rad_refined = optimize.optimize_CIF_rad_binary(A_CIF_rad, B_CIF_rad, shortest_distances_pair)
    rad_sum_binary = db.compute_rad_sum_binary(A_CIF_rad, B_CIF_rad,
                                            A_CIF_rad_refined, B_CIF_rad_refined,
                                            A_Pauling_rad, B_Pauling_rad)

    # Insert values for unique_labels, output_dict
    CN_counts = cn_featurizer.calculate_diff_counts_per_label(unique_labels, atom_pair_info_dict, rad_sum_binary, A, B, R, M, X)
    atom_labels = {'A': A, 'B': B, 'R': R, 'M': M, 'X': X}

    for label in unique_labels:
        df = cn_featurizer.process_labels(label, atom_pair_info_dict, CN_counts, cell_lengths, cell_angles_rad, atom_labels, rad_sum_binary, CIF_id, formula_string)
        coordinate_number_binary_df = pd.concat([coordinate_number_binary_df, df], ignore_index=True)
    
    return coordinate_number_binary_df




def get_coordinate_number_ternary_df(isBinary,
                                    coordinate_number_ternary_df,
                                    unique_atoms_tuple,
                                    unique_labels,
                                    atomic_pair_list,
                                    atom_pair_info_dict,
                                    CIF_data,
                                    radii_data):
    
    
    R, M, X  = unique_atoms_tuple[0][0], unique_atoms_tuple[1][0], unique_atoms_tuple[2][0]
    A = B = ''

    CIF_id, cell_lengths, cell_angles_rad, CIF_loop_values, formula_string = CIF_data

    atoms = [R, M, X]
    atom_radii = db.get_atom_radii(atoms, radii_data)
    R_CIF_rad, R_Pauling_rad = atom_radii[R]["CIF"], atom_radii[R]["Pauling"]
    M_CIF_rad, M_Pauling_rad = atom_radii[M]["CIF"], atom_radii[M]["Pauling"]
    X_CIF_rad, X_Pauling_rad = atom_radii[X]["CIF"], atom_radii[X]["Pauling"]

    # Initialize the shortest distances with a large number
    shortest_RR, shortest_MM, shortest_XX, shortest_RM, shortest_MX, shortest_RX = distance.find_shortest_pair_distances(isBinary, unique_atoms_tuple, atomic_pair_list)

    # Put distances into a dictionary
    shortest_distances_pair = {
        "RR": shortest_RR, "MM": shortest_MM, "XX": shortest_XX,
        "RM": shortest_RM, "MX": shortest_MX, "RX": shortest_RM
    }

    R_CIF_rad_refined, M_CIF_rad_refined, X_CIF_rad_refined = optimize.optimize_CIF_rad_ternary(R_CIF_rad, M_CIF_rad, X_CIF_rad, shortest_distances_pair)
    rad_sum_ternary = db.compute_rad_sum_ternary(
                                    R_CIF_rad, M_CIF_rad, X_CIF_rad,
                                    R_CIF_rad_refined, M_CIF_rad_refined, X_CIF_rad_refined,
                                    R_Pauling_rad, M_Pauling_rad, X_Pauling_rad
                                    )

    # Insert values for unique_labels, output_dict
    CN_counts = cn_featurizer.calculate_diff_counts_per_label(unique_labels, atom_pair_info_dict, rad_sum_ternary, A, B, R, M, X)
    atom_labels = {'A': A, 'B': B, 'R': R, 'M': M, 'X': X}

    for label in unique_labels:
        df = cn_featurizer.process_labels(label, atom_pair_info_dict, CN_counts, cell_lengths, cell_angles_rad, atom_labels, rad_sum_ternary, CIF_id, formula_string, isTernary=True)
        coordinate_number_ternary_df = pd.concat([coordinate_number_ternary_df, df], ignore_index=True)
        coordinate_number_ternary_df = coordinate_number_ternary_df.round(5)

    return coordinate_number_ternary_df


def get_coordniate_number_df(metrics, atom_counts, atom_labels, formula_string, label, dist_type, CIF_id):
    """
    Generates a DataFrame containing information derived from metrics, atom_counts, and other provided arguments.
    """
    data = {
        "CIF_id": [CIF_id],
        "Compound": [formula_string],
        "Central atom": [label],
        "CN method": [dist_type],
        "Coordination number": [metrics["number_of_vertices"]]
    }
    
    # Insert atom counts after CN_number
    for atom, count in zip(atom_labels, atom_counts):
        data[f"{atom} atom count in CN"] = [count]

    # Append the remaining columns
    data.update({
        "Volume of polyhedron": [metrics["Volume_of_polyhedron"]],
        "Distance from central atom to center of mass": [metrics["distance_to_center"]],
        "Number of edges": [metrics["number_of_edges"]],
        "Number of faces": [metrics["number_of_faces"]],
        "Shortest distance to center of any face": [metrics["shortest_distance_to_face"]],
        "Shortest distance to center of any edge": [metrics["shortest_distance_to_edge"]],
        "Volume of the inscribed sphere": [metrics["volume_of_inscribed_sphere"]],
        "Packing efficiency of inscribed sphere in polyhedron": [metrics["packing_efficiency"]]
    })

    return pd.DataFrame(data)


def update_log_dataframe(log_df, CIF_id, filename, formula_string, all_points, execution_time, running_total_time):
    """
    Updates the log DataFrame with the provided data.
    """    
    log_data = {
        "Filename": [os.path.basename(filename)],
        "Compound": [formula_string],
        "# of unique atoms after symmetry operations": [len(all_points)],
        "Execution Time (s)": [execution_time],
        "Total Execution Time (s)": [running_total_time]
    }

    df = pd.DataFrame(log_data)
    log_df = pd.concat([log_df, df], ignore_index=True)
    log_df = log_df.round(4)

    return log_df, log_data



def calculate_execution_time(start_time, running_total_time):
    """
    Calculatse execution time based on the provided start time and update the running total time.
    """
    end_time = time.time()
    execution_time = end_time - start_time
    running_total_time += execution_time
    
    return execution_time, running_total_time

