import featurizer.environment_binary as env_featurizer_binary
import featurizer.environment_ternary as env_featurizer_ternary
import util.log as log
import pandas as pd

def get_env_binary_df(
    atomic_environment_binary_df,
    unique_atoms_tuple,
    unique_labels,
    unique_shortest_labels,
    atom_pairs_info_dict,
    atom_counts,
    CIF_data):
        
    CIF_id, _, _, _, formula_string = CIF_data
    A, B = unique_atoms_tuple[0][0], unique_atoms_tuple[1][0]

    # 1st, 2nd binary features - # shortest distance count
    binary_distance_result = env_featurizer_binary.get_shortest_distances_count(unique_atoms_tuple,
                                                                            unique_labels,
                                                                            unique_shortest_labels,
                                                                            atom_pairs_info_dict,
                                                                            atom_counts)
    

    A_shortest_dist_count, B_shortest_dist_count, A_avg_shortest_dist_count, B_avg_shortest_dist_count = binary_distance_result

    # 3rd, 4th binary features - # of shortest distances within 5% tolerance from the shortest
    binary_shortest_dist_tol_res = env_featurizer_binary.get_shortest_dist_count_with_tol(unique_atoms_tuple,
                                                                            unique_labels,
                                                                            unique_shortest_labels,
                                                                            atom_pairs_info_dict,
                                                                            atom_counts)
    
    A_shortest_tol_dist_count, B_shortest_tol_dist_count, A_avg_shortest_dist_tol_count, B_avg_shortest_dist_tol_count = binary_shortest_dist_tol_res

    # 5th, 6th binary features - 2nd shortest distance / shortest distance
    second_by_first_shortest_dist_res = env_featurizer_binary.get_second_by_first_shortest_dist(unique_atoms_tuple,
                                                                            unique_labels,
                                                                            unique_shortest_labels,
                                                                            atom_pairs_info_dict,
                                                                            atom_counts)
    
    A_second_by_first_shortest_dist, B_second_by_first_shortest_dist, A_avg_second_by_first_shortest_dist, B_avg_second_by_first_shortest_dist = second_by_first_shortest_dist_res

    # 7th, 8th binary features - atom counts at the 2nd shortest distance
    second_shortest_dist_count_result = env_featurizer_binary.get_second_shortest_dist_count(unique_atoms_tuple,
                                                                            unique_labels,
                                                                            unique_shortest_labels,
                                                                            atom_pairs_info_dict,
                                                                            atom_counts)
    
    A_second_shortest_dist_count, B_second_shortest_dist_count, A_avg_second_shortest_dist_count, B_avg_second_shortest_dist_count = second_shortest_dist_count_result
    
    # 9th, 10th binary features - homoatomic shortest distance / shortest distance
    homoatomic_dist_by_shortest_dist_count_result = env_featurizer_binary.get_homoatomic_dist_by_shortest_dist_count(unique_atoms_tuple,
                                                                unique_labels,
                                                                unique_shortest_labels,
                                                                atom_pairs_info_dict,
                                                                atom_counts)

    A_homoatomic_dist_by_shortest_dist, B_homoatomic_dist_by_shortest_dist, A_avg_homoatomic_dist_by_shortest_dist, B_avg_homoatomic_dist_by_shortest_dist = homoatomic_dist_by_shortest_dist_count_result

    # 11th, 12th binary features - A, B count at the shortest distance
    A_B_count_at_shortest_dist_result, A_B_count_at_shortest_dist_avg_result = env_featurizer_binary.get_A_B_count_at_shortest_dist(unique_atoms_tuple,
                                                                unique_labels,
                                                                unique_shortest_labels,
                                                                atom_pairs_info_dict,
                                                                atom_counts)

    A_count_at_A_shortest_dist, B_count_at_A_shortest_dist, A_count_at_B_shortest_dist, B_count_at_B_shortest_dist = A_B_count_at_shortest_dist_result
    A_avg_new_count_at_A_shortest_dist, B_avg_new_count_at_A_shortest_dist, A_avg_new_count_at_B_shortest_dist, B_avg_new_count_at_B_shortest_dist = A_B_count_at_shortest_dist_avg_result
    
    atomic_environment_binary_data = {
        "CIF_id": [CIF_id],
        "Compound": [formula_string],
        "A": [A],
        "B": [B],
        "A_shortest_dist_count": [A_shortest_dist_count],
        "B_shortest_dist_count": [B_shortest_dist_count],
        "A_avg_shortest_dist_count": [A_avg_shortest_dist_count],
        "B_avg_shortest_dist_count": [B_avg_shortest_dist_count],

        "A_shortest_tol_dist_count": [A_shortest_tol_dist_count],
        "B_shortest_tol_dist_count": [B_shortest_tol_dist_count],

        "A_avg_shortest_dist_within_tol_count": [A_avg_shortest_dist_tol_count],
        "B_avg_shortest_dist_within_tol_count": [B_avg_shortest_dist_tol_count],

        "A_second_by_first_shortest_dist": [A_second_by_first_shortest_dist],
        "B_second_by_first_shortest_dist": [B_second_by_first_shortest_dist],

        "A_avg_second_by_first_shortest_dist": [A_avg_second_by_first_shortest_dist],
        "B_avg_second_by_first_shortest_dist": [B_avg_second_by_first_shortest_dist],

        "A_second_shortest_dist_count": [A_second_shortest_dist_count],
        "B_second_shortest_dist_count": [B_second_shortest_dist_count],

        "A_avg_second_shortest_dist_count": [A_avg_second_shortest_dist_count],
        "B_avg_second_shortest_dist_count": [B_avg_second_shortest_dist_count],

        "A_homoatomic_dist_by_shortest_dist": [A_homoatomic_dist_by_shortest_dist], 
        "B_homoatomic_dist_by_shortest_dist": [B_homoatomic_dist_by_shortest_dist],

        "A_avg_homoatomic_dist_by_shortest_dist": [A_avg_homoatomic_dist_by_shortest_dist], 
        "B_avg_homoatomic_dist_by_shortest_dist": [B_avg_homoatomic_dist_by_shortest_dist],

        "A_count_at_A_shortest_dist": [A_count_at_A_shortest_dist],
        "A_count_at_B_shortest_dist": [A_count_at_B_shortest_dist],
        
        "A_avg_count_at_A_shortest_dist": [A_avg_new_count_at_A_shortest_dist],
        "A_avg_count_at_B_shortest_dist": [A_avg_new_count_at_B_shortest_dist],
        
        "B_count_at_A_shortest_dist": [B_count_at_A_shortest_dist],
        "B_count_at_B_shortest_dist": [B_count_at_B_shortest_dist],

        "B_avg_count_at_A_shortest_dist": [B_avg_new_count_at_A_shortest_dist],
        "B_avg_count_at_B_shortest_dist": [B_avg_new_count_at_B_shortest_dist],
    }


    log.print_json_pretty("atomic_environment_binary_data", atomic_environment_binary_data)
    df = pd.DataFrame(atomic_environment_binary_data)     
    atomic_environment_binary_df = pd.concat([atomic_environment_binary_df, df], ignore_index=True)
    atomic_environment_binary_df = atomic_environment_binary_df.round(5)

    return atomic_environment_binary_df


def get_env_ternary_df(
    atomic_env_ternary_df,
    unique_atoms_tuple,
    unique_labels,
    unique_shortest_labels,
    atom_pairs_info_dict,
    atom_counts,
    CIF_data):

    CIF_id, _, _, _, formula_string = CIF_data
    R, M, X = unique_atoms_tuple[0][0], unique_atoms_tuple[1][0], unique_atoms_tuple[2][0]

    # 1st, 2nd ternary features - shortest distance count
    ternary_dist_res = env_featurizer_ternary.get_shortest_distances_count(unique_atoms_tuple,
                                                                unique_labels,
                                                                unique_shortest_labels,
                                                                atom_pairs_info_dict,
                                                                atom_counts)
    
    shortest_dist_count_res, avg_shortest_dist_count_res = ternary_dist_res
    R_shortest_dist_count, M_shortest_dist_count, X_shortest_dist_count = shortest_dist_count_res
    R_avg_shortest_dist_count, M_avg_shortest_dist_count, X_avg_shortest_dist_count = avg_shortest_dist_count_res


    # 3rd, 4th ternary features - # of shortest distances within 5% tolerance from the shortest
    shortest_tol_dist_count_res, avg_shortest_tol_dist_count_res = env_featurizer_ternary.get_shortest_dist_count_with_tol(unique_atoms_tuple,
                                                                            unique_labels,
                                                                            unique_shortest_labels,
                                                                            atom_pairs_info_dict,
                                                                            atom_counts)
    
    R_shortest_tol_dist_count, M_shortest_tol_dist_count, X_shortest_tol_dist_count = shortest_tol_dist_count_res
    R_avg_shortest_dist_within_tol_count, M_avg_shortest_dist_within_tol_count, X_avg_shortest_dist_within_tol_count = avg_shortest_tol_dist_count_res


    # 5th, 6th ternary features - 2nd shortest distance / shortest distance
    second_by_first_shortest_dist_res, avg_second_by_first_shortest_dist_res = env_featurizer_ternary.get_second_by_first_shortest_dist(unique_atoms_tuple,
                                                                            unique_labels,
                                                                            unique_shortest_labels,
                                                                            atom_pairs_info_dict,
                                                                            atom_counts)


    R_second_by_first_shortest_dist, M_second_by_first_shortest_dist, X_second_by_first_shortest_dist = second_by_first_shortest_dist_res
    R_avg_second_by_first_shortest_dist, M_avg_second_by_first_shortest_dist, X_avg_second_by_first_shortest_dist = avg_second_by_first_shortest_dist_res


    # 7th, 8th ternary features - atom counts at the 2nd shortest distance
    second_shortest_dist_count_res, avg_second_shortest_dist_count_res = env_featurizer_ternary.get_second_shortest_dist_count(
        unique_atoms_tuple,
        unique_labels,
        unique_shortest_labels,
        atom_pairs_info_dict,
        atom_counts)
    
    R_second_shortest_dist_count, M_second_shortest_dist_count, X_second_shortest_dist_count = second_shortest_dist_count_res
    R_avg_second_shortest_dist_count, M_avg_second_shortest_dist_count, X_avg_second_shortest_dist_count = avg_second_shortest_dist_count_res

    
    # 9th, 10th ternary features - homoatomic shortest distance / shortest distance
    homoatomic_dist_by_shortest_dist_res, avg_homoatomic_dist_by_shortest_dist_res = env_featurizer_ternary.get_homoatomic_dist_by_shortest_dist_countd(
        unique_atoms_tuple,
        unique_labels,
        unique_shortest_labels,
        atom_pairs_info_dict,
        atom_counts)
    
    R_homoatomic_dist_by_shortest_dist, M_homoatomic_dist_by_shortest_dist, X_homoatomic_dist_by_shortest_dist = homoatomic_dist_by_shortest_dist_res
    R_avg_homoatomic_dist_by_shortest_dist, M_avg_homoatomic_dist_by_shortest_dist, X_avg_homoatomic_dist_by_shortest_dist = avg_homoatomic_dist_by_shortest_dist_res


    # 11th, 12th ternary features - A, B count at the shortest distance
    R_X_X_count_at_shortest_dist_res = env_featurizer_ternary.get_R_X_X_count_at_shortest_dist(
        unique_atoms_tuple,
        unique_labels,
        unique_shortest_labels,
        atom_pairs_info_dict,
        atom_counts)
    
    count_at_R_shortest_dist_res, count_at_M_shortest_dist_res, count_at_X_shortest_dist_res, avg_count_at_R_shortest_dist_res, avg_count_at_M_shortest_dist_res, avg_count_at_X_shortest_dist_res = R_X_X_count_at_shortest_dist_res

    R_count_at_R_shortest_dist, M_count_at_R_shortest_dist, X_count_at_R_shortest_dist = count_at_R_shortest_dist_res
    R_count_at_M_shortest_dist, M_count_at_M_shortest_dist, X_count_at_M_shortest_dist = count_at_M_shortest_dist_res
    R_count_at_X_shortest_dist, M_count_at_X_shortest_dist, X_count_at_X_shortest_dist = count_at_X_shortest_dist_res
    R_avg_new_count_at_R_shortest_dist, M_avg_new_count_at_R_shortest_dist, X_avg_new_count_at_R_shortest_dist = avg_count_at_R_shortest_dist_res
    R_avg_new_count_at_M_shortest_dist, M_avg_new_count_at_M_shortest_dist, X_avg_new_count_at_M_shortest_dist = avg_count_at_M_shortest_dist_res
    R_avg_new_count_at_X_shortest_dist, M_avg_new_count_at_X_shortest_dist, X_avg_new_count_at_X_shortest_dist = avg_count_at_X_shortest_dist_res



    atomic_environment_ternary_data = {
        "CIF_id": [CIF_id],
        "Compound": [formula_string],
        "R": [R],
        "M": [M],
        "X": [X],
        "R_shortest_dist_count": R_shortest_dist_count,
        "M_shortest_dist_count": M_shortest_dist_count,
        "X_shortest_dist_count": X_shortest_dist_count,

        "R_avg_shortest_dist_count": R_avg_shortest_dist_count,
        "M_avg_shortest_dist_count": M_avg_shortest_dist_count,
        "X_avg_shortest_dist_count": X_avg_shortest_dist_count,

        "R_shortest_tol_dist_count": R_shortest_tol_dist_count,
        "M_shortest_tol_dist_count": M_shortest_tol_dist_count,
        "X_shortest_tol_dist_count": X_shortest_tol_dist_count,

        "R_avg_shortest_dist_within_tol_count": R_avg_shortest_dist_within_tol_count,
        "M_avg_shortest_dist_within_tol_count": M_avg_shortest_dist_within_tol_count,
        "X_avg_shortest_dist_within_tol_count": X_avg_shortest_dist_within_tol_count,

        "R_second_by_first_shortest_dist": R_second_by_first_shortest_dist,
        "M_second_by_first_shortest_dist": M_second_by_first_shortest_dist,
        "X_second_by_first_shortest_dist": X_second_by_first_shortest_dist,

        "R_avg_second_by_first_shortest_dist": R_avg_second_by_first_shortest_dist,
        "M_avg_second_by_first_shortest_dist": M_avg_second_by_first_shortest_dist,
        "X_avg_second_by_first_shortest_dist": X_avg_second_by_first_shortest_dist,

        "R_second_shortest_dist_count": R_second_shortest_dist_count,
        "M_second_shortest_dist_count": M_second_shortest_dist_count,
        "X_second_shortest_dist_count": X_second_shortest_dist_count,

        "R_avg_second_shortest_dist_count": R_avg_second_shortest_dist_count,
        "M_avg_second_shortest_dist_count": M_avg_second_shortest_dist_count,
        "X_avg_second_shortest_dist_count": X_avg_second_shortest_dist_count,

        "R_homoatomic_dist_by_shortest_dist": R_homoatomic_dist_by_shortest_dist, 
        "M_homoatomic_dist_by_shortest_dist": M_homoatomic_dist_by_shortest_dist, 
        "X_homoatomic_dist_by_shortest_dist": X_homoatomic_dist_by_shortest_dist,

        "R_avg_homoatomic_dist_by_shortest_dist": R_avg_homoatomic_dist_by_shortest_dist,
        "M_avg_homoatomic_dist_by_shortest_dist": M_avg_homoatomic_dist_by_shortest_dist,
        "X_avg_homoatomic_dist_by_shortest_dist": X_avg_homoatomic_dist_by_shortest_dist,

        "R_count_at_R_shortest_dist": R_count_at_R_shortest_dist,
        "R_count_at_M_shortest_dist": R_count_at_M_shortest_dist,
        "R_count_at_X_shortest_dist": R_count_at_X_shortest_dist,

        "R_avg_count_at_R_shortest_dist": R_avg_new_count_at_R_shortest_dist,
        "R_avg_count_at_M_shortest_dist": R_avg_new_count_at_M_shortest_dist,
        "R_avg_count_at_X_shortest_dist": R_avg_new_count_at_X_shortest_dist,

        "M_count_at_R_shortest_dist": M_count_at_R_shortest_dist,
        "M_count_at_M_shortest_dist": M_count_at_M_shortest_dist,
        "M_count_at_X_shortest_dist": M_count_at_X_shortest_dist,

        "M_avg_count_at_R_shortest_dist": M_avg_new_count_at_R_shortest_dist,
        "M_avg_count_at_M_shortest_dist": M_avg_new_count_at_M_shortest_dist,
        "M_avg_count_at_X_shortest_dist": M_avg_new_count_at_X_shortest_dist,

        "X_count_at_R_shortest_dist": X_count_at_R_shortest_dist,
        "X_count_at_M_shortest_dist": X_count_at_M_shortest_dist,
        "X_count_at_X_shortest_dist": X_count_at_X_shortest_dist,

        "X_avg_count_at_R_shortest_dist": X_avg_new_count_at_R_shortest_dist,
        "X_avg_count_at_M_shortest_dist": X_avg_new_count_at_M_shortest_dist,
        "X_avg_count_at_X_shortest_dist": X_avg_new_count_at_X_shortest_dist,
    }

    
    # Create DataFrame
    df = pd.DataFrame(atomic_environment_ternary_data)
    atomic_env_ternary_df = pd.concat([atomic_env_ternary_df, df], ignore_index=True)
    atomic_env_ternary_df = atomic_env_ternary_df.round(5)
    log.print_json_pretty("atomic_environment_ternary_data", atomic_environment_ternary_data)

    return atomic_env_ternary_df