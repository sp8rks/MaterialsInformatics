import click
import os
import glob
import pandas as pd
import time
import math
from click import style
import util.data as data
import preprocess.optimize as optimize
import featurizer.distance as distance
import preprocess.cif_parser as cif_parser
import featurizer.coordinate_number as cn_featurizer
import featurizer.interatomic as interatomic_featurizer
import util.folder as folder
import preprocess.supercell as supercell
import util.unitcell as unitcell
import util.log as log


def get_interatomic_binary_df(filename,
                        interatomic_binary_df,
                        interatomic_universal_df,
                        all_points,
                        unique_atoms_tuple,
                        atomic_pair_list,
                        CIF_data,
                        radii_data):
    
    CIF_id, cell_lengths, cell_angles_rad, CIF_loop_values, formula_string = CIF_data

    # Intialize variables used for universal features
    shortest_homoatomic_distance_by_2_by_atom_size = 0.0
    shortest_heteroatomic_distance_by_sum_of_atom_sizes = 0.0
    shortest_homoatomic_distance_by_2_by_refined_atom_sizes = 0.0
    shortest_heteroatomic_distance_by_refined_atom_sizes = 0.0
    highest_refined_percent_diff = 0.0
    lowest_refined_percent_diff = 0.0
    packing_efficiency = 0.0


    A, B = unique_atoms_tuple[0][0], unique_atoms_tuple[1][0]
    atoms = [A, B]
    atoms_for_radii = [unique_atoms_tuple[0][0], unique_atoms_tuple[1][0]]  # [A, B]
    atom_radii = data.get_atom_radii(atoms_for_radii, radii_data)
    A_CIF_rad = atom_radii[A]["CIF"]
    B_CIF_rad = atom_radii[B]["CIF"]

    shortest_AA, shortest_BB, shortest_AB = distance.find_shortest_pair_distances(True, unique_atoms_tuple, atomic_pair_list)
    shortest_distances_pair = {"AA": shortest_AA, "BB": shortest_BB, "AB": shortest_AB}
    radii, obj_value = optimize.optimize_CIF_rad_binary(A_CIF_rad, B_CIF_rad, shortest_distances_pair, True)
    A_CIF_rad_refined, B_CIF_rad_refined = radii
    
    interatomic_distA_A = shortest_AA
    interatomic_distB_B = shortest_BB
    interatomic_distA_B = shortest_AB

    interatomic_distA_A_rad = shortest_AA/2
    interatomic_distB_B_rad = shortest_BB/2
    interatomic_distA_B_rad = shortest_AB/2

    A_A_radius_sum = 2 * A_CIF_rad_refined
    B_B_radius_sum = 2 * B_CIF_rad_refined
    A_B_radius_sum = A_CIF_rad_refined + B_CIF_rad_refined

    diff_A = A_CIF_rad_refined - A_CIF_rad
    diff_B = B_CIF_rad_refined - B_CIF_rad

    percent_diff_A = (A_CIF_rad_refined - A_CIF_rad)/ A_CIF_rad
    percent_diff_B = (B_CIF_rad_refined - B_CIF_rad)/ B_CIF_rad

    # Interatomic distance features
    Asize_by_Bsize = A_CIF_rad/B_CIF_rad
    distA_A_by_byAsize = interatomic_distA_A_rad/A_CIF_rad
    distB_B_by_byBsize = interatomic_distB_B_rad/B_CIF_rad
    distA_B_by_byAsizebyBsize = interatomic_distA_B/ (A_CIF_rad + B_CIF_rad)

    interatomic_A_A_minus_ref_diff = ((interatomic_distA_A - A_A_radius_sum)/ interatomic_distA_A)
    interatomic_B_B_minus_ref_diff = ((interatomic_distB_B - B_B_radius_sum) / interatomic_distB_B)
    interatomic_A_B_minus_ref_diff = ((interatomic_distA_B - A_B_radius_sum) / interatomic_distA_B)
    
    atoms = (A, B)
    CIF_rad_refined_dict = {
        A: A_CIF_rad_refined,
        B: B_CIF_rad_refined
    }
    packing_efficiency = unitcell.compute_packing_efficiency(atoms, CIF_loop_values, CIF_rad_refined_dict, cell_lengths, cell_angles_rad)

    interatomic_binary_data = {
        "CIF_id": [CIF_id],
        "Compound": [formula_string],
        "A": [A],
        "B": [B],
        "distAA": [interatomic_distA_A],
        "distBB": [interatomic_distB_B],
        "distAB": [interatomic_distA_B],
        "Asize": [A_CIF_rad],
        "Bsize": [B_CIF_rad],
        "Asize_by_Bsize": [Asize_by_Bsize],
        "distAA_by2_byAsize": [distA_A_by_byAsize],
        "distBB_by2_byBsize": [distB_B_by_byBsize],
        "distAB_by2_byAsizebyBsize": [distA_B_by_byAsizebyBsize],
        "Asize_ref": [A_CIF_rad_refined],
        "Bsize_ref": [B_CIF_rad_refined],
        "percent_diff_A_by_100": [percent_diff_A],
        "percent_diff_B_by_100": [percent_diff_B],
        "distAA_minus_ref_diff": [interatomic_A_A_minus_ref_diff],
        "distBB_minus_ref_diff": [interatomic_B_B_minus_ref_diff],
        "distAB_minus_ref_diff": [interatomic_A_B_minus_ref_diff],
        "refined_packing_eff": [packing_efficiency],
        "R_factor": [obj_value]
    }


    df = pd.DataFrame(interatomic_binary_data)
    interatomic_binary_df = pd.concat([interatomic_binary_df, df], ignore_index=True)
    interatomic_binary_df = interatomic_binary_df.round(5)

    # log.print_json_pretty("interatomic_binary_data", interatomic_binary_data)
    
    # Binary - Universal
    # Get the shortest homo/heteroatomic distance
    homoatomic_distances = {key: shortest_distances_pair[key] for key in ["AA", "BB"]}
    heteroatomic_distance = shortest_distances_pair["AB"]
    shortest_homoatomic_distance = min(homoatomic_distances.values())
    shortest_heteroatomic_distance = heteroatomic_distance

    # Define the CIF and refined radii
    cif_radii = {"AA": A_CIF_rad, "BB": B_CIF_rad, "AB": (A_CIF_rad + B_CIF_rad)}
    refined_radii = {"AA": A_CIF_rad_refined, "BB": B_CIF_rad_refined, "AB": (A_CIF_rad_refined + B_CIF_rad_refined)}
    percent_diffs = [percent_diff_A, percent_diff_B]

    # Find key of shortest_homoatomic_distance in distances
    shortest_homo_key = [k for k, v in shortest_distances_pair.items() if v == shortest_homoatomic_distance][0]

    # Extract 9 universal features
    shortest_homoatomic_distance_by_2_by_atom_size = (shortest_homoatomic_distance / 2) / cif_radii[shortest_homo_key]
    shortest_heteroatomic_distance_by_sum_of_atom_sizes = shortest_heteroatomic_distance / cif_radii["AB"]
    shortest_homoatomic_distance_by_2_by_refined_atom_sizes = (shortest_homoatomic_distance / 2) / refined_radii[shortest_homo_key]
    shortest_heteroatomic_distance_by_refined_atom_sizes = shortest_heteroatomic_distance / refined_radii["AB"]
    highest_refined_percent_diff = max([abs(p) for p in percent_diffs])
    lowest_refined_percent_diff = min([abs(p) for p in percent_diffs])

    interatomic_universal_data = {
        "CIF_id": [CIF_id],
        "Compound": [formula_string],
        "Shortest homoatomic distance": [shortest_homoatomic_distance],
        "Shortest heteroatomic distance": [shortest_heteroatomic_distance],
        "Shortest homoatomic distance by 2 by atom size": [shortest_homoatomic_distance_by_2_by_atom_size],
        "Shortest heteroatomic distance by sum of atom sizes": [shortest_heteroatomic_distance_by_sum_of_atom_sizes],
        "Shortest homoatomic distance by 2 by refined atom size": [shortest_homoatomic_distance_by_2_by_refined_atom_sizes],
        "Shortest heteroatomic distance by sum of refined sizes": [shortest_heteroatomic_distance_by_refined_atom_sizes],
        "Highest refined percent difference by 100 (abs value)": [highest_refined_percent_diff],
        "Lowest refined percent difference by 100 (abs value)": [lowest_refined_percent_diff],
        "Packing efficiency": [packing_efficiency]
    }

    df = pd.DataFrame(interatomic_universal_data)
    interatomic_universal_df = pd.concat([interatomic_universal_df, df], ignore_index=True)
    interatomic_universal_df = interatomic_universal_df.round(5)

    return interatomic_binary_df, interatomic_universal_df



def get_interatomic_ternary_df(filename,
                         interatomic_ternary_df,
                        interatomic_universal_df,
                        all_points,
                        unique_atoms_tuple,
                        atomic_pair_list,
                        CIF_data,
                        radii_data):
    
    CIF_id, cell_lengths, cell_angles_rad, CIF_loop_values, formula_string = CIF_data
    R, M, X = unique_atoms_tuple[0][0], unique_atoms_tuple[1][0], unique_atoms_tuple[2][0]         
    atom_R_count = atom_M_count = atom_X_count = 0   
    atoms = [R, M, X]
    atom_radii = data.get_atom_radii(atoms, radii_data)
    R_CIF_rad, R_Pauling_rad = atom_radii[R]["CIF"], atom_radii[R]["Pauling"]
    M_CIF_rad, M_Pauling_rad = atom_radii[M]["CIF"], atom_radii[M]["Pauling"]
    X_CIF_rad, X_Pauling_rad = atom_radii[X]["CIF"], atom_radii[X]["Pauling"]

    # Initialize the shortest distances with a large number
    shortest_RR, shortest_MM, shortest_XX, shortest_RM, shortest_MX, shortest_RX = distance.find_shortest_pair_distances(False, unique_atoms_tuple, atomic_pair_list)

    # Put distances into a dictionary
    shortest_distances_pair = {
        "RR": shortest_RR, "MM": shortest_MM, "XX": shortest_XX,
        "RM": shortest_RM, "MX": shortest_MX, "RX": shortest_RM
    }
    
    radii, obj_value = optimize.optimize_CIF_rad_ternary(R_CIF_rad, M_CIF_rad, X_CIF_rad, shortest_distances_pair, True)
    R_CIF_rad_refined, M_CIF_rad_refined, X_CIF_rad_refined = radii

    atoms = (R, M, X)
    CIF_rad_refined_dict = {
        R: R_CIF_rad_refined,
        M: M_CIF_rad_refined,
        X: X_CIF_rad_refined
    }

    packing_efficiency = unitcell.compute_packing_efficiency(atoms, CIF_loop_values, CIF_rad_refined_dict, cell_lengths, cell_angles_rad)

    Rsize_by_Msize = R_CIF_rad/M_CIF_rad
    Msize_by_Xsize = M_CIF_rad/X_CIF_rad
    Rsize_by_Xsize = R_CIF_rad/X_CIF_rad

    interatomic_distR_R_rad = shortest_RR/2
    interatomic_distM_M_rad = shortest_MM/2
    interatomic_distX_X_rad = shortest_XX/2
    interatomic_distR_M_rad = shortest_RM/2
    interatomic_distM_X_rad = shortest_MX/2
    interatomic_distR_X_rad = shortest_RX/2

    #d(R-R)/2/R, d(M-M)/2/M, d(X-X)/2/X features
    distR_R_by2_byRsize = interatomic_distR_R_rad/R_CIF_rad
    distM_M_by2_byMsize = interatomic_distM_M_rad/M_CIF_rad
    distX_X_by2_byXsize = interatomic_distX_X_rad/X_CIF_rad

    #d(R-M)/(R+M), d(R-X)/(R+X), d(M-X)//(M+X) features
    distR_M_byRsizebyMsize = shortest_RM/(R_CIF_rad + M_CIF_rad)
    distM_X_byMsizebyXsize = shortest_MX/(M_CIF_rad + X_CIF_rad)
    distR_X_byRsizebyXsize = shortest_RX/(R_CIF_rad + X_CIF_rad)

    # Rrefined, Mrefined, Xrefined features
    R_R_radius_sum = 2 * R_CIF_rad_refined
    M_M_radius_sum = 2 * M_CIF_rad_refined
    X_X_radius_sum = 2 * X_CIF_rad_refined

    R_M_radius_sum = 2 * R_CIF_rad_refined + M_CIF_rad_refined
    M_X_radius_sum = 2 * M_CIF_rad_refined + X_CIF_rad_refined
    R_X_radius_sum = 2 * R_CIF_rad_refined + X_CIF_rad_refined

    diff_R = R_CIF_rad_refined - R_CIF_rad
    diff_M = M_CIF_rad_refined - M_CIF_rad
    diff_X = X_CIF_rad_refined - X_CIF_rad

    #R,M,X refinemnt difference percent features
    percent_diff_R = (R_CIF_rad_refined - R_CIF_rad)/ R_CIF_rad
    percent_diff_M = (M_CIF_rad_refined - M_CIF_rad)/ M_CIF_rad
    percent_diff_X = (X_CIF_rad_refined - X_CIF_rad)/ X_CIF_rad

    #(R-R, M-M, X-X, R-M, M-X, R-X interatomic distance) features
    interatomic_R_R_minus_ref_diff = ((shortest_RR - R_R_radius_sum) / shortest_RR)
    interatomic_M_M_minus_ref_diff = ((shortest_MM - M_M_radius_sum) / shortest_MM)
    interatomic_X_X_minus_ref_diff = ((shortest_XX - X_X_radius_sum) / shortest_XX)
    interatomic_R_M_minus_ref_diff = ((shortest_RM - R_M_radius_sum) / shortest_RM)
    interatomic_R_X_minus_ref_diff = ((shortest_RX - R_X_radius_sum) / shortest_RX)
    interatomic_M_X_minus_ref_diff = ((shortest_MX - M_X_radius_sum) / shortest_MX)

    interatomic_ternary_data = {
        "CIF_id": [CIF_id],
        "Compound": [formula_string],
        "R": [R],
        "M": [M],
        "X": [X],
        "distRR": [shortest_RR],
        "distMM": [shortest_MM],
        "distXX": [shortest_XX],
        "distRM": [shortest_RM],
        "distMX": [shortest_MX],
        "distRX": [shortest_RX],
        "Rsize": [R_CIF_rad],
        "Msize": [M_CIF_rad],
        "Xsize": [X_CIF_rad],
        "Rsize_by_Msize": [Rsize_by_Msize],
        "Msize_by_Xsize": [Msize_by_Xsize],
        "Rsize_by_Xsize": [Rsize_by_Xsize],
        "distRR_by2_byRsize": [distR_R_by2_byRsize],
        "distMM_by2_byMsize": [distM_M_by2_byMsize],
        "distXX_by2_byXsize": [distX_X_by2_byXsize],
        "distRM_byRsizebyMsize": [distR_M_byRsizebyMsize],
        "distMX_byMsizebyXsize": [distM_X_byMsizebyXsize],
        "distRX_byRsizebyXsize": [distR_X_byRsizebyXsize],
        "Rsize_ref": [R_CIF_rad_refined],
        "Msize_ref": [M_CIF_rad_refined],
        "Xsize_ref": [X_CIF_rad_refined],
        "percent_diff_R_by_100": [percent_diff_R],
        "percent_diff_M_by_100": [percent_diff_M],
        "percent_diff_X_by_100": [percent_diff_X],
        "distRR_minus_ref_diff": [interatomic_R_R_minus_ref_diff],
        "distMM_minus_ref_diff": [interatomic_M_M_minus_ref_diff],
        "distXX_minus_ref_diff": [interatomic_X_X_minus_ref_diff],
        "distRM_minus_ref_diff": [interatomic_R_M_minus_ref_diff],
        "distMX_minus_ref_diff": [interatomic_M_X_minus_ref_diff],
        "distRX_minus_ref_diff": [interatomic_R_X_minus_ref_diff],
        "refined_packing_eff": [packing_efficiency],
        "R_factor": [obj_value]
    }

    df = pd.DataFrame(interatomic_ternary_data)
    interatomic_ternary_df = pd.concat([interatomic_ternary_df, df], ignore_index=True)
    interatomic_ternary_df = interatomic_ternary_df.round(5)


    # Get the shortest homo/hetroatomic distance
    homoatomic_distances = {key: shortest_distances_pair[key] for key in ["RR", "MM", "XX"]}
    heteroatomic_distances = {key: shortest_distances_pair[key] for key in ["RM", "MX", "RX"]}
    shortest_homoatomic_distance = min(homoatomic_distances.values())
    shortest_heteroatomic_distance = min(heteroatomic_distances.values())

    # Parse the CIF radis based on the shortest homo/heteroatomic distance
    cif_radii = {"RR": R_CIF_rad, "MM": M_CIF_rad, "XX": X_CIF_rad,
                "RM": (R_CIF_rad + M_CIF_rad), "MX": (M_CIF_rad + X_CIF_rad), "RX": (R_CIF_rad + X_CIF_rad)}

    refined_radii = {"RR": R_CIF_rad_refined, "MM": M_CIF_rad_refined, "XX": X_CIF_rad_refined,
                    "RM": (R_CIF_rad_refined + M_CIF_rad_refined),
                    "MX": (M_CIF_rad_refined + X_CIF_rad_refined),
                    "RX": (R_CIF_rad_refined + X_CIF_rad_refined)}

    percent_diffs = [percent_diff_R, percent_diff_M, percent_diff_X]

    # Find key of shortest_homoatomic_distance and shortest_heteroatomic_distance in distances
    shortest_homo_key = [k for k, v in shortest_distances_pair.items() if v == shortest_homoatomic_distance][0]
    shortest_hetero_key = [k for k, v in shortest_distances_pair.items() if v == shortest_heteroatomic_distance][0]

    # Extract 9 universal features for Ternary
    shortest_homoatomic_distance_by_2_by_atom_size = (shortest_homoatomic_distance / 2) / cif_radii[shortest_homo_key]
    shortest_heteroatomic_distance_by_sum_of_atom_sizes = shortest_heteroatomic_distance / cif_radii[shortest_hetero_key]
    shortest_homoatomic_distance_by_2_by_refined_atom_sizes = (shortest_homoatomic_distance / 2) / refined_radii[shortest_homo_key]
    shortest_heteroatomic_distance_by_refined_atom_sizes = shortest_heteroatomic_distance / refined_radii[shortest_hetero_key]
    highest_refined_percent_diff = max([abs(p) for p in percent_diffs])
    lowest_refined_percent_diff = min([abs(p) for p in percent_diffs])


    interatomic_universal_data = {
        "CIF_id": [CIF_id],
        "Compound": [formula_string],
        "Shortest homoatomic distance": [shortest_homoatomic_distance],
        "Shortest heteroatomic distance": [shortest_heteroatomic_distance],
        "Shortest homoatomic distance by 2 by atom size": [shortest_homoatomic_distance_by_2_by_atom_size],
        "Shortest heteroatomic distance by sum of atom sizes": [shortest_heteroatomic_distance_by_sum_of_atom_sizes],
        "Shortest homoatomic distance by 2 by refined atom size": [shortest_homoatomic_distance_by_2_by_refined_atom_sizes],
        "Shortest heteroatomic distance by sum of refined sizes": [shortest_heteroatomic_distance_by_refined_atom_sizes],
        "Highest refined percent difference by 100 (abs value)": [highest_refined_percent_diff],
        "Lowest refined percent difference by 100 (abs value)": [lowest_refined_percent_diff],
        "Packing efficiency": [packing_efficiency]
    }

    # log.print_json_pretty("interatomic_universal_data", interatomic_universal_data)
    df = pd.DataFrame(interatomic_universal_data)
    interatomic_universal_df = pd.concat([interatomic_universal_df, df], ignore_index=True)
    interatomic_universal_df = interatomic_universal_df.round(5)

    return interatomic_ternary_df, interatomic_universal_df
