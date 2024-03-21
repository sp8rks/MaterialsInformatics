import pandas as pd
import util.data as data
import preprocess.optimize as optimize
import featurizer.distance as distance
import featurizer.environment_wyckoff as env_wychoff_featurizer
import numpy as np
import util.log as log

def get_atom_values(atom, xl):
    row = xl[xl.iloc[:, 0] == atom]
    return {
        'CIF_rad': row['CIF radius element'].values[0],
        'Pauling_rad': row['Pauling R(CN12)'].values[0],
        'Group': row['Group'].values[0],
        'Mendeleev_number': row['Mendeleev number'].values[0],
        'valence_e': row['valence e total'].values[0],
        'Pauling_EN': row['Pauling EN'].values[0],
        'MB_EN': row['Martynov Batsanov EN'].values[0]
    }


def get_atomic_environment(CIF_loop_values):
    # Initialize a dictionary to store element information
    atomic_env = {}

    # Get the number of atoms
    num_atoms = len(CIF_loop_values[0])
    
    # Loop over all atoms
    for i in range(num_atoms):
        # Get atomic info
        label = CIF_loop_values[0][i]
        type_symbol = CIF_loop_values[1][i]
        multiplicity = int(CIF_loop_values[2][i])

        # If the atom is not in the dictionary, initialize it with default values
        if type_symbol not in atomic_env:
            atomic_env[type_symbol] = {
                "sites": 0,
                "multiplicity": 0,
                "lowest_wyckoff_multiplicity": multiplicity,
                "lowest_wyckoff_element": label
            }

        # Update the atom information in the dictionary
        atomic_env[type_symbol]["sites"] += 1
        atomic_env[type_symbol]["multiplicity"] += multiplicity

        # Update the element with the lowest Wyckoff multiplicity
        if multiplicity < atomic_env[type_symbol]["multiplicity"]:
            atomic_env[type_symbol]["lowest_wyckoff_multiplicity"] = multiplicity
            atomic_env[type_symbol]["lowest_wyckoff_element"] = label

    # Return the atomic environment dictionary
    return atomic_env


def get_atomic_environment_in_binary(CIF_loop_values, A, B):
    atomic_env = get_atomic_environment(CIF_loop_values)
    A_info, B_info = None, None

    # Check if the desired elements are present
    if A in atomic_env:
        A_info = atomic_env[A]
    if B in atomic_env:
        B_info = atomic_env[B]

    return A_info, B_info


def get_atomic_environment_in_ternary(CIF_loop_values, R, M, X):
    atomic_env = get_atomic_environment(CIF_loop_values)
    R_info, M_info, X_info = None, None, None

    # Check if the desired elements are present
    if R in atomic_env:
        R_info = atomic_env[R]
    if M in atomic_env:
        M_info = atomic_env[M]
    if X in atomic_env:
        X_info = atomic_env[X]

    return R_info, M_info, X_info

def get_env_wychoff_binary_df(filename,
                        xl,
                        atomic_environment_wyckoff_binary_df,
                        atomic_environment_wyckoff_universal_df,
                        unique_atoms_tuple,
                        CIF_loop_values,
                        radii_data,
                        CIF_data,
                        atomic_pair_list):

    A, B = unique_atoms_tuple[0][0], unique_atoms_tuple[1][0]
    A_info, B_info = get_atomic_environment_in_binary(CIF_loop_values, A, B)
    atoms_for_radii = [unique_atoms_tuple[0][0], unique_atoms_tuple[1][0]]  # [A, B]
    atom_radii = data.get_atom_radii(atoms_for_radii, radii_data)
    CIF_id, cell_lengths, cell_angles_rad, CIF_loop_values, formula_string = CIF_data

    A_CIF_rad = atom_radii[A]["CIF"]
    B_CIF_rad = atom_radii[B]["CIF"]

    shortest_AA, shortest_BB, shortest_AB = distance.find_shortest_pair_distances(True, unique_atoms_tuple, atomic_pair_list)
    shortest_distances_pair = {"AA": shortest_AA, "BB": shortest_BB, "AB": shortest_AB}
    radii, _ = optimize.optimize_CIF_rad_binary(A_CIF_rad, B_CIF_rad, shortest_distances_pair, True)
    A_CIF_rad_refined, B_CIF_rad_refined = radii

    A_sites_total = A_info['sites']
    A_multiplicity_total = A_info['multiplicity']
    A_lowest_wyckoff_multiplicity = A_info['lowest_wyckoff_multiplicity']

    B_sites_total = B_info['sites']
    B_multiplicity_total = B_info['multiplicity']
    B_lowest_wyckoff_multiplicity = B_info['lowest_wyckoff_multiplicity']

    # Create a list to store the elements with the lowest Wyckoff label
    lowest_wyckoff_elements = []

    # Determine the lowest Wyckoff label between A and B
    min_wyckoff_multiplicity = min(A_lowest_wyckoff_multiplicity, B_lowest_wyckoff_multiplicity)

    # If A or B have the lowest Wyckoff label, add them to the list
    if A_lowest_wyckoff_multiplicity == min_wyckoff_multiplicity:
        lowest_wyckoff_elements.append(A)
        
    if B_lowest_wyckoff_multiplicity == min_wyckoff_multiplicity:
        lowest_wyckoff_elements.append(B)

    identical_lowest_wyckoff_multiplicity_count = len(lowest_wyckoff_elements)
    
    atomic_environment_binary_Wyckoff_data = {
        "CIF_id": [CIF_id],
        "Compound": [formula_string],
        "A": [A],
        "B": [B],
        "lowest_wyckoff_elements": [lowest_wyckoff_elements],
        "A_lowest_wyckoff_label": [A_lowest_wyckoff_multiplicity],
        "B_lowest_wyckoff_label": [B_lowest_wyckoff_multiplicity],
        "identical_lowest_wyckoff_count": [identical_lowest_wyckoff_multiplicity_count],
        "A_sites_total": [A_sites_total],
        "B_sites_total": [B_sites_total],
        "A_multiplicity_total": [A_multiplicity_total],
        "B_multiplicity_total": [B_multiplicity_total],
    }
    
    # log.print_json_pretty("atomic_environment_binary_Wyckoff_data", atomic_environment_binary_Wyckoff_data)

            
    df = pd.DataFrame(atomic_environment_binary_Wyckoff_data)
    atomic_environment_wyckoff_binary_df = pd.concat([atomic_environment_wyckoff_binary_df, df], ignore_index=True)
    atomic_environment_wyckoff_binary_df.round(5)
    # log.print_json_pretty("atomic_environment_binary_Wyckoff_data", atomic_environment_binary_Wyckoff_data)


    first_data = {
                    "CIF_id": [CIF_id],
                    "Compound": [formula_string],
                    "lowest_wyckoff_elements": [lowest_wyckoff_elements],
                }

    atomic_environment_Wyckoff_universal_data = {}
    A_atom_property_values_from_Excel = get_atom_values(A, xl)
    B_atom_property_values_from_Excel = get_atom_values(B, xl)
    CIF_rad_refined_values = {'A': A_CIF_rad_refined, 'B': B_CIF_rad_refined}

    # Include 'CIF_rad_refined' in your properties list
    for property in ['Mendeleev_number', 'valence_e', 'CIF_rad', 'CIF_rad_refined', 'Pauling_rad', 'Pauling_EN', 'MB_EN']:
        if property == 'CIF_rad_refined':
            A_property_value = CIF_rad_refined_values['A']
            B_property_value = CIF_rad_refined_values['B']
        else:
            A_property_value = A_atom_property_values_from_Excel[property]
            B_property_value = B_atom_property_values_from_Excel[property]
        
        highest_property_element = A if A_property_value > B_property_value else B
        lowest_property_element = A if A_property_value < B_property_value else B

        lowest_wyckoff_property_values = []
        # Loop over each element with the lowest Wyckoff symbol
        for element in lowest_wyckoff_elements:
            # Fetch the property value for the element and append it to the list
            property_value = A_property_value if element == A else B_property_value
            lowest_wyckoff_property_values.append(property_value)
        
        # Fetch the corresponding number of sites and total multiplicity for these elements
        highest_property_sites = A_sites_total if highest_property_element == A else B_sites_total
        lowest_property_sites = A_sites_total if lowest_property_element == A else B_sites_total
        highest_property_multiplicity = A_multiplicity_total if highest_property_element == A else B_multiplicity_total
        lowest_property_multiplicity = A_multiplicity_total if lowest_property_element == A else B_multiplicity_total
        
        atomic_environment_Wyckoff_universal_data.update({
            f"{property}_of_elements_with_lowest_wyckoff": [np.around(lowest_wyckoff_property_values, 4)],
            f"highest_{property}_sites": [highest_property_sites],
            f"lowest_{property}_sites": [lowest_property_sites],
            f"highest_{property}_multiplicity": [highest_property_multiplicity],
            f"lowest_{property}_multiplicity": [lowest_property_multiplicity]
        })
                
    # Create an empty dictionary
    ordered_data = {}

    # First add keys that contain 'lowest_wyckoff'
    for key, value in atomic_environment_Wyckoff_universal_data.items():
        if 'lowest_wyckoff' in key:
            ordered_data[key] = value

    # Then add the rest
    for key, value in atomic_environment_Wyckoff_universal_data.items():
        if 'lowest_wyckoff' not in key:
            ordered_data[key] = value

                            
    # Merge first_data and ordered_data
    merged_data = {**first_data, **ordered_data}

    df = pd.DataFrame(merged_data)     
    atomic_environment_wyckoff_universal_df = pd.concat([atomic_environment_wyckoff_universal_df, df], ignore_index=True)
    atomic_environment_wyckoff_universal_df = atomic_environment_wyckoff_universal_df.round(5)

    # Loop through all columns and remove square brackets from lists
    for column in atomic_environment_wyckoff_binary_df.columns:
        atomic_environment_wyckoff_binary_df[column] = atomic_environment_wyckoff_binary_df[column].apply(lambda x: ', '.join(map(str, x)) if isinstance(x, list) else x)

    return atomic_environment_wyckoff_binary_df, atomic_environment_wyckoff_universal_df



def get_env_wychoff_ternary_df(filename,
                        xl,
                        atomic_environment_wyckoff_ternary_df,
                        atomic_environment_wyckoff_universal_df,
                        unique_atoms_tuple,
                        CIF_loop_values,
                        radii_data,
                        CIF_data,
                        atomic_pair_list):
    R, M, X = unique_atoms_tuple[0][0], unique_atoms_tuple[1][0], unique_atoms_tuple[2][0]
    CIF_id, cell_lengths, cell_angles_rad, CIF_loop_values, formula_string = CIF_data
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
    
    radii, _ = optimize.optimize_CIF_rad_ternary(R_CIF_rad, M_CIF_rad, X_CIF_rad, shortest_distances_pair, True)
    R_CIF_rad_refined, M_CIF_rad_refined, X_CIF_rad_refined = radii

    atoms = (R, M, X)

    R_info, M_info, X_info = env_wychoff_featurizer.get_atomic_environment_in_ternary(CIF_loop_values, R, M, X)

    R_sites_total = R_info['sites']
    R_multiplicity_total = R_info['multiplicity']
    R_lowest_wyckoff_multiplicity = R_info['lowest_wyckoff_multiplicity']

    M_sites_total = M_info['sites']
    M_multiplicity_total = M_info['multiplicity']
    M_lowest_wyckoff_multiplicity = M_info['lowest_wyckoff_multiplicity']

    X_sites_total = X_info['sites']
    X_multiplicity_total = X_info['multiplicity']
    X_lowest_wyckoff__multiplicity = X_info['lowest_wyckoff_multiplicity']

    lowest_wyckoff_elements = []

    min_wyckoff_multiplicity = min(R_lowest_wyckoff_multiplicity, M_lowest_wyckoff_multiplicity, X_lowest_wyckoff__multiplicity)

    if R_lowest_wyckoff_multiplicity == min_wyckoff_multiplicity:
        lowest_wyckoff_elements.append(R)
                    
    if M_lowest_wyckoff_multiplicity == min_wyckoff_multiplicity:
        lowest_wyckoff_elements.append(M)

    if X_lowest_wyckoff__multiplicity == min_wyckoff_multiplicity:
        lowest_wyckoff_elements.append(X)

    identical_lowest_wyckoff_count = len(lowest_wyckoff_elements)
                
    atomic_environment_Wyckoff_ternary_data = {
        "CIF_id": [CIF_id],
        "Compound": [formula_string],
        "R": [R],
        "M": [M],
        "X": [X],
        "lowest_wyckoff_elements": [lowest_wyckoff_elements],
        "R_lowest_wyckoff_label": [R_lowest_wyckoff_multiplicity],
        "M_lowest_wyckoff_label": [M_lowest_wyckoff_multiplicity],
        "X_lowest_wyckoff_label": [X_lowest_wyckoff__multiplicity],
        "identical_lowest_wyckoff_count": [identical_lowest_wyckoff_count],
        "R_sites_total": [R_sites_total],
        "M_sites_total": [M_sites_total],
        "X_sites_total": [X_sites_total],
        "R_multiplicity_total": [R_multiplicity_total],
        "M_multiplicity_total": [M_multiplicity_total],
        "X_multiplicity_total": [X_multiplicity_total],
    }

    
    df = pd.DataFrame(atomic_environment_Wyckoff_ternary_data)
    atomic_environment_wyckoff_ternary_df = pd.concat([atomic_environment_wyckoff_ternary_df, df], ignore_index=True)
    
    atomic_environment_wyckoff_ternary_df.round(5)
    # log.print_json_pretty("atomic_environment_ternary_Wyckoff_data", atomic_environment_Wyckoff_ternary_data)


    first_data = {
        "CIF_id": [CIF_id],
        "Compound": [formula_string],
        "lowest_wyckoff_elements": [lowest_wyckoff_elements],
    }

    # Create an empty dictionary to hold additional data
    atomic_environment_Wyckoff_universal_data = {}

    R_values = get_atom_values(R, xl)
    M_values = get_atom_values(M, xl)
    X_values = get_atom_values(X, xl)
    CIF_rad_refined_values = {R: R_CIF_rad_refined, M: M_CIF_rad_refined, X: X_CIF_rad_refined}

    for property in ['Mendeleev_number', 'valence_e', 'CIF_rad', 'CIF_rad_refined', 'Pauling_rad', 'Pauling_EN', 'MB_EN']:
        if property == 'CIF_rad_refined':
            R_value = CIF_rad_refined_values[R]
            M_value = CIF_rad_refined_values[M]
            X_value = CIF_rad_refined_values[X]
        else:
            R_value = R_values[property]
            M_value = M_values[property]
            X_value = X_values[property]

        property_values = {R: R_value, M: M_value, X: X_value}
        highest_property_element = max(property_values, key=property_values.get)
        lowest_property_element = min(property_values, key=property_values.get)

        lowest_wyckoff_property_values = []
        for element in lowest_wyckoff_elements:
            property_value = property_values[element]
            lowest_wyckoff_property_values.append(property_value)
        
        highest_property_sites = R_sites_total if highest_property_element == 'R' else M_sites_total if highest_property_element == 'M' else X_sites_total
        lowest_property_sites = R_sites_total if lowest_property_element == 'R' else M_sites_total if lowest_property_element == 'M' else X_sites_total
        highest_property_multiplicity = R_multiplicity_total if highest_property_element == 'R' else M_multiplicity_total if highest_property_element == 'M' else X_multiplicity_total
        lowest_property_multiplicity = R_multiplicity_total if lowest_property_element == 'R' else M_multiplicity_total if lowest_property_element == 'M' else X_multiplicity_total
        
        atomic_environment_Wyckoff_universal_data.update({
            f"{property}_of_elements_with_lowest_wyckoff": [np.around(lowest_wyckoff_property_values, 4)],
            f"highest_{property}_sites": [highest_property_sites],
            f"lowest_{property}_sites": [lowest_property_sites],
            f"highest_{property}_multiplicity": [highest_property_multiplicity],
            f"lowest_{property}_multiplicity": [lowest_property_multiplicity]
        })
                
    # Create an empty dictionary
    ordered_data = {}

    # First add keys that contain 'lowest_wyckoff'
    for key, value in atomic_environment_Wyckoff_universal_data.items():
        if 'lowest_wyckoff' in key:
            ordered_data[key] = value

    # Then add the rest
    for key, value in atomic_environment_Wyckoff_universal_data.items():
        if 'lowest_wyckoff' not in key:
            ordered_data[key] = value

    # Merge first_data and ordered_data
    merged_data = {**first_data, **ordered_data}

    df = pd.DataFrame(merged_data)     
    atomic_environment_wyckoff_universal_df = pd.concat([atomic_environment_wyckoff_universal_df, df], ignore_index=True)
    atomic_environment_wyckoff_universal_df = atomic_environment_wyckoff_universal_df.round(5)

    # Loop through all columns and remove square brackets from lists
    for column in atomic_environment_wyckoff_ternary_df.columns:
        atomic_environment_wyckoff_ternary_df[column] = atomic_environment_wyckoff_ternary_df[column].apply(lambda x: ', '.join(map(str, x)) if isinstance(x, list) else x)


    return atomic_environment_wyckoff_ternary_df, atomic_environment_wyckoff_universal_df