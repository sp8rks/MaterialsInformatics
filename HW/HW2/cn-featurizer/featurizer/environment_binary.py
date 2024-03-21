import preprocess.cif_parser as cif_parser
from collections import defaultdict

def get_unique_shortest_labels(atom_pairs_info_dict, unique_labels, cif_parser):
    '''
    This function processes given atom pair information and unique labels to determine the shortest labels representing
    unique atom types. The purpose is to identify unique atom types based on their shortest distance in a crystal structure 
    (usually from CIF files). For atoms with multiple labels (e.g., multiple occurrences in the structure), it selects the label with the shortest 
    interatomic distance, assuming this represents the most relevant configuration for that atom.
    '''          

    unique_shortest_labels = []
    atom_counts = defaultdict(int)

    # Iterate over unique labels to count occurrences of each atom type
    for label in unique_labels:
        # The atom type is the first part of the label, before the number
        atom_type = cif_parser.get_atom_type(label)
        # Increment the count for this atom type
        atom_counts[atom_type] += 1

    # Find atom types with more than one label
    multiple_labels = [atom for atom, count in atom_counts.items() if count > 1]
    
    if multiple_labels:
        for multiple_label in multiple_labels:
            # Find the labels in the atom_pairs_info_dict that start with this atom type
            matching_labels = [label for label in atom_pairs_info_dict.keys() if label.startswith(multiple_label)]
            shortest_distance = float('inf')  # initialize with infinity
            shortest_label = None
            
            for label in matching_labels:
                # Check if this label's shortest distance is shorter than the current shortest distance
                current_shortest_distance = min(atom_pairs_info_dict[label]["shortest_distances"])
                if current_shortest_distance < shortest_distance:
                    shortest_distance = current_shortest_distance
                    shortest_label = label
            
            # Remove all atom type except the label with the shortest distance
            matching_labels = [label for label in unique_labels if label.startswith(multiple_label)]
            unique_shortest_labels = [label for label in unique_labels if label not in matching_labels]
            unique_shortest_labels.append(shortest_label)
    else:
        unique_shortest_labels = unique_labels

    return unique_shortest_labels, atom_counts



def get_shortest_distances_count(unique_atoms_tuple, unique_labels, unique_shortest_labels, atom_pairs_info_dict, atom_counts):
    """
    Retrieves the shortest distances count and average shortest distances count for the given unique atom types based on their labels.

    This function iterates over the given unique shortest labels and, using the provided atom pairs information dictionary, 
    determines the count of shortest distances and the average shortest distances count for each atom type.
    """
    A, B = unique_atoms_tuple[0][0], unique_atoms_tuple[1][0]

    for label in unique_shortest_labels:        
        if label in atom_pairs_info_dict:
            data_dict = atom_pairs_info_dict[label]

            if label.startswith(A):
                A_shortest_dist_count = data_dict["shortest_distances_count"][0]
            elif label.startswith(B):
                B_shortest_dist_count = data_dict["shortest_distances_count"][0]

    # 2nd Binary features - average shortest distance count
    sum_counts = defaultdict(int)
    for label in unique_labels:
        data_dict = atom_pairs_info_dict[label]
        atom_type = cif_parser.get_atom_type(label)
        sum_counts[atom_type] += data_dict["shortest_distances_count"][0]

    average_counts = {}
    for atom, count in sum_counts.items():
        average_counts[atom] = count / atom_counts[atom]

    # print("Average shortest distances count:", average_counts)
    A_avg_shortest_dist_count = average_counts[A]
    B_avg_shortest_dist_count = average_counts[B]

    return A_shortest_dist_count, B_shortest_dist_count, A_avg_shortest_dist_count, B_avg_shortest_dist_count


def get_shortest_dist_count_with_tol(unique_atoms_tuple, unique_labels, unique_shortest_labels, atom_pairs_info_dict, atom_counts):
    """
    Retrieves the shortest distances count and average shortest distances count within a tolerance of 5% for the given unique atom types based on their labels.

    The function first determines the shortest distance and the average shortest distance within the tolerance for each unique label.
    """
    A, B = unique_atoms_tuple[0][0], unique_atoms_tuple[1][0]
    A_shortest_tol_dist_count = 0
    B_shortest_tol_dist_count = 0

    sum_counts_within_tol = defaultdict(int)

    # Combine Steps 1 & 2. Calculate the shortest distances and determine how many are within the 5% boundary

    for label, atom_dict in atom_pairs_info_dict.items():
        if label in unique_labels:
            shortest_distances_pair = [info[2] for info in atom_dict["pair_info"]]
            shortest_distance = min(shortest_distances_pair)
            tolerance = 0.05 * shortest_distance


            count_within_tolerance = sum(1 for distance in shortest_distances_pair if distance <= shortest_distance + tolerance)
            
            atom_type = cif_parser.get_atom_type(label)  # Assuming cif_parser.get_atom_type() can handle your label format
            sum_counts_within_tol[atom_type] += count_within_tolerance

            if label in unique_shortest_labels:
                if label.startswith(A):
                    A_shortest_tol_dist_count += count_within_tolerance
                elif label.startswith(B):
                    B_shortest_tol_dist_count += count_within_tolerance

    # 4th Feature - Avg # of shortest distances within 5% tolerance from the shortest
    average_counts_within_tol = {}
    for atom, count in sum_counts_within_tol.items():
        average_counts_within_tol[atom] = count / atom_counts[atom]

    A_avg_shortest_dist_within_tol_count = average_counts_within_tol.get(A, 0)
    B_avg_shortest_dist_within_tol_count = average_counts_within_tol.get(B, 0)

    return A_shortest_tol_dist_count, B_shortest_tol_dist_count, A_avg_shortest_dist_within_tol_count, B_avg_shortest_dist_within_tol_count


def get_second_by_first_shortest_dist(unique_atoms_tuple,
                                        unique_labels,
                                        unique_shortest_labels,
                                        atom_pairs_info_dict,
                                        atom_counts):
    
    A, B = unique_atoms_tuple[0][0], unique_atoms_tuple[1][0]

    A_second_by_first_shortest_dist = 0
    B_second_by_first_shortest_dist = 0

    for label in unique_shortest_labels:
        if label in atom_pairs_info_dict:
            data_dict = atom_pairs_info_dict[label]
            first_shortest_dist = data_dict["shortest_distances"][0]
            second_shortest_dist = data_dict["shortest_distances"][1]

            second_by_first_shortest_dist =  second_shortest_dist / first_shortest_dist

            if label.startswith(A):
                A_second_by_first_shortest_dist = second_by_first_shortest_dist
            elif label.startswith(B):
                B_second_by_first_shortest_dist = second_by_first_shortest_dist

    # For an element with 2 or more
    A_avg_second_by_first_shortest_dist = 0
    B_avg_second_by_first_shortest_dist = 0

    # Initialize dictionaries
    sum_ratios = {}
    label_counts = {}

    for label in unique_labels:
        if label in atom_pairs_info_dict:
            data_dict = atom_pairs_info_dict[label]
            first_shortest_dist = data_dict["shortest_distances"][0]
            second_shortest_dist = data_dict["shortest_distances"][1]

            # Compute the ratio
            second_by_first_shortest_dist =  second_shortest_dist / first_shortest_dist

            # The atom type is the first part of the label, before the number
            atom_type =  cif_parser.get_atom_type(label)

            # Initialize sum_ratios and label_counts for this atom type if needed
            if atom_type not in sum_ratios:
                sum_ratios[atom_type] = 0
                label_counts[atom_type] = 0

            # Sum up the ratios for this atom type
            sum_ratios[atom_type] += second_by_first_shortest_dist

            # Count the labels for this atom type
            label_counts[atom_type] += 1

    # Now you can compute the averages
    average_ratios = {}
    for atom_type in sum_ratios.keys():
        if label_counts[atom_type] > 0:  # Avoid division by zero
            average_ratios[atom_type] = sum_ratios[atom_type] / label_counts[atom_type]
        else:
            average_ratios[atom_type] = 0

    for label in unique_shortest_labels:
        # Check if the label belongs to unique_shortest_labels   
        if label.startswith(A):
            A_avg_second_by_first_shortest_dist = average_ratios[A]
        elif label.startswith(B):
            B_avg_second_by_first_shortest_dist = average_ratios[B]
    return A_second_by_first_shortest_dist, B_second_by_first_shortest_dist, A_avg_second_by_first_shortest_dist, B_avg_second_by_first_shortest_dist

def get_second_shortest_dist_count(unique_atoms_tuple,
                                        unique_labels,
                                        unique_shortest_labels,
                                        atom_pairs_info_dict,
                                        atom_counts):
    
    A, B = unique_atoms_tuple[0][0], unique_atoms_tuple[1][0]
    A_second_shortest_dist_count = 0
    B_second_shortest_dist_count = 0

    for label in unique_shortest_labels:
        if label in atom_pairs_info_dict:
            data_dict = atom_pairs_info_dict[label]
            second_shortest_dist = data_dict["shortest_distances"][1]

            if label.startswith(A):
                A_second_shortest_dist_count = data_dict["shortest_distances_count"][1]
            elif label.startswith(B):
                B_second_shortest_dist_count = data_dict["shortest_distances_count"][1]

 
    sum_second_shortest_counts = defaultdict(int)

    for label in unique_labels:
        data_dict = atom_pairs_info_dict[label]

        # The atom type is the first part of the label, before the number
        atom_type =  cif_parser.get_atom_type(label)

        # Sum up the shortest_distances_count for this atom type
        sum_second_shortest_counts[atom_type] += data_dict["shortest_distances_count"][1]

    # Prepare a dictionary to store average shortest_distances_count for each atom type
    average_counts = {}

    for atom, count in sum_second_shortest_counts.items():
        # Divide the sum by the number of labels for this atom to get the average
        average_counts[atom] = count / atom_counts[atom]

    A_avg_second_shortest_dist_count = average_counts[A]
    B_avg_second_shortest_dist_count = average_counts[B]

    return A_second_shortest_dist_count, B_second_shortest_dist_count, A_avg_second_shortest_dist_count, B_avg_second_shortest_dist_count


def get_homoatomic_dist_by_shortest_dist_count(unique_atoms_tuple,
                                        unique_labels,
                                        unique_shortest_labels,
                                        atom_pairs_info_dict,
                                        atom_counts):
    
    A, B = unique_atoms_tuple[0][0], unique_atoms_tuple[1][0]

    A_homoatomic_dist_by_shortest_dist = 0
    B_homoatomic_dist_by_shortest_dist = 0

    for label in unique_shortest_labels:
        if label in atom_pairs_info_dict:
            data_dict = atom_pairs_info_dict[label]
            shortest_dist = data_dict["shortest_distances"][0]
            
            pair_info = data_dict["pair_info"]

            # Loop through each pair in pair_info
            for pair in pair_info:
                # Check the labels of both elements in the pair
                first_label = pair[1][0]
                second_label = pair[1][1]
                dist = pair[2]

                if label.startswith(A) and (first_label.startswith(A) and second_label.startswith(A)):
                    A_homoatomic_dist_by_shortest_dist = dist/shortest_dist
                    break
                elif label.startswith(B) and (first_label.startswith(B) and second_label.startswith(B)):
                    B_homoatomic_dist_by_shortest_dist = dist/shortest_dist
                    break

    A_total_dist_by_shortest_dist = 0
    B_total_dist_by_shortest_dist = 0

    for label in unique_labels:
        if label in atom_pairs_info_dict:
            data_dict = atom_pairs_info_dict[label]
            shortest_dist = data_dict["shortest_distances"][0]

            pair_info = data_dict["pair_info"]

            # Loop through each pair in pair_info
            for pair in pair_info:
                # Check the labels of both elements in the pair
                first_label = pair[1][0]
                second_label = pair[1][1]
                dist = pair[2]

                if label.startswith(A) and (first_label.startswith(A) and second_label.startswith(A)):
                    A_total_dist_by_shortest_dist += dist/shortest_dist
                    break
                elif label.startswith(B) and (first_label.startswith(B) and second_label.startswith(B)):
                    B_total_dist_by_shortest_dist += dist/shortest_dist
                    break

    # Compute average for each label after the loop
    A_avg_homoatomic_dist_by_shortest_dist = A_total_dist_by_shortest_dist / atom_counts[A] if atom_counts[A] > 0 else 0
    B_avg_homoatomic_dist_by_shortest_dist = B_total_dist_by_shortest_dist / atom_counts[B] if atom_counts[B] > 0 else 0

    return A_homoatomic_dist_by_shortest_dist, B_homoatomic_dist_by_shortest_dist, A_avg_homoatomic_dist_by_shortest_dist, B_avg_homoatomic_dist_by_shortest_dist


def get_A_B_count_at_shortest_dist(unique_atoms_tuple,
                                    unique_labels,
                                    unique_shortest_labels,
                                    atom_pairs_info_dict,
                                    atom_counts):
    A_count_at_A_shortest_dist = 0
    B_count_at_A_shortest_dist = 0
    A_count_at_B_shortest_dist = 0
    B_count_at_B_shortest_dist = 0

    A, B = unique_atoms_tuple[0][0], unique_atoms_tuple[1][0]

    for label in unique_shortest_labels:
        if label in atom_pairs_info_dict:
            data_dict = atom_pairs_info_dict[label]
            shortest_dist = data_dict["shortest_distances"][0]
            pair_info = data_dict["pair_info"]

            # Loop through each pair in pair_info
            for pair in pair_info:
                # Check the labels of both elements in the pair
                second_label = pair[1][1]
                dist = pair[2]

                # Only consider the pair if its distance is equal to the shortest distance for the label
                if round(dist, 2) == shortest_dist:
                    # Check combinations of first and second label
                    if label.startswith(A):
                        if second_label.startswith(A):
                            A_count_at_A_shortest_dist += 1
                        elif second_label.startswith(B):
                            B_count_at_A_shortest_dist += 1

                    elif label.startswith(B):
                        if second_label.startswith(A):
                            A_count_at_B_shortest_dist += 1
                        elif second_label.startswith(B):
                            B_count_at_B_shortest_dist += 1


    A_new_count_at_A_shortest_dist = 0
    B_new_count_at_A_shortest_dist = 0

    A_new_count_at_B_shortest_dist = 0
    B_new_count_at_B_shortest_dist = 0

    for label in unique_labels:
        if label in atom_pairs_info_dict:
            data_dict = atom_pairs_info_dict[label]
            shortest_dist = data_dict["shortest_distances"][0]
            pair_info = data_dict["pair_info"]

            # Loop through each pair in pair_info
            for pair in pair_info:
                # Check the labels of both elements in the pair
                first_label = pair[1][0]
                second_label = pair[1][1]
                dist = pair[2]

                # Only consider the pair if its distance is equal to the shortest distance for the label
                if round(dist, 2) == shortest_dist:
                    # Check combinations of first and second label
                    if label.startswith(A):
                        if second_label.startswith(A):
                            A_new_count_at_A_shortest_dist += 1
                        elif second_label.startswith(B):
                            B_new_count_at_A_shortest_dist += 1

                    elif label.startswith(B):
                        if second_label.startswith(A):
                            A_new_count_at_B_shortest_dist += 1
                        elif second_label.startswith(B):
                            B_new_count_at_B_shortest_dist += 1


    # Calculate average for new counts
    A_avg_new_count_at_A_shortest_dist = A_new_count_at_A_shortest_dist / atom_counts[A] if atom_counts[A] > 0 else 0
    B_avg_new_count_at_A_shortest_dist = B_new_count_at_A_shortest_dist / atom_counts[A] if atom_counts[A] > 0 else 0

    A_avg_new_count_at_B_shortest_dist = A_new_count_at_B_shortest_dist / atom_counts[B] if atom_counts[B] > 0 else 0
    B_avg_new_count_at_B_shortest_dist = B_new_count_at_B_shortest_dist / atom_counts[B] if atom_counts[B] > 0 else 0
    
    return (A_count_at_A_shortest_dist, B_count_at_A_shortest_dist, A_count_at_B_shortest_dist, B_new_count_at_B_shortest_dist), (A_avg_new_count_at_A_shortest_dist, B_avg_new_count_at_A_shortest_dist, A_avg_new_count_at_B_shortest_dist, B_avg_new_count_at_B_shortest_dist)


