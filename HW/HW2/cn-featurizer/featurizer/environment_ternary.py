import preprocess.cif_parser as cif_parser
from collections import defaultdict


def get_shortest_distances_count(unique_atoms_tuple,
                                unique_labels,
                                unique_shortest_labels,
                                atom_pairs_info_dict,
                                atom_counts):
    
    R_shortest_dist_count = M_shortest_dist_count = X_shortest_dist_count = 0.0

    R, M, X = unique_atoms_tuple[0][0], unique_atoms_tuple[1][0], unique_atoms_tuple[2][0]
    
    for label in unique_shortest_labels:
        if label in atom_pairs_info_dict:
            data_dict = atom_pairs_info_dict[label]

            if label.startswith(R):
                R_shortest_dist = data_dict["shortest_distances"][0]
                R_shortest_dist_count = data_dict["shortest_distances_count"][0]
            elif label.startswith(M):
                M_shortest_dist = data_dict["shortest_distances"][0]
                M_shortest_dist_count = data_dict["shortest_distances_count"][0]
            elif label.startswith(X):
                X_shortest_dist = data_dict["shortest_distances"][0]
                X_shortest_dist_count = data_dict["shortest_distances_count"][0]

    sum_counts = defaultdict(int)

    for label in unique_labels:
        data_dict = atom_pairs_info_dict[label]

        # The atom type is the first part of the label, before the number
        atom_type = cif_parser.get_atom_type(label)

        # Sum up the shortest_distances_count for this atom type
        sum_counts[atom_type] += data_dict["shortest_distances_count"][0]

    # Prepare a dictionary to store average shortest_distances_count for each atom type
    average_counts = {}

    for atom, count in sum_counts.items():
        # Divide the sum by the number of labels for this atom to get the average
        average_counts[atom] = count / atom_counts[atom]

    R_avg_shortest_dist_count = average_counts[R]
    M_avg_shortest_dist_count = average_counts[M]
    X_avg_shortest_dist_count = average_counts[X]

    return (R_shortest_dist_count, M_shortest_dist_count, X_shortest_dist_count), (R_avg_shortest_dist_count, M_avg_shortest_dist_count, X_avg_shortest_dist_count)


# 3rd, 4th ternary features - # of shortest distances within 5% tolerance from the shortest
def get_shortest_dist_count_with_tol(unique_atoms_tuple,
                                    unique_labels,
                                    unique_shortest_labels,
                                    atom_pairs_info_dict,
                                    atom_counts):        
        
    R, M, X = unique_atoms_tuple[0][0], unique_atoms_tuple[1][0], unique_atoms_tuple[2][0]
    R_shortest_tol_dist_count = M_shortest_tol_dist_count = X_shortest_tol_dist_count = 0
    for label, atom_dict in atom_pairs_info_dict.items():
        # Check if the label belongs to unique_shortest_labels
        if label in unique_labels:
            # Extract all distances from pair_info for this label
            shortest_distances_pair = [info[2] for info in atom_dict["pair_info"]]
            
            shortest_distance = min(shortest_distances_pair)
            tolerance = 0.05 * shortest_distance

            # Count how many distances are within the 5% tolerance
            count_within_tolerance = sum(1 for distance in shortest_distances_pair if distance <= shortest_distance + tolerance)

            # Add the count to the dictionary
            atom_dict["count_within_5_percent_of_shortest"] = count_within_tolerance


    # Step 2. Filter the points where it is within the 5% boundary and count them
    for label, atom_dict in atom_pairs_info_dict.items():
        # Check if the label belongs to unique_shortest_labels
        if label in unique_shortest_labels:
            # Extract all distances from pair_info for this label
            
            count_within_tolerance = atom_dict["count_within_5_percent_of_shortest"]

            if label.startswith(R):
                R_shortest_tol_dist_count = count_within_tolerance
            elif label.startswith(M):
                M_shortest_tol_dist_count = count_within_tolerance
            elif label.startswith(X):
                X_shortest_tol_dist_count = count_within_tolerance

    # Step 3. Determine the average count
    sum_counts_within_tol = defaultdict(int)

    # Iterate over the labels in unique_shortest_labels
    for label in unique_labels:
        data_dict = atom_pairs_info_dict[label]

        # The atom type is the first part of the label, before the number
        atom_type = ''.join(filter(str.isalpha, label))

        # Sum up the counts of distances within the 5% tolerance for this atom type
        sum_counts_within_tol[atom_type] += data_dict["count_within_5_percent_of_shortest"]

    # Prepare a dictionary to store average counts of distances within the 5% tolerance for each atom type
    average_counts_within_tol = {}

    for atom, count in sum_counts_within_tol.items():
        # Divide the sum by the number of labels for this atom to get the average
        average_counts_within_tol[atom] = count / atom_counts[atom]

    R_avg_shortest_dist_within_tol_count = average_counts_within_tol[R]
    M_avg_shortest_dist_within_tol_count = average_counts_within_tol[M]
    X_avg_shortest_dist_within_tol_count = average_counts_within_tol[X]

    shortest_tol_dist_count_res = (R_shortest_tol_dist_count, M_shortest_tol_dist_count, X_shortest_tol_dist_count)
    avg_shortest_tol_dist_count_res = (R_avg_shortest_dist_within_tol_count, M_avg_shortest_dist_within_tol_count, X_avg_shortest_dist_within_tol_count)

    return shortest_tol_dist_count_res, avg_shortest_tol_dist_count_res


def get_second_by_first_shortest_dist(
    unique_atoms_tuple,
    unique_labels,
    unique_shortest_labels,
    atom_pairs_info_dict,
    atom_counts):

    R, M, X = unique_atoms_tuple[0][0], unique_atoms_tuple[1][0], unique_atoms_tuple[2][0]

    R_second_by_first_shortest_dist = 0
    M_second_by_first_shortest_dist = 0
    X_second_by_first_shortest_dist = 0

    for label in unique_shortest_labels:
        if label in atom_pairs_info_dict:
            data_dict = atom_pairs_info_dict[label]
            first_shortest_dist = data_dict["shortest_distances"][0]
            second_shortest_dist = data_dict["shortest_distances"][1]

            second_by_first_shortest_dist =  second_shortest_dist / first_shortest_dist

            if label.startswith(R):
                R_second_by_first_shortest_dist = second_by_first_shortest_dist
            elif label.startswith(M):
                M_second_by_first_shortest_dist = second_by_first_shortest_dist
            elif label.startswith(X):
                X_second_by_first_shortest_dist = second_by_first_shortest_dist

    # For an element with 2 or more
    R_avg_second_by_first_shortest_dist = 0
    M_avg_second_by_first_shortest_dist = 0
    X_avg_second_by_first_shortest_dist = 0

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
            atom_type = cif_parser.get_atom_type(label)

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
        if label.startswith(R):
            R_avg_second_by_first_shortest_dist = average_ratios[R]
        elif label.startswith(M):
            M_avg_second_by_first_shortest_dist = average_ratios[M]
        elif label.startswith(X):
            X_avg_second_by_first_shortest_dist = average_ratios[X]

    second_by_first_shortest_dist_res = (R_second_by_first_shortest_dist, M_second_by_first_shortest_dist, X_second_by_first_shortest_dist)
    avg_second_by_first_shortest_dist_res = (R_avg_second_by_first_shortest_dist, M_avg_second_by_first_shortest_dist, X_avg_second_by_first_shortest_dist)

    return second_by_first_shortest_dist_res, avg_second_by_first_shortest_dist_res




# 7th, 8th ternary features - atom counts at the 2nd shortest distance
def get_second_shortest_dist_count(
    unique_atoms_tuple,
    unique_labels,
    unique_shortest_labels,
    atom_pairs_info_dict,
    atom_counts 
    ):

    R, M, X = unique_atoms_tuple[0][0], unique_atoms_tuple[1][0], unique_atoms_tuple[2][0]

    R_second_shortest_dist_count = 0
    M_second_shortest_dist_count = 0
    X_second_shortest_dist_count = 0

    for label in unique_shortest_labels:
        if label in atom_pairs_info_dict:
            data_dict = atom_pairs_info_dict[label]
            second_shortest_dist = data_dict["shortest_distances"][1]

            if label.startswith(R):
                R_second_shortest_dist_count = data_dict["shortest_distances_count"][1]
            elif label.startswith(M):
                M_second_shortest_dist_count = data_dict["shortest_distances_count"][1]
            elif label.startswith(X):
                X_second_shortest_dist_count = data_dict["shortest_distances_count"][1]


    sum_second_shortest_counts = defaultdict(int)

    for label in unique_labels:
        data_dict = atom_pairs_info_dict[label]

        # The atom type is the first part of the label, before the number
        atom_type = cif_parser.get_atom_type(label)

        # Sum up the shortest_distances_count for this atom type
        sum_second_shortest_counts[atom_type] += data_dict["shortest_distances_count"][1]

    # Prepare a dictionary to store average shortest_distances_count for each atom type
    average_counts = {}

    for atom, count in sum_second_shortest_counts.items():
        # Divide the sum by the number of labels for this atom to get the average
        average_counts[atom] = count / atom_counts[atom]

    R_avg_second_shortest_dist_count = average_counts[R]
    M_avg_second_shortest_dist_count = average_counts[M]
    X_avg_second_shortest_dist_count = average_counts[X]

    second_shortest_dist_count_res = (R_second_shortest_dist_count, M_second_shortest_dist_count, X_second_shortest_dist_count)
    avg_second_shortest_dist_count_res = (R_avg_second_shortest_dist_count, M_avg_second_shortest_dist_count, X_avg_second_shortest_dist_count)

    return second_shortest_dist_count_res, avg_second_shortest_dist_count_res






# 9th, 10th ternary features - homoatomic shortest distance / shortest distance
def get_homoatomic_dist_by_shortest_dist_countd(
    unique_atoms_tuple,
    unique_labels,
    unique_shortest_labels,
    atom_pairs_info_dict,
    atom_counts 
    ):
    
    R, M, X = unique_atoms_tuple[0][0], unique_atoms_tuple[1][0], unique_atoms_tuple[2][0]
    
    R_homoatomic_dist_by_shortest_dist = 0
    M_homoatomic_dist_by_shortest_dist = 0
    X_homoatomic_dist_by_shortest_dist = 0

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

                if label.startswith(R) and (first_label.startswith(R) and second_label.startswith(R)):
                    R_homoatomic_dist_by_shortest_dist = dist/shortest_dist
                    break
                elif label.startswith(M) and (first_label.startswith(M) and second_label.startswith(M)):
                    M_homoatomic_dist_by_shortest_dist = dist/shortest_dist
                    break
                elif label.startswith(X) and (first_label.startswith(X) and second_label.startswith(X)):
                    X_homoatomic_dist_by_shortest_dist = dist/shortest_dist
                    break

    # Feature 10. Average homoatomic shortest distance / shortest distance
    R_total_dist_by_shortest_dist = 0
    M_total_dist_by_shortest_dist = 0
    X_total_dist_by_shortest_dist = 0

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

                if label.startswith(R) and (first_label.startswith(R) and second_label.startswith(R)):
                    R_total_dist_by_shortest_dist += dist/shortest_dist
                    break
                elif label.startswith(M) and (first_label.startswith(M) and second_label.startswith(M)):
                    M_total_dist_by_shortest_dist += dist/shortest_dist
                    break
                elif label.startswith(X) and (first_label.startswith(X) and second_label.startswith(X)):
                    X_total_dist_by_shortest_dist += dist/shortest_dist
                    break

    # Compute average for each label after the loop
    R_avg_homoatomic_dist_by_shortest_dist = R_total_dist_by_shortest_dist / atom_counts[R] if atom_counts[R] > 0 else 0
    M_avg_homoatomic_dist_by_shortest_dist = M_total_dist_by_shortest_dist / atom_counts[M] if atom_counts[M] > 0 else 0
    X_avg_homoatomic_dist_by_shortest_dist = X_total_dist_by_shortest_dist / atom_counts[X] if atom_counts[X] > 0 else 0

    homoatomic_dist_by_shortest_dist_res = (R_homoatomic_dist_by_shortest_dist, M_homoatomic_dist_by_shortest_dist, X_homoatomic_dist_by_shortest_dist)
    avg_homoatomic_dist_by_shortest_dist_res = (R_avg_homoatomic_dist_by_shortest_dist, M_avg_homoatomic_dist_by_shortest_dist, X_avg_homoatomic_dist_by_shortest_dist)

    return homoatomic_dist_by_shortest_dist_res, avg_homoatomic_dist_by_shortest_dist_res


# 11th, 12th ternary features - A, B count at the shortest distance
def get_R_X_X_count_at_shortest_dist(
    unique_atoms_tuple,
    unique_labels,
    unique_shortest_labels,
    atom_pairs_info_dict,
    atom_counts 
    ):

    R, M, X = unique_atoms_tuple[0][0], unique_atoms_tuple[1][0], unique_atoms_tuple[2][0]

    # Feature 11. Number of X, at the shortest distance
    R_count_at_R_shortest_dist = 0
    M_count_at_R_shortest_dist = 0
    X_count_at_R_shortest_dist = 0

    R_count_at_M_shortest_dist = 0
    M_count_at_M_shortest_dist = 0
    X_count_at_M_shortest_dist = 0

    R_count_at_X_shortest_dist = 0
    M_count_at_X_shortest_dist = 0
    X_count_at_X_shortest_dist = 0

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
                    if label.startswith(R):
                        if second_label.startswith(R):
                            R_count_at_R_shortest_dist += 1
                        elif second_label.startswith(M):
                            M_count_at_R_shortest_dist += 1
                        elif second_label.startswith(X):
                            X_count_at_R_shortest_dist += 1
                    elif label.startswith(M):
                        if second_label.startswith(R):
                            R_count_at_M_shortest_dist += 1
                        elif second_label.startswith(M):
                            M_count_at_M_shortest_dist += 1
                        elif second_label.startswith(X):
                            X_count_at_M_shortest_dist += 1
                    elif label.startswith(X):
                        if second_label.startswith(R):
                            R_count_at_X_shortest_dist += 1
                        elif second_label.startswith(M):
                            M_count_at_X_shortest_dist += 1
                        elif second_label.startswith(X):
                            X_count_at_X_shortest_dist += 1

    # Feature 12. Avg number of M, R, X at the shortest distance
    R_new_count_at_R_shortest_dist = 0
    M_new_count_at_R_shortest_dist = 0
    X_new_count_at_R_shortest_dist = 0

    R_new_count_at_M_shortest_dist = 0
    M_new_count_at_M_shortest_dist = 0
    X_new_count_at_M_shortest_dist = 0

    R_new_count_at_X_shortest_dist = 0
    M_new_count_at_X_shortest_dist = 0
    X_new_count_at_X_shortest_dist = 0

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
                    if label.startswith(R):
                        if second_label.startswith(R):
                            R_new_count_at_R_shortest_dist += 1
                        elif second_label.startswith(M):
                            M_new_count_at_R_shortest_dist += 1
                        elif second_label.startswith(X):
                            X_new_count_at_R_shortest_dist += 1
                    elif label.startswith(M):
                        if second_label.startswith(R):
                            R_new_count_at_M_shortest_dist += 1
                        elif second_label.startswith(M):
                            M_new_count_at_M_shortest_dist += 1
                        elif second_label.startswith(X):
                            X_new_count_at_M_shortest_dist += 1
                    elif label.startswith(X):
                        if second_label.startswith(R):
                            R_new_count_at_X_shortest_dist += 1
                        elif second_label.startswith(M):
                            M_new_count_at_X_shortest_dist += 1
                        elif second_label.startswith(X):
                            X_new_count_at_X_shortest_dist += 1

    # Calculate average
    R_avg_new_count_at_R_shortest_dist = R_new_count_at_R_shortest_dist / atom_counts[R] if atom_counts[R] > 0 else 0
    M_avg_new_count_at_R_shortest_dist = M_new_count_at_R_shortest_dist / atom_counts[R] if atom_counts[R] > 0 else 0
    X_avg_new_count_at_R_shortest_dist = X_new_count_at_R_shortest_dist / atom_counts[R] if atom_counts[R] > 0 else 0

    R_avg_new_count_at_M_shortest_dist = R_new_count_at_M_shortest_dist / atom_counts[M] if atom_counts[M] > 0 else 0
    M_avg_new_count_at_M_shortest_dist = M_new_count_at_M_shortest_dist / atom_counts[M] if atom_counts[M] > 0 else 0
    X_avg_new_count_at_M_shortest_dist = X_new_count_at_M_shortest_dist / atom_counts[M] if atom_counts[M] > 0 else 0

    R_avg_new_count_at_X_shortest_dist = R_new_count_at_X_shortest_dist / atom_counts[X] if atom_counts[X] > 0 else 0
    M_avg_new_count_at_X_shortest_dist = M_new_count_at_X_shortest_dist / atom_counts[X] if atom_counts[X] > 0 else 0
    X_avg_new_count_at_X_shortest_dist = X_new_count_at_X_shortest_dist / atom_counts[X] if atom_counts[X] > 0 else 0


    count_at_R_shortest_dist_res = (R_count_at_R_shortest_dist, M_count_at_R_shortest_dist, X_count_at_R_shortest_dist)
    count_at_M_shortest_dist_res = (R_count_at_M_shortest_dist, M_count_at_M_shortest_dist, X_count_at_M_shortest_dist)
    count_at_X_shortest_dist_res = (R_count_at_X_shortest_dist, M_count_at_X_shortest_dist, X_count_at_X_shortest_dist)


    avg_count_at_R_shortest_dist_res = (R_avg_new_count_at_R_shortest_dist, M_avg_new_count_at_R_shortest_dist, X_avg_new_count_at_R_shortest_dist)
    avg_count_at_M_shortest_dist_res = (R_avg_new_count_at_M_shortest_dist, M_avg_new_count_at_M_shortest_dist, X_avg_new_count_at_M_shortest_dist)
    avg_count_at_X_shortest_dist_res = (R_avg_new_count_at_X_shortest_dist, M_avg_new_count_at_X_shortest_dist, X_avg_new_count_at_X_shortest_dist)

    return count_at_R_shortest_dist_res, count_at_M_shortest_dist_res, count_at_X_shortest_dist_res, avg_count_at_R_shortest_dist_res, avg_count_at_M_shortest_dist_res, avg_count_at_X_shortest_dist_res