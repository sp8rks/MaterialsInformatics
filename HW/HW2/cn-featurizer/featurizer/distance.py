from preprocess.cif_parser import get_atom_type


def find_shortest_pair_distances(isBinary, unique_atoms_tuple, atomic_pair_list):
    """
    Computes the shortest distances between pairs of atoms based on their labels.
    
    For a binary system, this function calculates the shortest distances for AA, BB, and AB pairs.
    For a ternary system, it calculates the shortest distances for RR, MM, XX, RM, MX, and RX pairs.
    """
    
    if isBinary:
        pair_keys = [('AA', unique_atoms_tuple[0][0], unique_atoms_tuple[0][0]),
                     ('BB', unique_atoms_tuple[1][0], unique_atoms_tuple[1][0]),
                     ('AB', unique_atoms_tuple[0][0], unique_atoms_tuple[1][0])]
    else:
        pair_keys = [('RR', unique_atoms_tuple[0][0], unique_atoms_tuple[0][0]),
                     ('MM', unique_atoms_tuple[1][0], unique_atoms_tuple[1][0]),
                     ('XX', unique_atoms_tuple[2][0], unique_atoms_tuple[2][0]),
                     ('RM', unique_atoms_tuple[0][0], unique_atoms_tuple[1][0]),
                     ('MX', unique_atoms_tuple[1][0], unique_atoms_tuple[2][0]),
                     ('RX', unique_atoms_tuple[0][0], unique_atoms_tuple[2][0])]
    
    shortest_distances = {key: float('inf') for key, _, _ in pair_keys}
    
    for pair in atomic_pair_list:
        label1 = get_atom_type(pair['labels'][0])
        label2 = get_atom_type(pair['labels'][1])
        distance = pair['distance']

        for key, atom1, atom2 in pair_keys:
            if (label1 == atom1 and label2 == atom2) or (label1 == atom2 and label2 == atom1):
                shortest_distances[key] = min(shortest_distances[key], distance)

    return tuple(shortest_distances[key] for key, _, _ in pair_keys)

