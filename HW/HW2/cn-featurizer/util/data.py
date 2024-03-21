def get_radii_data():
    """
    Return a dictionary of element radii data.
    """
    data = {
        'Si': [1.176, 1.316], 'Sc': [1.641, 1.620], 'Fe': [1.242, 1.260],
        'Co': [1.250, 1.252], 'Ni': [1.246, 1.244], 'Ga': [1.243, 1.408],
        'Ge': [1.225, 1.366], 'Y': [1.783, 1.797], 'Ru': [1.324, 1.336],
        'Rh': [1.345, 1.342], 'Pd': [1.376, 1.373], 'In': [1.624, 1.660],
        'Sn': [1.511, 1.620], 'Sb': [1.434, 1.590], 'La': [1.871, 1.871],
        'Ce': [1.819, 1.818], 'Pr': [1.820, 1.824], 'Nd': [1.813, 1.818],
        'Sm': [1.793, 1.850], 'Eu': [1.987, 2.084], 'Gd': [1.787, 1.795],
        'Tb': [1.764, 1.773], 'Dy': [1.752, 1.770], 'Ho': [1.745, 1.761],
        'Er': [1.734, 1.748], 'Tm': [1.726, 1.743], 'Yb': [1.939, 1.933],
        'Lu': [1.718, 1.738], 'Os': [1.337, 1.350], 'Ir': [1.356, 1.355],
        'Pt': [1.387, 1.385], 'Th': [1.798, 1.795], 'U': [1.377, 1.51],
        'Al': [1.310, 1.310]
    }

    radii_data = {k: {'CIF_radius_element': v[0], 'Pauling_R(CN12)': v[1]} for k, v in data.items()}
    
    return radii_data


def get_atom_radii(atoms, radii_data):
    """
    Returns atom radii data for a list of atoms from radii data
    """
    radii = {}
    for atom in atoms:
        radii[atom] = {
            "CIF": radii_data[atom]['CIF_radius_element'],
            "Pauling": radii_data[atom]['Pauling_R(CN12)']
        }
        
    return radii
    

def compute_rad_sum_binary(A_CIF, B_CIF, A_CIF_refined, B_CIF_refined, A_Pauling, B_Pauling):
    """
    Computes sum of radii for binary compounds.
    """
    return {
        "CIF_rad_sum": {
            "A_A": A_CIF * 2,
            "A_B": A_CIF + B_CIF,
            "B_A": B_CIF + A_CIF,
            "B_B": B_CIF * 2
        },
        "CIF_rad_refined_sum": {
            "A_A": A_CIF_refined * 2,
            "A_B": A_CIF_refined + B_CIF_refined,
            "B_A": B_CIF_refined + A_CIF_refined,
            "B_B": B_CIF_refined * 2
        },
        "Pauling_rad_sum": {
            "A_A": A_Pauling * 2,
            "A_B": A_Pauling + B_Pauling,
            "B_A": B_Pauling + A_Pauling,
            "B_B": B_Pauling * 2
        }
    }


def compute_rad_sum_ternary(R_CIF, M_CIF, X_CIF, R_CIF_refined, M_CIF_refined, X_CIF_refined,R_Pauling, M_Pauling, X_Pauling):
    """
    Computes sum of radii for ternary compounds.
    """
    return {
        "CIF_rad_sum": {
            "R_R": R_CIF * 2,
            "R_M": R_CIF + M_CIF,
            "R_X": R_CIF + X_CIF,
            "M_R": M_CIF + R_CIF,
            "M_M": M_CIF * 2,
            "M_X": M_CIF + X_CIF,
            "X_R": X_CIF + R_CIF,
            "X_M": X_CIF + M_CIF,
            "X_X": X_CIF * 2
        },
        "CIF_rad_refined_sum": {
            "R_R": R_CIF_refined * 2,
            "R_M": R_CIF_refined + M_CIF_refined,
            "R_X": R_CIF_refined + X_CIF_refined,
            "M_R": M_CIF_refined + R_CIF_refined,
            "M_M": M_CIF_refined * 2,
            "M_X": M_CIF_refined + X_CIF_refined,
            "X_R": X_CIF_refined + R_CIF_refined,
            "X_M": X_CIF_refined + M_CIF_refined,
            "X_X": X_CIF_refined * 2
        },
        "Pauling_rad_sum": {
            "R_R": R_Pauling * 2,
            "R_M": R_Pauling + M_Pauling,
            "R_X": R_Pauling + X_Pauling,
            "M_R": M_Pauling + R_Pauling,
            "M_M": M_Pauling * 2,
            "M_X": M_Pauling + X_Pauling,
            "X_R": X_Pauling + R_Pauling,
            "X_M": X_Pauling + M_Pauling,
            "X_X": X_Pauling * 2
        }
    }
