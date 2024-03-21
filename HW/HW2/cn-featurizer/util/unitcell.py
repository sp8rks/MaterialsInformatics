import math
import sympy

def get_unit_cell_volume(lenghts, angles):
    a_length, b_length, c_length = lenghts
    alpha_angle, beta_angle, gamma_angle = angles

    volume = (a_length * b_length * c_length *
             sympy.sqrt(1 - sympy.cos(alpha_angle)**2 - sympy.cos(beta_angle)**2 - sympy.cos(gamma_angle)**2 + 
                       2 * sympy.cos(alpha_angle) * sympy.cos(beta_angle) * sympy.cos(gamma_angle)))
    return volume




def compute_packing_efficiency(atoms, CIF_loop_values, CIF_rad_refined_dict, cell_lengths, cell_angles_rad):
    # Initialize a dictionary for atom counts
    atom_counts = {atom: 0 for atom in atoms}

    # Unpack CIF loop values
    atom_site_types = CIF_loop_values[1]
    atom_site_symmetry_multiplicities = CIF_loop_values[2]
    
    # Calculate atom counts
    for atom_site_type, atom_site_symmetry_multiplicity in zip(atom_site_types, atom_site_symmetry_multiplicities):
        if atom_site_type in atom_counts:
            atom_counts[atom_site_type] += int(atom_site_symmetry_multiplicity)
    
    # Calculate total volume for all atom types
    total_volume = 0
    for atom, count in atom_counts.items():
        total_volume += count * (4/3) * math.pi * CIF_rad_refined_dict[atom]**3

    # Get the volume of the unit cell
    vol_of_unt_cell = get_unit_cell_volume(cell_lengths, cell_angles_rad)
    
    # Calculate packing efficiency
    packing_eff_refined = total_volume / vol_of_unt_cell
    
    return round(packing_eff_refined, 5)
