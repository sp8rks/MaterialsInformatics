import click
import os
import time
from click import style
import pandas as pd
import util.data as db
import util.dataframe as df
from fractions import Fraction
import preprocess.cif_parser as cif_parser 
import util.folder as folder
import util.log as log
import preprocess.supercell as supercell
import featurizer.interatomic as interatomic_featurizer
import featurizer.environment_wyckoff as env_wychoff_featurizer
import featurizer.environment_binary as env_featurizer_binary
import featurizer.environment_ternary as env_featurizer_ternary
import featurizer.environment_dataframe as env_dataframe
import featurizer.coordinate_number_dataframe as coordinate_number_dataframe
from collections import defaultdict

def round_df(df):
    numeric_cols = df.select_dtypes(include=['float64']).columns
    df[numeric_cols] = df[numeric_cols].round(4)
    return df

def print_atom_pairs_info(atom_pairs_info_dict):
    for label, data in atom_pairs_info_dict.items():
        print("Pair info:")
        for pair_info in data["pair_info"][:10]:
            print(pair_info)

def main():
    # Configurations and initializations
    radii_data = db.get_radii_data()
    loop_tags = cif_parser.get_loop_tags()

    # Get the default atom count or prompt the user
    skip_based_on_atoms = click.confirm('Do you want to skip any CIF files based on the number of unique atoms in the supercell (Default: N)?')

    if skip_based_on_atoms:
        MAX_ATOMS_COUNT = click.prompt('Enter the threshold for the maximum number of atoms in the supercell. Files with atoms exceeding this count will be skipped', type=int)
    else:
        MAX_ATOMS_COUNT = float('inf')  # A large number to essentially disable skipping

    # Directory fetch
    script_directory = os.path.dirname(os.path.abspath(__file__))    
    cif_folder_directory = folder.choose_CIF_directory(script_directory)

    # Get a list of all .cif files in the chosen directory
    files_lst = [os.path.join(cif_folder_directory, file) for file in os.listdir(cif_folder_directory) if file.endswith('.cif')]
    total_files = len(files_lst)
    
    # Number of files
    total_files = len(files_lst)

    running_total_time = 0  # initialize running total execution time

    property_file = './element_database/element_properties_for_ML-my elements.xlsx'
    xl = pd.read_excel(property_file) # contains CIF, Pauling radius
    

    # Initialize DataFrames to store results
    interatomic_binary_df = interatomic_ternary_df = interatomic_universal_df = pd.DataFrame()
    atomic_env_wyckoff_binary_df = atomic_env_wyckoff_ternary_df = atomic_env_wyckoff_universal_df = pd.DataFrame()
    atomic_env_binary_df = atomic_env_ternary_df = pd.DataFrame()
    featurizer_log_entries = []

    coordinate_number_binary_df = pd.DataFrame()
    coordinate_number_binary_max_df = pd.DataFrame()
    coordinate_number_binary_min_df = pd.DataFrame()
    coordinate_number_binary_avg_df = pd.DataFrame()

    coordinate_number_ternary_df = pd.DataFrame()
    coordinate_number_ternary_max_df = pd.DataFrame()
    coordinate_number_ternary_min_df = pd.DataFrame()
    coordinate_number_ternary_avg_df = pd.DataFrame()

    
    num_files_processed = 0
    # Loop through each CIF file
    for idx, filename in enumerate(files_lst, start=1):
        start_time = time.time()
        filename_base = os.path.basename(filename)
        print(f"\n({idx}/{total_files}) Processing {filename_base}...")

        isBinary = isTernary = False
        A = B = R = M = X = ''

        if not cif_parser.valid_cif(filename):
            print(f"\n({idx}/{total_files}) Rejecting {filename_base}... No structure found.")
            continue
        
        cif_parser.preprocess_cif_file(filename)
        cif_parser.take_care_of_atomic_site(filename)

        CIF_block = cif_parser.get_CIF_block(filename)
        CIF_id = CIF_block.name
        cell_lengths, cell_angles_rad = supercell.process_cell_data(CIF_block)
        CIF_loop_values = cif_parser.get_loop_values(CIF_block, loop_tags)
        all_coords_list = supercell.get_coords_list(CIF_block, CIF_loop_values)
        all_points, unique_labels, unique_atoms_tuple = supercell.get_points_and_labels(all_coords_list, CIF_loop_values)
        
        # Check the number of atoms in the supercell to skip the file
        if cif_parser.exceeds_atom_count_limit(all_points, MAX_ATOMS_COUNT):
            click.echo(style(f"Skipped - {filename_base} has {len(all_points)} atoms", fg="yellow"))
            continue

        num_files_processed += 1
        unique_atoms_tuple, num_of_unique_atoms, formula_string = cif_parser.extract_formula_and_atoms(CIF_block)
        atomic_pair_list = supercell.get_atomic_pair_list(all_points, cell_lengths, cell_angles_rad)
        atom_pair_info_dict = supercell.get_atom_pair_info_dict(unique_labels, atomic_pair_list)
        
        # Check for the type of compound: unary, binary, or ternary
        isBinary = num_of_unique_atoms == 2
        isTernary = num_of_unique_atoms == 3

        CIF_data = (CIF_id, cell_lengths, cell_angles_rad, CIF_loop_values, formula_string)
            
        # Initialize variables for atomic enviornment
        atom_counts = defaultdict(int)
        unique_shortest_labels, atom_counts = env_featurizer_binary.get_unique_shortest_labels(atom_pair_info_dict, unique_labels, cif_parser)
        
        if isBinary:
            A, B = unique_atoms_tuple[0][0], unique_atoms_tuple[1][0]
            interatomic_binary_df, interatomic_universal_df = interatomic_featurizer.get_interatomic_binary_df(filename,
                                                        interatomic_binary_df,
                                                        interatomic_universal_df,
                                                        all_points,
                                                        unique_atoms_tuple,
                                                        atomic_pair_list,
                                                        CIF_data,
                                                        radii_data)
            
            atomic_env_wyckoff_binary_df, atomic_env_wyckoff_universal_df = env_wychoff_featurizer.get_env_wychoff_binary_df(filename,
                                                        xl,
                                                        atomic_env_wyckoff_binary_df,
                                                        atomic_env_wyckoff_universal_df,
                                                        unique_atoms_tuple,
                                                        CIF_loop_values,
                                                        radii_data,
                                                        CIF_data,
                                                        atomic_pair_list)
                        

            atomic_env_binary_df = env_dataframe.get_env_binary_df(
                                                        atomic_env_binary_df,
                                                        unique_atoms_tuple,
                                                        unique_labels,
                                                        unique_shortest_labels,
                                                        atom_pair_info_dict,
                                                        atom_counts,
                                                        CIF_data)
            
            coordinate_number_binary_df = coordinate_number_dataframe.get_coordinate_number_binary_df(
                                                        isBinary,
                                                        coordinate_number_binary_df,
                                                        unique_atoms_tuple,
                                                        unique_labels,
                                                        atomic_pair_list,  # Added the missing comma
                                                        atom_pair_info_dict,
                                                        CIF_data,
                                                        radii_data)
                        
        if isTernary:
            R, M, X = unique_atoms_tuple[0][0], unique_atoms_tuple[1][0], unique_atoms_tuple[2][0]
            interatomic_ternary_df, interatomic_universal_df = interatomic_featurizer.get_interatomic_ternary_df(filename,
                                                        interatomic_ternary_df,
                                                        interatomic_universal_df,
                                                        all_points,
                                                        unique_atoms_tuple,
                                                        atomic_pair_list,
                                                        CIF_data,
                                                        radii_data)

            atomic_env_wyckoff_ternary_df, atomic_env_wyckoff_universal_df = env_wychoff_featurizer.get_env_wychoff_ternary_df(filename,
                                                        xl,
                                                        atomic_env_wyckoff_ternary_df,
                                                        atomic_env_wyckoff_universal_df,
                                                        unique_atoms_tuple,
                                                        CIF_loop_values,
                                                        radii_data,
                                                        CIF_data,
                                                        atomic_pair_list)

            atomic_env_ternary_df = env_dataframe.get_env_ternary_df(
                                                        atomic_env_ternary_df,
                                                        unique_atoms_tuple,
                                                        unique_labels,
                                                        unique_shortest_labels,
                                                        atom_pair_info_dict,
                                                        atom_counts,
                                                        CIF_data)      
            
            coordinate_number_ternary_df = coordinate_number_dataframe.get_coordinate_number_ternary_df(
                                                        isBinary,
                                                        coordinate_number_ternary_df,
                                                        unique_atoms_tuple,
                                                        unique_labels,
                                                        atomic_pair_list,  # Added the missing comma
                                                        atom_pair_info_dict,
                                                        CIF_data,
                                                        radii_data)
           

        end_time = time.time()
        execution_time = end_time - start_time
        running_total_time += execution_time
        click.echo(style(f"{execution_time:.2f}s to process {len(all_points)} atoms (total time {running_total_time:.2f}s)", fg="green"))

        featurizer_log_entries.append({
            "Filename": filename_base,
            "CIF": CIF_id,
            "Compound": formula_string,
            "Number of atoms": len(all_points),
            "Execution time (s)": execution_time,
            "Total_time (s)": running_total_time
        })

    featurizer_log_df = pd.DataFrame(featurizer_log_entries)
    featurizer_log_df = featurizer_log_df.round(3)

    if num_files_processed != 0:
        cols_to_keep = ['CIF_id', 'Compound', 'Central atom']
        click.echo(style(f"Saving csv files in the csv folder", fg="blue"))
        atomic_env_wyckoff_universal_df = df.join_columns_with_comma(atomic_env_wyckoff_universal_df)
        if not coordinate_number_binary_df.empty:
            # Save the original DataFrame to CSV before any modification
            binary_non_numeric_cols_to_remove = coordinate_number_binary_df.select_dtypes(include=['object']).columns.difference(cols_to_keep)
            coordinate_number_binary_df = coordinate_number_binary_df.drop(binary_non_numeric_cols_to_remove, axis=1)                        
            atomic_env_wyckoff_binary_df = df.wyckoff_mapping_to_number_binary(atomic_env_wyckoff_binary_df)
            coordinate_number_binary_avg_df = coordinate_number_binary_df.groupby(cols_to_keep).mean().reset_index()
            coordinate_number_binary_min_df = coordinate_number_binary_df.groupby(cols_to_keep).min().reset_index()
            coordinate_number_binary_max_df = coordinate_number_binary_df.groupby(cols_to_keep).max().reset_index()
            folder.save_to_csv_directory(cif_folder_directory, round_df(coordinate_number_binary_df), "coordination_number_binary_all")
            folder.save_to_csv_directory(cif_folder_directory, round_df(coordinate_number_binary_avg_df), "coordination_number_binary_avg")
            folder.save_to_csv_directory(cif_folder_directory, round_df(coordinate_number_binary_min_df), "coordination_number_binary_min")
            folder.save_to_csv_directory(cif_folder_directory, round_df(coordinate_number_binary_max_df), "coordination_number_binary_max")
            folder.save_to_csv_directory(cif_folder_directory, round_df(interatomic_binary_df), "interatomic_features_binary")
            folder.save_to_csv_directory(cif_folder_directory, round_df(atomic_env_binary_df), "atomic_environment_features_binary")
            folder.save_to_csv_directory(cif_folder_directory, round_df(atomic_env_wyckoff_binary_df), "atomic_environment_wyckoff_multiplicity_features_binary")


        if not coordinate_number_ternary_df.empty:
            ternary_non_numeric_cols_to_remove = coordinate_number_ternary_df.select_dtypes(include=['object']).columns.difference(cols_to_keep)
            coordinate_number_ternary_df = coordinate_number_ternary_df.drop(ternary_non_numeric_cols_to_remove, axis=1)   
            coordinate_number_ternary_avg_df = coordinate_number_ternary_df.groupby(cols_to_keep).mean().reset_index()
            coordinate_number_ternary_min_df = coordinate_number_ternary_df.groupby(cols_to_keep).min().reset_index()
            coordinate_number_ternary_max_df = coordinate_number_ternary_df.groupby(cols_to_keep).max().reset_index()
            atomic_env_wyckoff_ternary_df = df.wyckoff_mapping_to_number_ternary(atomic_env_wyckoff_ternary_df)            
            folder.save_to_csv_directory(cif_folder_directory, round_df(coordinate_number_ternary_df), "coordination_number_ternary_all")
            folder.save_to_csv_directory(cif_folder_directory, round_df(coordinate_number_ternary_avg_df), "coordination_number_ternary_avg")
            folder.save_to_csv_directory(cif_folder_directory, round_df(coordinate_number_ternary_min_df), "coordination_number_ternary_min")
            folder.save_to_csv_directory(cif_folder_directory, round_df(coordinate_number_ternary_max_df), "coordination_number_ternary_max")
            folder.save_to_csv_directory(cif_folder_directory, round_df(interatomic_ternary_df), "interatomic_features_ternary")
            folder.save_to_csv_directory(cif_folder_directory, round_df(atomic_env_ternary_df), "atomic_environment_features_ternary")
            folder.save_to_csv_directory(cif_folder_directory, round_df(atomic_env_wyckoff_ternary_df), "atomic_environment_wyckoff_multiplicity_features_tenary")
 
        
        folder.save_to_csv_directory(cif_folder_directory, round_df(interatomic_universal_df), "interatomic_features_universal")
        folder.save_to_csv_directory(cif_folder_directory, round_df(atomic_env_wyckoff_universal_df), "atomic_environment_wyckoff_multiplicity_features_universal")
        folder.save_to_csv_directory(cif_folder_directory, round_df(featurizer_log_df), "featurizer_log")
    
if __name__ == "__main__":
    main()