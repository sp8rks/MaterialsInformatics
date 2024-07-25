def join_columns_with_comma(df):

    columns = [
        'Mendeleev_number_of_elements_with_lowest_wyckoff',
        'valence_e_of_elements_with_lowest_wyckoff',
        'lowest_wyckoff_elements',
        'CIF_rad_of_elements_with_lowest_wyckoff',
        'CIF_rad_refined_of_elements_with_lowest_wyckoff',
        'Pauling_rad_of_elements_with_lowest_wyckoff',
        'Pauling_EN_of_elements_with_lowest_wyckoff',
        'MB_EN_of_elements_with_lowest_wyckoff'
    ]

    for column in columns:
        df[column] = df[column].apply(lambda x: ', '.join(map(str, x)))

    return df


def wyckoff_mapping_to_number_binary(df):
    wyckoff_mapping = {'a': '1', 'b': '2', 'c': '3', 'd': '4', 'e': '5', 'f': '6', 'g': '7', 'h': '8', 'i': '9', 'j': '10'}
    columns_to_map_binary = ['A_lowest_wyckoff_label', 'B_lowest_wyckoff_label']

    for col in columns_to_map_binary:
        df[col] = df[col].replace(wyckoff_mapping)

    return df


def wyckoff_mapping_to_number_ternary(df):
    wyckoff_mapping = {'a': '1', 'b': '2', 'c': '3', 'd': '4', 'e': '5', 'f': '6', 'g': '7', 'h': '8', 'i': '9', 'j': '10'}
    columns_to_map_ternary = ['R_lowest_wyckoff_label', 'M_lowest_wyckoff_label', 'X_lowest_wyckoff_label']

    for col in columns_to_map_ternary:
        df[col] = df[col].replace(wyckoff_mapping)

    return df

