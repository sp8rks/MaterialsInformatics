import collections
import re
import pandas as pd
import numpy as np
import tqdm
import os
dirpath = os.getcwd()


def get_sym_dict(f, factor):
    sym_dict = collections.defaultdict(float)
    for m in re.finditer(r"([A-Z][a-z]*)\s*([-*\.\d]*)", f):
        el = m.group(1)
        amt = 1
        if m.group(2).strip() != "":
            amt = float(m.group(2))
        sym_dict[el] += amt * factor
        f = f.replace(m.group(), "", 1)
    if f.strip():
        raise CompositionError("{} is an invalid formula!".format(f))
    return sym_dict


def parse_formula(formula):
    """
    Args:
        formula (str): A string formula, e.g. Fe2O3, Li3Fe2(PO4)3
    Returns:
        Composition with that formula.
    Notes:
        In the case of Metallofullerene formula (e.g. Y3N@C80),
        the @ mark will be dropped and passed to parser.
    """
    # for Metallofullerene like "Y3N@C80"
    formula = formula.replace("@", "")
    formula = formula.replace('[', '(')
    formula = formula.replace(']', ')')
    m = re.search(r"\(([^\(\)]+)\)\s*([\.\d]*)", formula)
    if m:
        factor = 1
        if m.group(2) != "":
            factor = float(m.group(2))
        unit_sym_dict = get_sym_dict(m.group(1), factor)
        expanded_sym = "".join(["{}{}".format(el, amt)
                                for el, amt in unit_sym_dict.items()])
        expanded_formula = formula.replace(m.group(), expanded_sym)
        return parse_formula(expanded_formula)
    return get_sym_dict(formula, 1)


class CompositionError(Exception):
    """Exception class for composition errors"""
    pass


def _fractional_composition_L(formula):
    elmap = parse_formula(formula)
    elamt = {}
    natoms = 0
    for k, v in elmap.items():
        if abs(v) >= 0.05:
            elamt[k] = v
            natoms += abs(v)
    comp_frac = {}
    for key in elamt:
        comp_frac[key] = elamt[key] / natoms
    atoms, counts = list(comp_frac.keys()), list(comp_frac.values())
    return atoms, counts


def _fractional_composition(formula):
    elmap = parse_formula(formula)
    elamt = {}
    natoms = 0
    for k, v in elmap.items():
        if abs(v) >= 0.05:
            elamt[k] = v
            natoms += abs(v)
    comp_frac = {}
    for key in elamt:
        comp_frac[key] = elamt[key] / natoms
    return comp_frac


def _element_composition(formula):
    elmap = parse_formula(formula)
    elamt = {}
    natoms = 0
    for k, v in elmap.items():
        if abs(v) >= 0.05:
            elamt[k] = v
            natoms += abs(v)
    return elamt


def _assign_features(matrices, elem_info, formulae, sum_feat=False):
    formula_mat, count_mat, elem_mat, target_mat = matrices
    elem_symbols, elem_index, elem_missing = elem_info
    if sum_feat:
        sum_feats = []
    avg_feats = []
    range_feats = []
    var_feats = []
    targets = []
    formulas = []
    skipped_formula = []

    for h in tqdm.tqdm(range(len(formulae))):
        elem_list = formula_mat[h]
        target = target_mat[h]
        formula = formulae[h]

        count = np.zeros(len(elem_list))
        count[:len(count_mat[h])] = count_mat[h]
        comp_mat = np.zeros(shape=(len(elem_list), elem_mat.shape[-1]))
        i = 0
        for elem in elem_list:
            if elem in elem_missing:
                skipped_formula.append(formula)
                i = i + 1
            else:
                row = elem_index[elem_symbols.index(elem)]
                comp_mat[i, :] = elem_mat[row]
                i = i + 1
        comp_mat = comp_mat * count.reshape(len(elem_list), 1)
        if sum_feat:
            sum_feats.append(comp_mat.sum(axis=0))
        avg_feats.append(comp_mat.mean(axis=0))
        range_feats.append(np.ptp(comp_mat, axis=0))
        var_feats.append(comp_mat.var(axis=0))
        targets.append(target)
        formulas.append(formula)
    if len(skipped_formula) < 0:
        print('NOTE: Your data contain formula with exotic elements.',
              'These were skipped.')
    if sum_feat:
        feats = np.concatenate([sum_feats, avg_feats, range_feats, var_feats],
                               axis=1)
    else:
        feats = np.concatenate([avg_feats, var_feats, range_feats], axis=1)
    return feats, targets, formulas, skipped_formula


def generate_features(df, elem_prop='oliynyk', drop_duplicates=True):
    '''
    Parameters
    ----------
    df: Pandas.DataFrame()
        Two column dataframe of form:
            df.columns.values = array(['formula', 'target'], dtype=object)

    elem_prop: str:
        valid element properties:
                'oliynyk',
                'jarvis',
                'atom2vec',
                'magpie',
                'mat2vec',
                'onehot'

    Return
    ----------
    X: pd.DataFrame()
        Feature Matrix with NaN values filled using the median feature value
        for dataset
    y: pd.Series()
        Target values
    formulae: pd.Series()
        Formula associated with X and y
    '''

    if drop_duplicates:
        if df['formula'].value_counts()[0] > 1:
            df.drop_duplicates('formula', inplace=True)
            print('duplicate formula removed using default pandas function')

    all_symbols = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na',
                   'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc',
                   'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga',
                   'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb',
                   'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb',
                   'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm',
                   'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
                   'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl',
                   'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa',
                   'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md',
                   'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg',
                   'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']

    elem_props = pd.read_csv(dirpath + '/element_properties/'
                             +elem_prop+'.csv')
    elem_props.index = elem_props['element'].values
    elem_props.drop(['element'], inplace=True, axis=1)

    elem_symbols = elem_props.index.tolist()
    elem_index = np.arange(0, elem_props.shape[0], 1)
    elem_missing = list(set(all_symbols) - set(elem_symbols))

    column_names = np.concatenate(['avg_'+elem_props.columns.values,
                                   'var_'+elem_props.columns.values,
                                   'range_'+elem_props.columns.values])

    # make empty list where we will store the property value
    targets = []
    # stro formula
    formulae = []
    # add the values to the list using a for loop

    elem_mat = elem_props.values

    formula_mat = []
    count_mat = []
    target_mat = []
    print('\tprocessing input data ...'.title())
    for index in tqdm.tqdm(df.index.values):
        formula, target = df.loc[index, 'formula'], df.loc[index, 'target']
        if 'x' in formula:
            continue
        l1, l2 = _fractional_composition_L(formula)
        formula_mat.append(l1)
        count_mat.append(l2)
        target_mat.append(target)
        formulae.append(formula)
    print('\tfeaturizing compositions ...'.title())
    matrices = [formula_mat, count_mat, elem_mat, target_mat]
    elem_info = [elem_symbols, elem_index, elem_missing]
    feats, targets, formulae, skipped = _assign_features(matrices,
                                                         elem_info,
                                                         formulae)
    print('\tcreating pandas objects ...'.title())

    # split feature vectors and targets as X and y
    X = pd.DataFrame(feats, columns=column_names, index=formulae)
    y = pd.Series(targets, index=formulae, name='target')
    formulae = pd.Series(formulae, index=formulae, name='formula')

    # reset dataframe indices
    X.reset_index(drop=True, inplace=True)
    y.reset_index(drop=True, inplace=True)
    formulae.reset_index(drop=True, inplace=True)

    # drop elements that aren't included in the elmenetal properties list.
    # These will be returned as feature rows completely full of Nan values.
    X.dropna(inplace=True, how='all')
    y = y.loc[X.index]
    formulae = formulae.loc[X.index]

    # get the column names
    cols = X.columns.values
    # find the mean value of each column
    median_values = X[cols].median()
    # fill the missing values in each column with the columns mean value
    X[cols] = X[cols].fillna(median_values.iloc[0])
    return X, y, formulae, skipped
