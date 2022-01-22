from CBFV import composition
import pandas as pd

data = [['Si1O2', 10], ['Al2O3', 15], ['Hf1C1Zr1', 14]]

df = pd.DataFrame(data, columns = ['formula', 'target'])
X, y, formulae, skipped = composition.generate_features(df, 
    elem_prop='mat2vec')
