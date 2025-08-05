import pandas as pd
from sklearn.preprocessing import normalize, StandardScaler
from cbfv.composition import generate_features as gf


class Featurize():
    def __init__(self, data_path, scale=True, save=True):
        self.data_path = data_path
        self.scale = scale
        self.save = save
        self.get_xy()

    def get_xy(self):
        df = pd.read_csv(self.data_path)
        df.columns = ['formula', 'target']
        self.df = df
        X, y, formulae, skipped = gf(df, elem_prop='oliynyk')
        self.columns = X.columns
        self.X = X
        self.y = y
        if self.scale:
            self.scaler = StandardScaler()
            self.X = normalize(self.scaler.fit_transform(X))
        self.formula = formulae
        self.skipped = skipped


train_file = 'data/ael_bulk_modulus_vrh/train.csv'

# greate a model (featurization of train data here)
feats = Featurize(train_file, scale=True, save=True)

# set training and predicted data as pandas dataframes (can be saved to csv)
X = pd.DataFrame(feats.X, columns=feats.columns)
X.to_csv('featurized_data/X.csv', index=False)
