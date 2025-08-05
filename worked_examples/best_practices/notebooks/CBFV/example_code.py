import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import normalize, StandardScaler

from cbfv.composition import generate_features as gf


class Model():
    def __init__(self, data_path, scale=True, save=True):
        self.params = {'C': 10, 'gamma': 1}
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

    def run_gridsearch(self):
        print('running grid search')
        list_C = [0.1, 1, 10, 100]
        list_gamma = list_C
        parameters = {'C': list_C, 'gamma': list_gamma}
        clf = GridSearchCV(SVR(), parameters, cv=3)
        clf.fit(self.X, self.y)
        self.clf = clf
        self.params = clf.best_params_
        print('best score: {:0.3f}'.format(clf.best_score_))

    def fit(self):
        print('fitting model to training data')
        self.model = SVR(**self.params)
        self.model.fit(self.X, self.y)

    def predict(self, data_path, save_name='predictions.csv'):
        df = pd.read_csv(data_path)
        X, y, formulae, skipped = gf(df, elem_prop='oliynyk')
        self.X_pred = X
        if self.scale:
            self.X_pred = normalize(self.scaler.transform(self.X_pred))
        self.y_pred = pd.Series(self.model.predict(self.X_pred),
                                index=formulae.index)
        df_pred = pd.concat([formulae, y, self.y_pred], axis=1)
        df_pred.columns = ['formula', 'actual', 'prediction']
        df_pred.to_csv(save_name, index=False)


# Define the data we want to featurize for ML uses
train_file = 'data/ael_bulk_modulus_vrh/train.csv'
val_file = 'data/ael_bulk_modulus_vrh/val.csv'

# greate a model (featurization of train data here)
model = Model(train_file, scale=True, save=True)

# run grid search for better results
model.run_gridsearch()

# fit the model to training data
model.fit()
# generate predictions for input file. (featurization of predictions here)
model.predict(val_file)

# set training and predicted data as pandas dataframes (can be saved to csv)
X_train = pd.DataFrame(model.X, columns=model.columns)
X_test = pd.DataFrame(model.X_pred, columns=model.columns)
