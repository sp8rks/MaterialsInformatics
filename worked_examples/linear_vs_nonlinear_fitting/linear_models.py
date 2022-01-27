from pandas import read_csv
from sklearn.linear_model import Ridge
from CBFV import composition
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

# %%

# load the dataset and name it for cbfv
df_train = read_csv('cp_train.csv')
df_test = read_csv('cp_test.csv')

rename_dict = {'Cp': 'target'}
df_train = df_train.rename(columns=rename_dict)
df_test = df_test.rename(columns=rename_dict)

#create cbfv
X_train, y_train, formulae, skipped = composition.generate_features(df_train, elem_prop='mat2vec')
X_test, y_test, formulae, skipped = composition.generate_features(df_test, elem_prop='mat2vec')

#scale and normalize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train = normalize(X_train_scaled)
X_test = normalize(X_test_scaled)

# define model with no hyperparameter tuning for now
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

#run model
y_pred = model.predict(X_test)

#evaluate and collect metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse_val = mean_squared_error(y_test, y_pred, squared=False)

#print metrics
print(f'R^2 on test set is {r2:.2f}')
print(f'MAE on test set is {mae:.2f}')
print(f'RMSE on test set is {rmse_val:.2f}')







