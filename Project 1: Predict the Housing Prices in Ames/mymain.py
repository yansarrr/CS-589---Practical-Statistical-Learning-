###########################################
# Step 0: Load necessary libraries
#
import numpy as np
import pandas as pd
from scipy.stats import mstats
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge, RidgeCV


def preprocess_features(features, linear=False):
    # drop known bad features
    if linear:
        features = features.iloc[:, ~features.columns.isin(['Street', 'Utilities', 'Condition_2', 'Roof_Matl', 'Heating', 'Pool_QC', 'Misc_Feature', 'Low_Qual_Fin_SF', 'Pool_Area', 'Longitude','Latitude'])]

    # drop features with few unique values
    # numerical_features = features.select_dtypes([np.number])
    # categorical_features = features.columns[~features.columns.isin(numerical_features.columns)].values
    # for categorical_feature in categorical_features:
    #     if not feature_contains_enough_uniques(features, categorical_feature, 4):
    #         print('removing', categorical_feature)
    #         features = features.iloc[:, ~features.columns.isin([categorical_feature])]

    # encode categorical features
    features = pd.get_dummies(features)

    # fill NaNs with 0
    features['Garage_Yr_Blt'] = features['Garage_Yr_Blt'].fillna(0)

    # winsorize
    if linear:
        for feat in ("Lot_Frontage", "Lot_Area", "Mas_Vnr_Area", "BsmtFin_SF_2", "Bsmt_Unf_SF", "Total_Bsmt_SF", "Second_Flr_SF", 'First_Flr_SF', "Gr_Liv_Area", "Garage_Area", "Wood_Deck_SF", "Open_Porch_SF", "Enclosed_Porch", "Three_season_porch", "Screen_Porch", "Misc_Val"):
            features[[feat]] = mstats.winsorize(features[[feat]].values, limits=[0.05, 0.05])

    return features


def feature_contains_enough_uniques(features, feature_name, unique_threshold):
    if len(features[feature_name].unique()) < unique_threshold:
        return False
    return True


# remove whichever features don't exist in both DataFrames
def balance_features(train_features, test_features):
    # print('train_features.s', train_features.shape)
    # print('test_features.s', test_features.shape)

    not_in_train = test_features.columns[~test_features.columns.isin(train_features.columns)].values
    not_in_test = train_features.columns[~train_features.columns.isin(test_features.columns)].values

    missing_values = list(not_in_train)
    missing_values.extend(not_in_test)
    # print('missing values', missing_values)

    train_features = train_features.iloc[:, ~train_features.columns.isin(missing_values)]
    test_features = test_features.iloc[:, ~test_features.columns.isin(missing_values)]

    return train_features, test_features


def preprocess_response(response):
    response = np.log(response)
    return response

###########################################
# Step 1: Preprocess training data
#         and fit two models
#
train = pd.read_csv('train.csv')
X_train = train.iloc[:, train.columns != 'Sale_Price']
y_train = train.iloc[:, train.columns == 'Sale_Price']

y_train = preprocess_response(y_train)
linear_X_train = preprocess_features(X_train, linear=True)
tree_X_train = preprocess_features(X_train, linear=False)


###########################################
# Step 2: Preprocess test data
#         and output predictions into two files
#
X_test = pd.read_csv('test.csv')
linear_X_test = preprocess_features(X_test, linear=True)
tree_X_test = preprocess_features(X_test, linear=False)
linear_X_train, linear_X_test = balance_features(linear_X_train, linear_X_test)
tree_X_train, tree_X_test = balance_features(tree_X_train, tree_X_test)

# linear regressor
ridge_cv = RidgeCV(alphas=np.linspace(1, 40, num=40), scoring='neg_mean_squared_error', normalize=False)
ridge_cv.fit(linear_X_train, y_train)
linear_regressor = Ridge(alpha=ridge_cv.alpha_)
linear_regressor.fit(linear_X_train, y_train)
linear_y_pred = linear_regressor.predict(linear_X_test).reshape(-1, 1)

# tree regressor
tree_regressor = GradientBoostingRegressor(
    n_estimators=300,
    max_depth=5,
    min_samples_leaf=1,
    random_state=0,
    learning_rate=0.1
)
tree_regressor.fit(tree_X_train, y_train)
tree_y_pred = tree_regressor.predict(tree_X_test).reshape(-1, 1)

# output
linear_df = pd.DataFrame()
linear_df['PID'] = X_test['PID']
linear_df['Sale_Price'] = np.exp(linear_y_pred)

tree_df = pd.DataFrame()
tree_df['PID'] = X_test['PID']
tree_df['Sale_Price'] = np.exp(tree_y_pred)

np.savetxt('mysubmission1.txt', linear_df, header='PID, Sale_Price', fmt='%d, %1.1f', comments=' ')
np.savetxt('mysubmission2.txt', tree_df, header='PID, Sale_Price', fmt='%d, %1.1f', comments=' ')
