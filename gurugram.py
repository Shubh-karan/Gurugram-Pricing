#importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

#splitting data into test and train sets

def shuffle_and_split(data, test_ratio):
    np.random.seed(42) #so that we will get same random output
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data)*test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

data = pd.read_csv(r"Scikit Learn\Practical SKLearn\housing.csv")

#creating 2 sets

train_set, test_set = shuffle_and_split(data,0.2)

data["income_cat"] = pd.cut(data["median_income"],bins=[0,3,6,9,12,np.inf], labels=[1,2,3,4,5]) #creating bins to make categories of median income so that all the values of this cloumn come in whole test and train set

# data["income_cat"].value_counts().sort_index().plot.bar(rot=0, grid=True)  
# plt.title("Income Categories Distribution") 
# plt.xlabel("Income Category")
# plt.ylabel("Number of Instances")
# plt.show()

#working on stratified shuffle split

split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)

for train_index,test_index in split.split(data,data["income_cat"]):
    strat_train_set = data.loc[train_index]
    strat_test_set = data.loc[test_index]

#dropping income_cat column from both test and train set
for i in (strat_train_set,strat_test_set):
    i.drop("income_cat", axis=1, inplace=True)

df = strat_train_set.copy()

#plot of houses wrt their values

# df.plot(kind="scatter", x = "longitude", y = "latitude", grid = True, cmap = "jet", c = "median_house_value")
# plt.show()

#checking for the correlation using scatter matrix
# df.drop("ocean_proximity", axis=1, inplace=True)
# corr_matrix = df.corr()

# This is a quick way to see which features have the strongest positive or negative correlation with median_house_value.
# Values close to +1 → strong positive correlation
# Values close to -1 → strong negative correlation
# Values near 0 → weak or no correlation

# print(corr_matrix["median_house_value"].sort_values(ascending=False))
# attributes = ["housing_median_age", "median_income", "median_house_value"]
# scatter_matrix(df[attributes],figsize=(12,8))
# plt.show()

#starting data preprocessing here

#here using print(df.describe()) in the count section of total_bedrooms, you can see count is less there than other means there are null values, we need to tackle these null values as well
housing_labels = df["median_house_value"].copy()
housing_features = df.drop("median_house_value", axis=1)
#now using impute we will handle missing values
imputer = SimpleImputer(strategy="median")
#it takes noly number values from data not strings etc as here we have ocean_proximity
housing_num = housing_features.select_dtypes(include = [np.number])
imputed_x = imputer.fit_transform(housing_num)
housing_new = pd.DataFrame(imputed_x, columns=housing_num.columns, index=housing_num.index)
housing_new["ocean_proximity"] = df["ocean_proximity"]
# here we are trying or showing how we can work with ordinal encoder but are not working with that as of now as t isn't suits here
# ordinal_encoder = OrdinalEncoder()
# housing_encoded = ordinal_encoder.fit_transform(housing_new)
# housing_cat = pd.DataFrame(housing_encoded,columns=housing_new.columns, index=housing_new.index)
# here we are going to use OneHotEncoder so that we get values as matrix
onehot_encoder = OneHotEncoder()
housing_encoded = onehot_encoder.fit_transform(housing_new[["ocean_proximity"]])

housing_cat = pd.DataFrame(
    housing_encoded.toarray(),
    columns=['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'], #onehot_encoder.get_feature_names_out(["ocean_proximity"]) we can also use this funtion so that it will give automatic column names to the column
    index=housing_new.index
)
df = pd.concat([housing_new,housing_cat],axis=1)
df = df.drop("ocean_proximity", axis=1)
#working on scaling data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

#here below we are going to check check which model will fit best here
#Linear Regressor
linear_mod = LinearRegression()
linear_mod.fit(df_scaled,housing_labels)
linear_predicts = linear_mod.predict(df_scaled)
linear_rmse = -cross_val_score(linear_mod,df_scaled,housing_labels,scoring="neg_root_mean_squared_error", cv=10)
print(f"Root mean sqaure error for Linear Regressor is {pd.Series(linear_rmse).mean()}")

#Decision Tree Regressor
deciontree_mod = DecisionTreeRegressor()
deciontree_mod.fit(df_scaled,housing_labels)
deciontree_predicts = deciontree_mod.predict(df_scaled)
deciontree_rmse = -cross_val_score(deciontree_mod,df_scaled,housing_labels,scoring="neg_root_mean_squared_error", cv=10)
print(f"Root mean sqaure error for Decision Treee Regressor is {pd.Series(deciontree_rmse).mean()}")

#random forest regressor
rfr_mod = RandomForestRegressor()
rfr_mod.fit(df_scaled,housing_labels)
rfr_predicts = rfr_mod.predict(df_scaled)
rfr_rmse = -cross_val_score(rfr_mod,df_scaled,housing_labels,scoring="neg_root_mean_squared_error", cv=10)
print(f"Root mean sqaure error for Random Forest Regressor is {pd.Series(rfr_rmse).mean()}")