#importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix

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
df.drop("ocean_proximity", axis=1, inplace=True)
corr_matrix = df.corr()

# This is a quick way to see which features have the strongest positive or negative correlation with median_house_value.
# Values close to +1 → strong positive correlation
# Values close to -1 → strong negative correlation
# Values near 0 → weak or no correlation

print(corr_matrix["median_house_value"].sort_values(ascending=False))
attributes = ["housing_median_age", "median_income", "median_house_value"]
scatter_matrix(df[attributes],figsize=(12,8))
plt.show()