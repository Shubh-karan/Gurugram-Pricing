import pandas as pd
from sklearn.ensemble import RandomForestClassifier

data = pd.read_excel("C:\\Users\\LENOVO\\OneDrive\\Data Science\\Scikit Learn\\Practical SKLearn\\iris.xlsx")

x = data.iloc[:,:-1]
y = data.iloc[:,-1]

model = RandomForestClassifier()

model.fit(x,y)
print("Enter details below: ")
sepal_length = float(input("Enter sepal length: "))
sepal_width = float(input("Enter sepal width: "))
petal_length = float(input("Enter petal length: "))
petal_width = float(input("Enter petal width: "))
prediction = model.predict([[sepal_length,sepal_width,petal_length,petal_width]])
print(prediction)
