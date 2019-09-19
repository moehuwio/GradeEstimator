from sklearn import linear_model
import matplotlib.pyplot as pyplot
from matplotlib import style
import numpy as np
import pandas as pd
import sklearn
import pickle

data = pd.read_csv("student-mat.csv", sep=";")
data = data[["G1","G2","G3","studytime","failures","absences"]]

predict = "G3"

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

xTrain, xTest, yTrain, yTest = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
best = 0
"""""
for _ in range(20):

    xTrain, xTest, yTrain, yTest = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
    linear = linear_model.LinearRegression()
    linear.fit(xTrain, yTrain)

    acc = linear.score(xTest, yTest)
    print(acc) 

    if acc > best:
        best = acc
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f) 
            """


pickleIn = open("studentmodel.pickle", "rb")
linear = pickle.load(pickleIn)

print("coeh \n", linear.coef_)
print("int \n", linear.intercept_)

predictions = linear.predict(xTest)

for x in range(len(predictions)):
    print(predictions[x], xTest[x], yTest[x])

p = "G2"
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()
