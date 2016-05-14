import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.tools.plotting import andrews_curves
from pandas.tools.plotting import radviz

# Read the dataset
iris = pd.read_csv("Iris.csv")

# Drop the false feature
iris = iris.drop("Id",axis=1)

# Numerically index the labels 
speciesMap = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
iris = iris.replace({'Species': speciesMap})

# Final features and labels
x = iris.drop('Species',axis=1)
yTrue = iris['Species']

# Deploy a classifier
# from sklearn.tree import DecisionTreeClassifier
# clf = DecisionTreeClassifier()

# It is hard to lose on Iris no matter what we choose
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier()

# Deploy cross-validation
from sklearn.cross_validation import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(
    x, yTrue, train_size=0.15, random_state=42
)

# Make some predictions
clf.fit(xTrain,yTrain)
yPred = clf.predict(xTest)

# Scoring via precision and recall
from sklearn.metrics import precision_recall_fscore_support
precision, recall, fscore, _ = precision_recall_fscore_support(
    yTest,yPred,average='macro'
)

# Precision and recall
print "Precision: ", precision
print "Recall: ", recall
print "F-Score: ", fscore

# 5-fold cross-validation score
from sklearn.cross_validation import cross_val_score
print cross_val_score(clf,x,yTrue,cv=5)


'''
setosa     = iris[iris['Species'] == 'Iris-setosa']
versicolor = iris[iris['Species'] == 'Iris-versicolor']
virginica  = iris[iris['Species'] == 'Iris-virginica']

# xAxis = "SepalLengthCm"
# yAxis = "SepalWidthCm"
xAxis = "PetalLengthCm"
yAxis = "PetalWidthCm"

# andrews_curves(iris.drop("Id",axis=1), "Species")
# radviz(iris.drop("Id",axis=1), "Species")
# plt.show()

from sklearn.cross_validation import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(
        x, yTrue, train_size = 0.80, random_state = 42
) 

clf.fit(xTrain,yTrain)
yPred = clf.transform(xTest)

from sklearn.metrics import precision_recall_fscore_support

# print precision_recall_fscore_support(yTest,yPred,average='macro')
'''

'''
# Plot the two of the attributes
p = setosa.plot(
    kind="scatter", 
    x=xAxis, 
    y=yAxis, 
    color="red", 
    label="setosa"
)

versicolor.plot(
    kind="scatter", 
    x=xAxis, 
    y=yAxis, 
    color="green", 
    label="versicolor",
    ax=p
)

virginica.plot(
        kind="scatter", 
        x=xAxis, 
        y=yAxis, 
        color="blue", 
        label="virginica",
        ax=p
)
plt.show()
'''

