# -*-coding:utf-8-*-
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import classification_report
# from sklearn.metrics import roc_auc_score

dataSet = load_iris()
data = dataSet['data'] # data
label = dataSet['target'] # data's lable
feature = dataSet['feature_names'] # feature's name
target = dataSet['target_names'] # name of the lable
print(target)
pd.set_option('display.max_columns', None)
df = pd.DataFrame(np.column_stack((data,label)),columns = np.append(feature,'label'))
print(df.head())# check the top 5 row

print(df.isnull().sum(axis=0).sort_values(ascending=False)/float(len(df)))# see the rate of missed value

# In sklearn's preprocessing, there is a function Imputer() to handle missing values.
# It provides median, mean, and mode to fill in missing values.
# Fortunately, there are no missing values ​​in our data set.

print(df['label'].value_counts()) # check the rate of each lable

StandardScaler().fit_transform(data)   # z score standard

# use OvR doing multiple logistic regression
ss = ShuffleSplit(n_splits = 1,test_size= 0.2) # seperate the dataset, 80% as training set
for tr,te in ss.split(data,label):
    xr = data[tr]
    xe = data[te]
    yr = label[tr]
    ye = label[te]
    clf = LogisticRegression(solver = 'lbfgs',multi_class = 'multinomial')
    clf.fit(xr,yr)
    predict = clf.predict(xe)
    print(classification_report(ye, predict))
# OvR regards multiple logistic regression as a binary logistic regression.
# The specific method is to select one class as a positive example each time,
# and the other categories as a negative case, and then do binary logistic regression
# to obtain the classification model of the class. Finally, multiple binary regression models are derived.
# The classification results are obtained according to the scores of each category.
