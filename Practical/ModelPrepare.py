import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
original = pd.read_csv('kidney_disease.csv')
ds = pd.read_csv('kidney_disease.csv')
ds.drop('id', inplace = True, axis = 1)

#dataset.describe()
ds.head()
type = ds.dtypes

''' this is a simple explaination code for the process of replacing with mode
mode = ds['bp'].mode()
mode[0]
ds['bp'].fillna(ds['bp'].mode()[0], inplace = True)
'''
ds['classification'].unique()
ds['classification'] = ds['classification'].replace('ckd\t', 'ckd')
ds['classification'].unique()

ds['dm'].unique()
ds['dm'] = ds['dm'].replace(' yes', 'yes')
ds['dm'] = ds['dm'].replace('\tyes', 'yes')
ds['dm'] = ds['dm'].replace('\tno', 'no')
ds['dm'].unique()

ds['pc'].unique()
ds['pcc'].unique()
ds['ba'].unique()
for i in ds.columns:
    print(ds)


ds.isna().sum()
###################### MANIPULATING DATA TO MAKE IT CAPABLE FOR LEARNING
#replacing nulls.
def nullifier(df):
    for i in df.columns:
        df[i].fillna(df[i].mode()[0], inplace = True)
nullifier(ds)


#encoding all the string rows to make operations on it.
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

def classify(df):
    for i in df.columns:
        if df[i].dtype == 'object':
            df[i] = le.fit_transform(df[i])
classify(ds)

ds['dm'].unique()
original['dm'].unique()
original['cad'].unique()
###################### SOME PLOTS
import seaborn as sns

mask = np.triu(np.ones_like(ds.corr()))

sns.heatmap(ds.corr()[['classification']].sort_values(by='classification', ascending=False), annot=True)

ds.corr().style.background_gradient()

sns.pairplot(ds,y_vars=['classification'])

###################### STARTING THE MODEL
x = ds.iloc[:,0:-1].values
y = ds.iloc[:,-1].values




# MULTI LINEAR REGRESSION
from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x,y , test_size=0.25 , random_state = 43)

#scaling train data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x_train ,y_train)

y_pred = lin_reg.predict(x_test)

from sklearn.metrics import r2_score
linear_score = r2_score(y_test,y_pred) #result = 63.7%


# POLYNOMIAL REGRESSION
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=3)
x_poly = poly_reg.fit_transform(x)
poly_reg.fit(x_poly , y)
lin_reg.fit(x_poly , y)
y_polypredict = lin_reg.predict(poly_reg.fit_transform(x))

from sklearn.metrics import r2_score
poly_score = r2_score(y_polypredict , y) # for polynomial degree 3 it gives 100% accuracy
#this is a case of overfitting so we'll ignore it

#LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(max_iter=500)
log_reg.fit(x_train, y_train)
y_log_pred = log_reg.predict(x_test)
log_score = r2_score(y_test, y_log_pred) #result = 87.5%

#SVM
from sklearn import svm
clf = svm.SVC(kernel="linear")
clf.fit(x_train, y_train)
y_svm_pred = clf.predict(x_test)
svm_score = r2_score(y_test, y_svm_pred) #result = 87.5%

#KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
#Testing to see optimal k-value
# Creating odd list K for KNN
neighbors = list(range(1,50))
# empty list that will hold cv scores
cv_scores = [ ]

#perform 10-fold cross-validation
for K in neighbors:
    knn = KNeighborsClassifier(n_neighbors = K)
    scores = cross_val_score(knn,x_train,y_train,cv = 10,scoring =
    "accuracy")
    cv_scores.append(scores.mean())

# Changing to mis classification error
mse = [1-x for x in cv_scores]
# determing best k
optimal_k = neighbors[mse.index(min(mse))]
print("The optimal no. of neighbors is {}".format(optimal_k))

#plotting knn scores
def plot_accuracy(knn_list_scores):
    pd.DataFrame({"K":[i for i in range(1,50)], "Accuracy":knn_list_scores}).set_index("K").plot.bar(figsize= (9,6),ylim=(0.78,0.83),rot=0)
    plt.show()
plot_accuracy(cv_scores)

model = KNeighborsClassifier(n_neighbors=optimal_k, metric="minkowski")
model.fit(x_train, y_train)
y_knn_pred = model.predict(x_test)
knn_score = r2_score(y_test, y_knn_pred) #result = 62.5%


#####COMPARE THE ACCURACY
from tabulate import tabulate
titles = ['Linear Regression','Polynomial Regression','Logistic Regression','Support Vector Machine','K Nearest Neighbour']
data = [linear_score,poly_score,log_score,svm_score,knn_score]
print(tabulate([data],headers=titles))
#we will export the logistic regression model

#####EXPORTING AS PICKLE
from sklearn.pipeline import Pipeline
pipeline = Pipeline(steps=[('sc', sc),('classifier', log_reg)])

import pickle
pickle.dump(pipeline, open('ml.pkl','wb'))
