# importing necessary libraries 
from sklearn import datasets 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
import numpy as np


data = np.loadtxt('ddd.csv', delimiter=',')

# loading the iris dataset 

m = data.shape[1]-1
# use slicing to extract subarrays
X = data[:,:m]
y = data[:,m]

# dividing X, y into train and test data 
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0) 

# training a linear SVM classifier 
from sklearn.svm import SVC 
svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train, y_train) 
svm_predictions = svm_model_linear.predict(X_test) 

# model accuracy for X_test 
accuracy = svm_model_linear.score(X_test, y_test) 
print(accuracy)

# creating a confusion matrix 
cm = confusion_matrix(y_test, svm_predictions) 
print(cm)