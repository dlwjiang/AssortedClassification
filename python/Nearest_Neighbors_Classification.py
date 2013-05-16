'''
Modern Analytics Homework 1
@author: David Jiang

Exercises: 1g, 1h, 1j
'''
import numpy as np
from Digit_Display import open_data
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix
from numpy import savetxt
import scipy as sp

"""
Classifies using KNN and creates a .csv submission file
takes a training file and a test file an input
"""
def Nearest_Neighbors(training_file,test_file):
    #setup data
    data = open_data(training_file)
    
    target = np.asarray( data[::,0].astype(np.float) )
    features = np.asarray( [row[1::].astype(np.float) for row in data])
    
    print target[0]
    print features[0]
    
    #setup and train KNN classifier
    neigh = neighbors.KNeighborsClassifier(n_neighbors=4)
    neigh.fit(features, target) 
    
    #open test data
    data_for_test = open_data(test_file)
    
    #get predictions
    predicted = neigh.predict(data_for_test)
    
    #save results to csv for submission
    savetxt('Data/submission_neighbors.csv', predicted, delimiter=',', fmt='%d')

"""
Creates confusion matrix
Calculates cross validation score
"""
def Nearest_Neighbors_Validation():
    #setup data
    filename = 'Data/train.csv'
    data = open_data(filename)
    
    target = data[::,0].astype(np.float) 
    features = [row[1::].astype(np.float) for row in data]
    
    for x in range(1,31,3):
        #setup and train KNN classifier
        neigh = neighbors.KNeighborsClassifier(n_neighbors=x)
        neigh.fit(features, target) 
        
        #cross validate for a score
        cv = cross_validation.cross_val_score(neigh, features, target, cv = 3, n_jobs = 2)
        print str(x) + " : " + str(cv)
        #results seem to indicate a k value of 3 or 4 to be the best
    
    #run KNN
    predicted = neigh.predict(features)
    #confusion matrix analysis on training data
    print confusion_matrix(target,predicted)
 
if __name__ == "__main__":
    
    #Nearest_Neighbors_Validation()
    Nearest_Neighbors('Data/train_small.csv','Data/test.csv')
    