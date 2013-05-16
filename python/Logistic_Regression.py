'''
Modern Analytics Homework 1
@author: David Jiang

Exercises: 2 all
'''
from sklearn.linear_model import LogisticRegression
from Digit_Display import open_data
import numpy as np
from sklearn import cross_validation

def Predict_Survivors():
    
    data = open_data('Data/Titanic/train.csv')
    
    target = data[::,0].astype(np.float)
    features = [row[1::] for row in data] 
    
    #filter out class and sex and age as main features
    # [1,2,3 are first/second/third class]  [0 is male, 1 is female] [0 is child, 1 is adult]
    filtered_features = strip_array(features)
     
    log_reg = LogisticRegression().fit(filtered_features,target)
      
    print log_reg.predict_proba([3,0]) #probability of third class male adult?
    print log_reg.predict_proba([1,1]) #probability of first class female adult?
    
    data_test     = open_data('Data/Titanic/test.csv')
    data_test_array = [row[0::] for row in data_test]
    
    data_test_array_stripped = strip_array(data_test_array)
    predicted = log_reg.predict(data_test_array_stripped)
    
    cv = cross_validation.cross_val_score(log_reg, filtered_features, target, cv = 3, n_jobs = 2)
    print cv
    
    np.savetxt('Data/Titanic/submission_titanic_class_gender_age.csv', predicted, delimiter=',', fmt='%d')
    
def strip_array(array):
    """
    -Takes the input feature array and strips it leaving only the relevant features.
    -Converts information into integers setting up dummy variables when relevant
    -Format will be [ Survived? , Passenger class , Sex , Age ]
    ---Returns the data as an array
    """
    
    #setup return value
    filtered_features =   [[float(x[0]),x[2]]  for x in array]
    
    #0 for male, 1 for female    
    for x in filtered_features:
        if (x[1] == 'male') :
            x[1] = 0
        else:
            x[1] = 1
    '''
    #fill in empty ages as 0
    #assuming newborns don't have proper records, could also assign age 1000 or randomly allocate or remove from dataset
    for x in filtered_features:
        if (x[2] == ''):                                  
            x[2] = 0
    
    #convert everything to a float
    for x in filtered_features:
        x[2] = float(x[2])
    
    #classify as 1 for adult(>=18), else 0
    for x in filtered_features:
        if (x[2] >= 18) :
            x[2] = 1
        else:
            x[2] = 0
    '''
    
    filtered_features = np.array(filtered_features)
    return filtered_features
    
if __name__ == "__main__":
    Predict_Survivors()
    
    
    
    
    
    
    
    
    