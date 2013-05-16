# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

'''
random forest classification of handwritten digits.
will use a small training set to classify a large one.
data is from kaggle's digita classification tutorial.
'''

import csv as csv 
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from numpy import savetxt

# <codecell>

def open_data(filename):
    #Open up the csv file 
    csv_file_object = csv.reader(open(filename, 'rb')) 
    header = csv_file_object.next() #skip header
                                    
    data=[]                         
    for row in csv_file_object:      
        data.append(row)   
                  
    data = np.array(data) 
        
    """
    target = data[::,0].astype(np.float) #first column, list of the numbers
    features = [row[1::].astype(np.float) for row in data] #list of the pixel values corresponding to numbers in 'target'
    """    
    
    return data

# <codecell>

data = open_data("train_small.csv")

# <codecell>

print data[:,0] #the first column, aka target
target = data[:,0]

# <codecell>

training = []
for each in data:
    training.append(each[1:])

# <codecell>

rf = RandomForestClassifier()

# <codecell>

rf.fit(training,target)

# <codecell>

'''
create training set
'''
test = open_data("train.csv")

# <codecell>

testSet = []
for each in test:
    testSet.append(each[1:])

# <codecell>

testSolutions = test[:,0]
print 'hi'
print len(testSolutions)

# <codecell>

predictions = rf.predict(testSet)

# <codecell>

print testSolutions

# <codecell>

print predictions

# <codecell>

"""
check solutions & accuracy
"""
numRight = 0.0
numTotal = 0.0
for i in range(len(predictions)):
    if (predictions[i] == testSolutions[i]) :
        numRight += 1
    numTotal += 1

print ("%d/%d  ---  %f3" ) % (numRight, numTotal, numRight/numTotal)

#35% which is good compared to random chance of 10%
#training set was too small at 33

# <codecell>


# <codecell>


# <codecell>

dataLarge = open_data("train.csv")
trainingLarge = []
targetLarge = dataLarge[:,0]

for each in dataLarge:
    trainingLarge.append(each[1:])

# <codecell>

"""
training on large data set
"""
rf.fit(trainingLarge,targetLarge)

# <codecell>

testSet2 = []
thisData =  open_data("test.csv")
for eachRow in thisData:
    testSet2.append(eachRow)

# <codecell>

predictionsLarge = rf.predict(testSet2)

# <codecell>

print len(predictionsLarge)
print predictionsLarge[0]

predictionsLarge = np.array(predictionsLarge)
saveThis = []
for eachThing in predictionsLarge:
    saveThis.append(int(eachThing))

# <codecell>

np.savetxt("newSubmission.csv", saveThis, delimiter='\n', fmt = "%d")

# <codecell>

# Kaggle name: LiJiang place 922, 94% accuracy 

# <codecell>


