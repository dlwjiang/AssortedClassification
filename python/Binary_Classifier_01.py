'''
Modern Analytics Homework 1
@author: David Jiang

Exercises: 1e,1f
'''

import numpy as np
from Digit_Display import open_data
import pylab as pl
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

"""
takes a .csv file name and does 1e and 1f per HW1
"""
def binary_classification(filename):
    
    data = open_data(filename)
    
    target = np.asarray( data[::,0].astype(np.float) )
    features = np.asarray( [row[1::].astype(np.float) for row in data] )
    
    """
    scan through target, and features, parsing out information for 0, and 1
    and append them to new arrays
    """   
    target_01 = []
    features_01 = []
    for x in range(len(target)):
        if target[x] == 0 or target[x] == 1:
            target_01.append(target[x])
            features_01.append(features[x])
    
    """
    examples with which to compare L2 distances
    """
    zero_sample = features_01[1]
    one_sample = features_01[0]
    
    """
    is a distance is a miss, place in hist_impostors.
    else place in hist_actual   
    """
    hist_actual = []
    hist_impostors = []
    
    ##following 2 variables used to calculate ROC curve
    #stores distances in order
    corresponding_distances = []
    #stores guesses, 1 if guess was correct, 0 if wrong
    guess_vector = []
    
    
    """
    Sort
    """
    for x in range(len(target_01)):
        #guess = what bin_decision believes the correct classification is
        guess = bin_decision(features_01[x],zero_sample,one_sample)[0]
        #corresponding L2 distance of the guess
        distance = bin_decision(features_01[x],zero_sample,one_sample)[1]
        
        corresponding_distances.append(distance)
        #check your guess, append information to histograms
        if (target_01[x] == guess) : #if guess was correct
            hist_actual.append(distance)
            guess_vector.append(0)
        elif (target_01[x] != guess) : #if guess was incorrect
            hist_impostors.append(distance)
            guess_vector.append(1)
        else:
            print 'error'
            
    #convert to np array
    actuals = np.array(hist_actual)
    impostors = np.array(hist_impostors)
    
    #show histograms and ROC curve
    show_2_histograms(actuals,impostors)
    show_roc_curve(guess_vector,corresponding_distances)
        
"""
compares features of input to features of 'one' and 'zero' using L2 distance
returns 1 if closer to 1, else returns zero
"""
def bin_decision(input_features,zero,one):   
    
    zero = np.linalg.norm(input_features - zero) #distance to zero
    one  = np.linalg.norm(input_features - one)  #distance to one
    
    if (one <= zero) :
        return [1,one]
    else:
        return [0,zero]
    
"""
Show 2 histograms together
"""
def show_2_histograms(info1,info2):
    plt.hist(info1, color = 'red',align='mid',stacked=True, normed=True, histtype = 'step')
    plt.hist(info2, color = 'blue',align='mid',stacked=True, normed=True,  histtype = 'step')
    plt.title("Dual Histogram")
    plt.xlabel("Distance")
    plt.ylabel("Percentage")
    plt.show()    

"""
Show a ROC curve
"""
def show_roc_curve(trueFalse,scores):
    fpr, tpr, thresholds = roc_curve(trueFalse,scores,pos_label=None)
    roc_auc = auc(fpr, tpr)
    print "Area under the ROC curve : %f" % roc_auc
    
    # Plot ROC curve
    #pl.clf()
    pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlim([0.0, 1.0])
    pl.ylim([0.0, 1.0])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title('Receiver operating characteristic example')
    pl.legend(loc="lower right")
    pl.show()
               
if __name__ == "__main__":
    filename = 'Data/train.csv'
    binary_classification(filename)
