'''
Modern Analytics Homework 1
@author: David Jiang

Exercises: 1d
'''
import numpy as np
from Digit_Display import open_data

def find_best_match_w_L2(filename):
    
    data = open_data(filename)
    
    target = data[::,0].astype(np.float)
    features = [row[1::].astype(np.float) for row in data]   
       
    zero_nine = [1,0,16,7,3,8,21,6,10,11] #corresponds to position of example of 0 -> 9
    
    for number in zero_nine: # for each number in zero_nine
                
        mininum = float("inf") 
        for i in range(len(target)):#loop through each row in target, find the L2 minimum
            if i != number: #exclude self, which would have an L2 of 0
                calculated_value = np.linalg.norm(features[number]-features[i]) #np.linalg.norm calculates L2 distance for a matrix
                if calculated_value < mininum:
                    mininum = calculated_value
                    closest_match_pos = i
                    
        #print out results        
        string = str(target[number]) + " (row[" + str(number) + "]) has the closest match with " + str(target[closest_match_pos]) + "(row[" + str(closest_match_pos) + "]) : " + str(mininum)    
        if target[number] != target[closest_match_pos]: # if the match is not correct, append an * as per instructions on homework   
            string += ' *'            
            print string
        else: #if correct, print normally
            print string
    

if __name__ == "__main__":
    
    filename = 'Data/train.csv'
    find_best_match_w_L2(filename)
    
    
    
    
    
    