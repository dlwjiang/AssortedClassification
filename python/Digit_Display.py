"""
Modern Analytics Homework 1
@author: David Jiang

Module includes digit display function & histogram function.
Exercises: 1a,1b,1c
"""

import csv as csv 
import numpy as np
import matplotlib.pyplot as plt
import Image


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
    
def count_prior_probability(data): #counts the number of each digit 0-9 for prior probability

    target = data[::,0] #first column, list of the numbers

    count = {'0':0 ,'1':0,'2':0,'3':0,'4':0,'5':0,'6':0,'7':0,'8':0,'9':0} #to store results
    
    for x in target:           
        count[str(x[0])] += 1            
    
    length = float(len(target))
    
    print count
    print ("0: %0.3f \n1: %0.3f \n2: %0.3f \n3: %0.3f \n4: %0.3f \n5: %0.3f \n6: %0.3f \n7: %0.3f \n8: %0.3f \n9: %0.3f" 
        % ((count['0']/length), (count['1']/length), (count['2']/length), (count['3']/length), (count['4']/length), (count['5']/length), (count['6']/length), (count['7']/length), (count['8']/length), (count['9']/length)))
        
def show_histogram(data):
    
    target = data[::,0].astype(np.float)
    
    plt.hist(target, align='right', normed=True)
    plt.title("Ugly Histogram")
    plt.xlabel("Number")
    plt.ylabel("Percentage")
    plt.show()
    
def print_row(features,num): #prints the number as defined in row[num] of the csv file
    
    img = Image.new( 'L', (28,28)) # create an image
    pixels = img.load() # create the pixel map
    
    row = features[num] 
    count = 0
    for i in range(28):
        for j in range(28):
            pixels[j,i] = 255 - int(row[count])
            count += 1    
    
    img.show()
    return img
    
if __name__ == "__main__":
    
    filename = 'Data/train.csv'
    data = open_data(filename)
    
    features = np.asarray( [row[1::] for row in data] )
    
    count_prior_probability(data)
    show_histogram(data)
    
    numbers = [1,0,16,7,3,8,21,6,10,11]
    
    #print an example of 0-9
    #ask about subplots   
    for x in numbers:
        print_row(features,x)
    
