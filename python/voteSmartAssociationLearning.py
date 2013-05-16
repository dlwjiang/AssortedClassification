# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import pickle
import numpy as np

# <codecell>

f1 = open('/Users/david/Desktop/votesmartData/bills20DataSet.txt', 'rb')
f2 = open('/Users/david/Desktop/votesmartData/bills20ItemMeaning.txt', 'rb')

dataVoteSmart = np.array(pickle.load(f1))
dataMeaning   = np.array(pickle.load(f2))

# <codecell>

'''
Visualizes rules
'''
def printRules(array):
    for result in array:
        print result,
        print dataMeaning[result]

printRules(dataVoteSmart[0])

# <codecell>

print len(dataVoteSmart)
print len(dataMeaning  )

# <codecell>

print np.array(dataVoteSmart)

# <codecell>

print np.array(dataMeaning)

# <codecell>

"""
==
======
==========
===============
Apriori Algorithm Example from Harrington Chpt 11.
Listing 11.1 & ll.2
==================================================
https://github.com/pbharrin/machinelearninginaction/blob/master/Ch11/apriori.py
"""
def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]

def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
                
    C1.sort()
    return map(frozenset, C1)#use frozen set so we
                            #can use it as a key in a dict    

def scanD(D, Ck, minSupport):
    ssCnt = {}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                if not ssCnt.has_key(can): ssCnt[can]=1
                else: ssCnt[can] += 1
    numItems = float(len(D))
    retList = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key]/numItems
        if support >= minSupport:
            retList.insert(0,key)
        supportData[key] = support
    return retList, supportData

def aprioriGen(Lk, k): #creates Ck
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1, lenLk): 
            L1 = list(Lk[i])[:k-2]; L2 = list(Lk[j])[:k-2]
            L1.sort(); L2.sort()
            if L1==L2: #if first k-2 elements are equal
                retList.append(Lk[i] | Lk[j]) #set union
    return retList

def apriori(dataSet, minSupport = 0.5):
    C1 = createC1(dataSet)
    D = map(set, dataSet)
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1]
    k = 2
    while (len(L[k-2]) > 0):
        Ck = aprioriGen(L[k-2], k)
        Lk, supK = scanD(D, Ck, minSupport)#scan DB to get Lk
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData

def generateRules(L, supportData, minConf=0.7):  #supportData is a dict coming from scanD
    bigRuleList = []
    for i in range(1, len(L)):#only get the sets with two or more items
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            if (i > 1):
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList         

def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    prunedH = [] #create new list to return
    for conseq in H:
        conf = supportData[freqSet]/supportData[freqSet-conseq] #calc confidence
        if conf >= minConf: 
            print freqSet-conseq,'-->',conseq,'conf:',conf
            brl.append((freqSet-conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH

def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    m = len(H[0])
    if (len(freqSet) > (m + 1)): #try further merging
        Hmp1 = aprioriGen(H, m+1)#create Hm+1 new candidates
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
        if (len(Hmp1) > 1):    #need at least two sets to merge
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)
            
def pntRules(ruleList, itemMeaning):
    for ruleTup in ruleList:
        for item in ruleTup[0]:
            print itemMeaning[item]
        print "           -------->"
        for item in ruleTup[1]:
            print itemMeaning[item]
        print "confidence: %f" % ruleTup[2]
        print       #print a blank line
        
   




# <codecell>

'''
Testing the (half) Apriori Algorithm
'''
dataset = loadDataSet()
D = map (set, dataset)
C1 = createC1(D)
retList, supportData = scanD(D,C1,0.60)

print dataset
print 'candidates:'
print C1
print len(C1)
print 'frequent:'
print retList



# <codecell>

'''
Testing the (full) Apriori Algorithm
'''
dataset = loadDataSet()
L, suppData = apriori(dataset, minSupport = 0.5)
print L

# <codecell>

'''
Testing rules algorithms
'''

rulesList = generateRules(L,suppData, minConf = 0.5)
print rulesList


# <codecell>

'''
testing generateRules on voteSmart data
'''
VS, VS_support = apriori(dataVoteSmart, minSupport = 0.3)

rulesList = generateRules(VS,VS_support, minConf = 0.99)

print(len(rulesList))


# <codecell>

'''
find the best minimum support:
'''
listLen = []

for i in range(2,10):   
    minSupp = float(i)/10
    a, a_support = apriori(dataVoteSmart, minSupport = minSupp)
    listLen.append(len(a))
    
print listLen

# <codecell>

#agree w/ book, 0.3 seems good
a, a_support = apriori(dataVoteSmart, minSupport = 0.3)

rulesListVS = generateRules(a, a_support, minConf = 1)

print rulesListVS[0]

'''
Perfects:
============================
frozenset([3])                --> frozenset([9]) conf: 1.0
frozenset([26, 3])            --> frozenset([0, 9]) conf: 1.0
frozenset([26, 3, 4])         --> frozenset([0, 9]) conf: 1.0
frozenset([26, 3, 23])        --> frozenset([0, 9]) conf: 1.0
frozenset([25, 26, 3])        --> frozenset([0, 9]) conf: 1.0
frozenset([26, 3, 7])         --> frozenset([0, 9]) conf: 1.0
frozenset([26, 3, 4, 7])      --> frozenset([0, 9]) conf: 1.0
frozenset([25, 26, 3, 7])     --> frozenset([0, 9]) conf: 1.0
frozenset([25, 26, 3, 4])     --> frozenset([0, 9]) conf: 1.0
frozenset([25, 26, 3, 23])    --> frozenset([0, 9]) conf: 1.0
frozenset([26, 3, 4, 23])     --> frozenset([0, 9]) conf: 1.0
frozenset([23, 26, 3, 7])     --> frozenset([0, 9]) conf: 1.0
frozenset([23, 25, 26, 3, 7]) --> frozenset([0, 9]) conf: 1.0
frozenset([25, 26, 3, 4, 23]) --> frozenset([0, 9]) conf: 1.0
frozenset([23, 26, 3, 4, 7])  --> frozenset([0, 9]) conf: 1.0
frozenset([25, 26, 3, 4, 7])  --> frozenset([0, 9]) conf: 1.0
'''

# <codecell>

printRules([0,1,3,4,7,9,23,25,26])

#some strong opinions about the health care bill...

# <codecell>

rulesListVS = generateRules(a, a_support, minConf = 0.95)

# <codecell>


