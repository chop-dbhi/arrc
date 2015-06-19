__author__ = 'Aaron J. Masino'

import numpy as np

def extractBy(condition, data, tol = 1e-6):
    not_condition = condition[:]==False
    return (data[condition], data[not_condition])

def partion(condition, data, ratios=[.6,.2,.2]):
    ''' returns two lists (l1,l2). l1 is a list of numpy arrays where each array contains indices
    into the data where the condition is True and l2 is a list of numpy arrays where each array contains
    indicies into the data where the condition is False. The len(l1)=len(l2)=len(ratios) and
    the lists in l1 and l2 have lengths determined by the ratio values.'''
    pos = np.where(condition)[0]
    neg = np.where(condition[:]==False)[0]

    #SHOULD ALSO USE np.where(condition) to split data
    #NEED TO MODIFY TO RETURN MASKS ONLY
    #MASK SHOULD BE AN 1D NUMPY ARRAY
    #if not (np.sum(ratios) == 1 or np.sum(ratios) == 1.0): raise Exception('Ratios must sum to 1, got {0}'.format(np.sum(ratios)))
    #(pos, neg) = extractBy(condition, data)
    pos_row_count = pos.shape[0]
    neg_row_count = neg.shape[0]
    s1 = 0
    s2 = 0
    s3 = 0
    s4 = 0
    pdata = []
    ndata = []

    for i in range(len(ratios)):
        r = ratios[i]
        if i==len(ratios)-1:
            s2 = pos_row_count
            s4 = neg_row_count
        else:
            s2 = min(s1 + int(round(r*pos_row_count)), pos_row_count)
            s4 = min(s3 + int(round(r*neg_row_count)), neg_row_count)
        if s2<=s1: raise Exception('Insufficient positive data for partition, s1={0}, s2={1}'.format(s1,s2))
        if s4<=s3: raise Exception('Insufficient negative data for partition, s3={0}, s4={1}'.format(s3,s4))
        pdata.append(pos[s1:s2])
        ndata.append(neg[s3:s4])
        s1 = s2
        s3 = s4
    return(pdata,ndata)