__author__ = 'Aaron J. Masino'

from sklearn import metrics
import numpy as np

def negpv(y_actual, y_predicted):
    '''calculates the negative predictive value as tn/(tn+fn), where tn = true predicted negatives
    fn = false predicted negatives
    assuems negative cases are indicated by 0 and positive cases by 1'''
    ya = np.array(y_actual)
    yp = np.array(y_predicted)
    ti = np.where(ya==0)  #find indices for actual negatives
    if len(ti[0])==0: return -1 #no actual negatives
    tn = 0
    neg_calls = 0
    for idx in range(len(ya)):
        if ya[idx]==0 and yp[idx]==0: tn +=1
        if yp[idx] == 0: neg_calls += 1
    #tn = len(ti[0]) - np.sum(yp[ti])
    if neg_calls>0: return tn/float(neg_calls)
    else: return np.nan
    #return tn/float(len(np.where(yp==0)[0]))

def specificity(y_actual, y_predicted):
    '''calculates specificity as #predicted true neg / #actual negatives
       assumed that negative cases are indicated as 0 and positive cases indicated by 1'''
    ya = np.array(y_actual)
    yp = np.array(y_predicted)
    ti = np.where(ya==0) #find indices for actual negatives
    if len(ti[0])==0: return -1 #no actual negatives
    #of true negatives is total negatives (ti), minus false positives (sum(yp[ti], i.e. count of non-zero values
    #in yp at the indices where ya is negative
    #true neg = len(ti) - sum(yp[ti])
    #spec = true_neg/len(ti)
    return 1 - np.sum(yp[ti])/float(len(ti[0]))

class PerformanceMetrics:
    def __init__(self, y_actual, y_predicted):
        #ensure we're using ndarray
        ya = np.array(y_actual)
        yp = np.array(y_predicted)
        self.f1 = metrics.f1_score(ya, yp)
        self.precision = metrics.precision_score(ya, yp)
        self.recall = metrics.recall_score(ya, yp)
        self.ppv = self.precision
        self.npv = negpv(ya,yp)
        self.sensitivity = self.recall
        self.specificity = specificity(ya, yp)
        self.pred_pos = np.sum(yp)
        self.act_pos = np.sum(ya)
        self.pred_neg = len(yp) - self.pred_pos
        self.act_neg = len(ya) - self.act_pos
        self.accuracy = len(np.where(ya==yp)[0])/float(len(ya))

class KFoldPerformanceMetrics:
    def __init__(self, pms):
        '''given a collection of PerformanceMetric objects, this object
        creates means and standard deviations for each metric'''
        self.pms = pms
        self.f1_mean = np.mean([pm.f1 for pm in pms])
        self.f1_std = np.std([pm.f1 for pm in pms])
        self.precision_mean = np.mean([pm.precision for pm in pms])
        self.precision_std = np.std([pm.precision for pm in pms])
        self.recall_mean = np.mean([pm.recall for pm in pms])
        self.recall_std = np.std([pm.recall for pm in pms])
        self.ppv_mean = np.mean([pm.ppv for pm in pms])
        self.ppv_std = np.std([pm.ppv for pm in pms])
        self.npv_mean = np.mean([pm.npv for pm in pms])
        self.npv_std = np.std([pm.npv for pm in pms])
        self.sensitivity_mean = np.mean([pm.sensitivity for pm in pms])
        self.sensitivity_std = np.std([pm.sensitivity for pm in pms])
        self.specificity_mean = np.mean([pm.specificity for pm in pms])
        self.specificity_std = np.std([pm.specificity for pm in pms])
        self.accuracy_mean = np.mean([pm.accuracy for pm in pms])
        self.accuracy_std = np.std([pm.accuracy for pm in pms])


def printPerformanceMetrics(y_actual, y_predicted):
    '''precision = tp / (tp + fp) where tp is number of true positives in the predicted labels, and fp is the number of
    false positives in the predicted labels
    recall = tp / (tp + fn) where tp is same as precision, and fn is the number of false negatives is the predicted labels
    f1 = 2 * (precision * recall) / (precision + recall)'''
    pm = PerformanceMetrics(y_actual, y_predicted)
    print 'Accuracy:\t\t\t{0}'.format(pm.accuracy)
    print 'F1-Score:\t\t\t{0}'.format(pm.f1)
    print 'PPV/Precision tp/pp:\t\t{0}'.format(pm.ppv)
    print 'NPV tn/pn:\t\t\t{0}'.format(pm.npv)
    print 'Sensitivity/Recall tp/[tp+fn]:\t{0}'.format(pm.recall)
    print 'Specificity tn/[tn+fp]:\t\t{0}'.format(pm.specificity)
    return pm

def printKFoldPerformanceMetrics(performanceMetrics):
    pm = KFoldPerformanceMetrics(performanceMetrics)
    print 'Accuracy:\t\t\t{0}'.format(pm.accuracy_mean)
    print 'F1-Score:\t\t\t{0}'.format(pm.f1_mean)
    print 'PPV/Precision tp/pp:\t\t{0}'.format(pm.ppv_mean)
    print 'NPV tn/pn:\t\t\t{0}'.format(pm.npv_mean)
    print 'Sensitivity/Recall tp/[tp+fn]:\t{0}'.format(pm.recall_mean)
    print 'Specificity tn/[tn+fp]:\t\t{0}'.format(pm.specificity_mean)
    return pm

def learningCurves(predictor, X_train, y_train, X_other, y_other, minI=0):
    trainErr = []
    cvErr = []
    for i in range(minI,len(y_train)):
        predictor.fit(X_train[0:i], y_train[0:i])
        trainErr.append(1-predictor.score(X_train[0:i], y_train[0:i]))
        cvErr.append(1-predictor.score(X_other, y_other))
    return(trainErr, cvErr)

def findKFoldMax(pmDict, metric_id='f1', rule_out_uniform=True):
    '''finds the best result in pmDict based on the metric_id.
       If rule_out_uniform is True, the selection rules out any choice
       that predicts all positive or all negative'''
    max_key = 0
    max_metric = 0
    means = {}
    for key in pmDict.keys():
        #pms = pmDict[key]
        #is_uniform = rule_out_uniform
        #if is_uniform:
        #    for pm in pms:
        #        total_cases = pm.act_pos + pm.act_neg
        #        is_uniform = is_uniform and (pm.pred_pos == total_cases or pm.pred_neg == total_cases)
        #if is_uniform: print 'WARNING: key={0} ruled out due to uniformity'.format(key)
        temp = None
        if metric_id == 'f1':
            temp = [pm.f1 for pm in pmDict[key]]
        temp = np.mean(temp)
        if temp>max_metric:
            max_metric = temp
            max_key = key
        means[key]=temp
    return (max_metric, max_key, means)