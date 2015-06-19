__author__ = 'Aaron J. Masino'

import numpy as np
from metrics import PerformanceMetrics
import sklearn

def printsf(text, out_file=None, fmode = 'a', print_to_screen=True, carriage_returns=1):
    if print_to_screen: print text
    if out_file:
        with open(out_file, fmode) as f:
            f.write(text)
            for idx in range(carriage_returns): f.write('\n')

def printsfPerformanceMetrics(pm, out_file_path, print_to_screen=True, carriage_returns=1):
    '''precision = tp / (tp + fp) where tp is number of true positives in the predicted labels, and fp is the number of
    false positives in the predicted labels
    recall = tp / (tp + fn) where tp is same as precision, and fn is the number of false negatives is the predicted labels
    f1 = 2 * (precision * recall) / (precision + recall)'''
    printsf('Accuracy:\t\t\t{0:.4f}'.format(pm.accuracy),out_file_path,print_to_screen=print_to_screen)
    printsf('F1-Score:\t\t\t{0:.4f}'.format(pm.f1),out_file_path,print_to_screen=print_to_screen)
    printsf('PPV/Precision tp/pp:\t\t{0:.4f}'.format(pm.ppv),out_file_path,print_to_screen=print_to_screen)
    printsf('NPV tn/pn:\t\t\t{0:.4f}'.format(pm.npv),out_file_path,print_to_screen=print_to_screen)
    printsf('Sensitivity/Recall tp/[tp+fn]:\t{0:.4f}'.format(pm.recall),out_file_path,print_to_screen=print_to_screen)
    printsf('Specificity tn/[tn+fp]:\t\t{0:.4f}'.format(pm.specificity),out_file_path,print_to_screen=print_to_screen,carriage_returns=2)

def printTwoClassConfusion(cm, out_file_path, print_to_screen=True):
    printsf('\n\tConfusion Matrix', out_file_path,print_to_screen=print_to_screen)
    printsf('A  \t   Predicted', out_file_path,print_to_screen=print_to_screen)
    printsf('c  \t   Normal\tAbnormal', out_file_path,print_to_screen=print_to_screen)
    printsf('t  Normal      {0}\t{1}'.format(cm[0,0],cm[0,1]), out_file_path,print_to_screen=print_to_screen)
    printsf('u  Abnormal    {0}\t{1}'.format(cm[1,0],cm[1,1]), out_file_path,print_to_screen=print_to_screen)
    printsf('a  ', out_file_path,print_to_screen=print_to_screen)
    printsf('l  ', out_file_path,print_to_screen=print_to_screen)

def print_data_stats(train_labels, test_labels,heading, f):
    tc = len(train_labels)
    ac=np.sum(train_labels==1)
    nc=np.sum(train_labels==0)
    printsf(heading, f)
    printsf('Total Training Cases:\t\t{0}'.format(tc),f)
    printsf('Abnormal Training Cases:\t{0}, {1:.2f}%'.format(ac, ac/float(tc)*100),f)
    printsf('Normal Training Cases:\t\t{0}, {1:.2f}%'.format(nc, nc/float(tc)*100),f)

    tc = len(test_labels)
    ac=np.sum(test_labels==1)
    nc=np.sum(test_labels==0)
    printsf('\nTotal Test Cases:\t{0}'.format(tc),f)
    printsf('Abnormal Test Cases:\t{0}, {1:.2f}%'.format(ac, ac/float(tc)*100),f)
    printsf('Normal Test Cases:\t{0}, {1:.2f}%'.format(nc, nc/float(tc)*100),f)

def print_grid_search_results(grid_search, name, out_file, test_input, test_labels):
    #print results
    printsf('\n{0} {1} Grid Search Results {0}'.format(30*'-',name),out_file)
    printsf('Best score: %0.3f' % grid_search.best_score_, out_file)
    printsf('Best parameter set:',out_file)
    best_params = grid_search.best_estimator_.get_params()
    for param_name in sorted(best_params.keys()):
        printsf("\t%s: %r" % (param_name, best_params[param_name]), out_file)

    #evaluate on test data
    best_classifier = grid_search.best_estimator_
    y_test_predicted = best_classifier.predict(test_input)
    printsf('\nTest Data Performance with Best Estimator Parameters\n',out_file)
    pm = PerformanceMetrics(test_labels, y_test_predicted)
    printsfPerformanceMetrics(pm, out_file)

    #print confusion matrix
    cm = sklearn.metrics.confusion_matrix(test_labels,y_test_predicted)
    printTwoClassConfusion(cm, out_file)
    return pm