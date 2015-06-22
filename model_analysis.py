from sklearn.externals import joblib

__author__ = 'Aaron J. Masino'

import time, os
from sklearn import linear_model, svm, tree
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
import pandas as pd
import numpy as np
from numpy.random import RandomState
from learn import wrangle, printers, nlp
import learn.sklearn_extensions as sklx

def load_report(path):
    f = open(path,'r')
    text = reduce(lambda x,y: x+y, f.readlines(), "")
    f.close()
    return text

def concatenate(d1,d2):
    d = d1.copy()
    d.update(d2)
    return d

def analyze_classifiers(region_key, classifiers, x_train, y_train, x_test, y_test, out_file):
    printers.printsf('{0}Analysis for {1} ear region{0}'.format(40*'-', region_key), out_file)
    for key,value in classifiers.items():
        clf = value[0] #the classifier
        usa = value[1] #use spare array
        parameters = value[2]
        vectorizer = CountVectorizer(input='content', decode_error='ignore', preprocessor=nlp.text_preprocessor)
        pipeline = (Pipeline(steps=[('vect', vectorizer),('clf',clf)]) if usa
                    else Pipeline(steps=[('vect', vectorizer),('sa',sklx.SparseToArray()),('clf',clf)]))
        gs = sklx.grid_analysis(pipeline,parameters, x_train, y_train)
        printers.print_grid_search_results(gs,key,out_file,x_test,y_test)



if __name__ == '__main__':
    # set common path variables
    label_file = './data/input/SDS_PV2_combined/SDS_PV2_class_labels.txt'
    report_path = './data/input/SDS_PV2_combined/reports'
    output_path = './data/output/{0}'
    label_data = pd.read_csv(label_file)
    region_keys = label_data.columns[2:6]
    standard_out_file = output_path.format('SDS_PV2_results_v1.txt')
    if not os.path.exists(os.path.dirname(standard_out_file)):
        os.makedirs(os.path.dirname(standard_out_file))
    now = time.localtime()
    (printers.
     printsf('{6}{0}-{1}-{2} {3}:{4}:{5}{6}'.
             format(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec, 40*'-'),
             standard_out_file, 'a',False))

    # static parameters
    kfolds = 5
    seed = 987654321
    # set the numpy random seed so results are reproducible
    rs = RandomState(987654321)

    # partition the data
    pos_cases, neg_cases = wrangle.partion(label_data['doc_norm']==1, label_data, ratios=[0.8,0.2])
    train_mask = np.concatenate((pos_cases[0], neg_cases[0]))
    test_mask = np.concatenate((pos_cases[1], neg_cases[1]))
    rs.shuffle(train_mask)
    rs.shuffle(test_mask)
    train_labels = label_data.iloc[train_mask]
    test_labels = label_data.iloc[test_mask]

    # print partition stats
    printers.printsf('{0}Data Partition Stats{0}'.format(40*'-'), standard_out_file)
    for key in region_keys:
        printers.print_data_stats(train_labels[key], test_labels[key],
                                  '{0}{1}{0}'.format(40*'-', key),standard_out_file)

    # read in the text reports
    train_reports = [load_report('{0}/{1}.txt'.format(report_path,pid)) for pid in train_labels['pid']]
    test_reports = [load_report('{0}/{1}.txt'.format(report_path,pid)) for pid in test_labels['pid']]

    # classifiers and parameters to consider for each region
    feature_parameters  = {'vect__binary':(False, True),
                   'vect__ngram_range': ((1,1),(1,2),(1,3)),
                   'vect__analyzer' : ('word', 'char_wb')}
    use_spare_array = True
    classifiers = ({
        'logistic_regression':(linear_model.LogisticRegression(),
                               use_spare_array,
                               concatenate(feature_parameters, {'clf__C': [1/x for x in [0.1, 0.3, 1.0, 3.0, 10.0]]})),
        'svm_linear':(svm.LinearSVC(tol=1e-6),
                      use_spare_array,
                      concatenate(feature_parameters, {'clf__C': [1/x for x in [0.1, 0.3, 1.0, 3.0, 10.0]]})),
        'svm_gaussian':(svm.SVC(tol=1e-6, kernel='rbf'),
                        use_spare_array,
                        concatenate(feature_parameters, {'clf__gamma': [.01, .03, 0.1, 0.3, 1.0, 3.0],
                                                 'clf__C': [1/x for x in [0.1, 0.3, 1.0, 3.0, 10.0]]})),
        'decision_tree':(tree.DecisionTreeClassifier(criterion='entropy', random_state=RandomState(seed)),
                         not use_spare_array,
                         concatenate(feature_parameters,{'clf__max_depth': [2, 3, 4, 5, 6, 7 , 8, 9, 10, 15, 20]})),
        'random_forest':(RandomForestClassifier(criterion='entropy', random_state=RandomState(seed)),
                         not use_spare_array,
                         concatenate(feature_parameters,{'clf__max_depth': [2, 3, 4, 5],
                                                         'clf__n_estimators': [5, 25, 50, 100, 150, 200]})),
        'naive_bayes':(BernoulliNB(alpha=1.0, binarize=None, fit_prior=True, class_prior=None),
                       use_spare_array,
                       {'vect__ngram_range':((1,1),(1,2),(1,3)),
                        'vect__analyzer':('word', 'char_wb')})
        })


    #analyze model performance for classifiers X regions
    #WARNING: This may run for hours (or even days) depending on the number of classifiers
    #and parameters considered
    for key in region_keys:
        y_train = train_labels[key]
        y_test = test_labels[key]
        analyze_classifiers(key, classifiers, train_reports, y_train, test_reports, y_test, standard_out_file)

