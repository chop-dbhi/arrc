from sklearn.externals import joblib

__author__ = 'Aaron J. Masino'

import time, os
from sklearn import linear_model, svm, tree
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
from numpy.random import RandomState
from learn import wrangle, printers
from nlp import util
import learn.sklearn_extensions as sklx
from nltk.corpus import stopwords
from learn.metrics import PerformanceMetrics
from os import listdir
from os.path import isfile, join

def load_report(path):
    f = open(path,'r')
    text = reduce(lambda x,y: x+y, f.readlines(), "")
    f.close()
    return text

def concatenate(d1,d2):
    d = d1.copy()
    d.update(d2)
    return d

#custom preprocessor to keep some stop words
english_stopwords = filter(lambda w: w not in ['no', 'not', 'under'], stopwords.words('english'))
def text_preprocessor(text):
    ct = util.replace_digits(text)
    ct = util.replace_numerals(ct)
    ct = util.replace_units(ct)
    _words = [word.lower() for word in util.words(ct)]
    _words = filter(lambda x: x not in english_stopwords and len(x)>=2, _words)
    _words = util.porter_stem(_words)
    return reduce(lambda x,y: '{0} {1}'.format(x,y), _words, "")

def analyze_classifiers(region_key, classifiers, x_train, y_train, x_test, y_test, out_file, preprocessor=text_preprocessor):
    printers.printsf('{0}Analysis for {1} ear region{0}'.format(40*'-', region_key), out_file)
    for key,value in classifiers.items():
        clf = value[0] #the classifier
        usa = value[1] #use spare array
        ubf = value[2] #use binary features (this is to support NB)
        parameters = value[3]
        vectorizer = CountVectorizer(input='content', decode_error='ignore', preprocessor=preprocessor, binary=ubf)
        pipeline = (Pipeline(steps=[('vect', vectorizer),('clf',clf)]) if usa
                    else Pipeline(steps=[('vect', vectorizer),('sa',sklx.SparseToArray()),('clf',clf)]))
        gs = sklx.grid_analysis(pipeline,parameters, x_train, y_train)
        printers.print_grid_search_results(gs,key,out_file,x_test,y_test)

if __name__ == '__main__':

    use_finding_impression_only = True
    analyze_baseline = False
    analyze_all_classifiers = True

    # static parameters
    kfolds = 5
    seed = 987654321
    # set the numpy random seed so results are reproducible
    rs = RandomState(987654321)

    # set common path variables
    label_file = './data/input/SDS_PV2_combined/SDS_PV2_class_labels.txt'
    report_path = './data/input/SDS_PV2_combined/{0}'
    file_suffix = ''
    if use_finding_impression_only:
        report_path = report_path.format('reports_single_find_impr')
        file_suffix = '_fi.txt'
    else:
        report_path = report_path.format('reports_single')
        file_suffix = '.txt'

    output_path = './data/output/{0}'


    standard_out_file = output_path.format('SDS_PV2_results_.txt')


    # read data
    n = 4
    if(use_finding_impression_only): n = 7
    pids_from_files = [f[0:-n] for f in listdir(report_path) if isfile(join(report_path,f)) and f.endswith('.txt')]
    all_label_data = pd.read_csv(label_file)
    label_data = all_label_data[all_label_data['pid'].isin(pids_from_files)]
    region_keys = label_data.columns[2:6]
    miss_labeled_file = output_path.format('SDS_PV2_missed_.txt')
    if not os.path.exists(os.path.dirname(standard_out_file)):
        os.makedirs(os.path.dirname(standard_out_file))
    now = time.localtime()
    (printers.
     printsf('{6}{0}-{1}-{2} {3}:{4}:{5}{6}'.
             format(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec, 40*'-'),
             standard_out_file, 'a',False))

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
    train_reports = [load_report('{0}/{1}{2}'.format(report_path, pid, file_suffix)) for pid in train_labels['pid']]
    test_reports = [load_report('{0}/{1}{2}'.format(report_path, pid, file_suffix)) for pid in test_labels['pid']]

    #------------------------------ BASELINE ANALYSIS -----------------------------------------------------------------

    if analyze_baseline:

        clf = linear_model.LogisticRegression(C=1000)
        clf = BernoulliNB(alpha=1.0, binarize=None, fit_prior=True, class_prior=None)

        usa = False  #use sparse array, should be false for NB classifier
        binary_features = True #should be true for NB classifier
        apply_text_preprocessing = False
        tpp = None
        if apply_text_preprocessing: tpp = text_preprocessor

        #start with no regulariztion, unigrams and no text preprocessing
        vectorizer = CountVectorizer(input='content', decode_error='ignore', analyzer='word',
                                     preprocessor=tpp, ngram_range=(1,1), stop_words=None, lowercase=False,
                                     binary=binary_features)

        pipeline = (Pipeline(steps=[('vect', vectorizer),('clf',clf)]) if usa
                    else Pipeline(steps=[('vect', vectorizer),('sa',sklx.SparseToArray()),('clf',clf)]))

        for key in region_keys:
            printers.printsf("{1} Performance for {0} region {1}".format(key,40*'-'), standard_out_file)
            y_train = train_labels[key]
            y_test = test_labels[key]
            pipeline.fit(train_reports, y_train)
            y_test_predicted = pipeline.predict(test_reports)
            pm = PerformanceMetrics(y_test, y_test_predicted)
            printers.printsfPerformanceMetrics(pm, standard_out_file)
            # print miss classified examples
            missed_labels = test_labels[y_test!=y_test_predicted]
            missed_pids = missed_labels['pid']
            missed_correct_class = missed_labels[key]
            missed = reduce(lambda x,y: x+'{0}\t{1}\n'.format(y[0],y[1]), zip(missed_pids, missed_correct_class),'')
            printers.printsf('{1}MISSED EXAMPLES REGION {0}{1}\n'.format(key, 40*'#'), miss_labeled_file)
            printers.printsf(missed, miss_labeled_file)
            #print confusion matrix
            cm = confusion_matrix(y_test,y_test_predicted)
            printers.printTwoClassConfusion(cm, standard_out_file)

    #------------------------------- CROSS VALIDATION ANALYSIS ALL CLASSIFIERS ----------------------------------------

    if analyze_all_classifiers:
        # classifiers and parameters to consider for each region
        feature_parameters  = {'vect__preprocessor':(None, text_preprocessor),
                        'vect__binary':(False, True),
                       'vect__ngram_range': ((1,1),(1,2),(1,3)),
                       'vect__analyzer' : ('word', 'char_wb')}
        nb_feature_parameters  = {'vect__ngram_range': ((1,1),(1,2),(1,3)),
                       'vect__analyzer' : ('word', 'char_wb')}
        use_spare_array = True
        use_binary_features = True
        classifiers = ({
            'logistic_regression':(linear_model.LogisticRegression(),
                                   use_spare_array,
                                   not use_binary_features,
                                   concatenate(feature_parameters, {'clf__C': [1/x for x in [0.1, 0.3, 1.0, 3.0, 10.0]]})),
            'svm_linear':(svm.LinearSVC(tol=1e-6),
                          use_spare_array,
                          not use_binary_features,
                          concatenate(feature_parameters, {'clf__C': [1/x for x in [0.1, 0.3, 1.0, 3.0, 10.0]]})),
            'svm_gaussian':(svm.SVC(tol=1e-6, kernel='rbf'),
                            use_spare_array,
                            not use_binary_features,
                            concatenate(feature_parameters, {'clf__gamma': [.01, .03, 0.1, 0.3, 1.0, 3.0],
                                                     'clf__C': [1/x for x in [0.1, 0.3, 1.0, 3.0, 10.0]]})),
            'decision_tree':(tree.DecisionTreeClassifier(criterion='entropy', random_state=RandomState(seed)),
                             not use_spare_array,
                             not use_binary_features,
                             concatenate(feature_parameters,{'clf__max_depth': [2, 3, 4, 5, 6, 7 , 8, 9, 10, 15, 20]})),
            'random_forest':(RandomForestClassifier(criterion='entropy', random_state=RandomState(seed)),
                             not use_spare_array,
                             not use_binary_features,
                             concatenate(feature_parameters,{'clf__max_depth': [2, 3, 4, 5],
                                                             'clf__n_estimators': [5, 25, 50, 100, 150, 200]})),
            'naive_bayes':(BernoulliNB(alpha=1.0, binarize=None, fit_prior=True, class_prior=None),
                           use_spare_array,
                           use_binary_features,
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

