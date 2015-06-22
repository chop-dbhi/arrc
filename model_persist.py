import ConfigParser, os, time
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
from sklearn import linear_model, svm, tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from learn.metrics import PerformanceMetrics
import learn.sklearn_extensions as sklx
import pandas as pd
import numpy as np
from numpy.random import RandomState
from learn import wrangle, printers, nlp

__author__ = 'Aaron J. Masino'

#SET A RANDOM SEED FOR REPEATABLE RESULTS
seed = 987654321

def load_linearSVC(config, section):
    c = 1.0
    if config.has_option(section,'C'):
        c = float(config.get(section,'C'))
    return svm.LinearSVC(tol=1e-6, C=c)

def load_gaussianSVC(config, section):
    c = 1.0
    gamma = 0.0
    if config.has_option(section,'C'):
        c = float(config.get(section,'C'))
    if config.has_option(section, 'gamma'):
        gamma = float(config.get(section,'gamma'))
    return svm.SVC(tol=1e-6, C=c, kernel='rbf', gamma=gamma)

def load_decisionTree(config, section):
    max_depth = None
    if config.has_option(section, 'max_depth'):
        max_depth = config.getint(section, 'max_depth')
    return tree.DecisionTreeClassifier(criterion='entropy', random_state=RandomState(seed), max_depth=max_depth)

def load_randomForestClassifier(config, section):
    max_depth = None
    n_estimators = 10
    if config.has_option(section, 'max_depth'):
        max_depth = config.getint(section, 'max_depth')
    if config.has_option(section, 'n_estimators'):
        n_estimators = config.getint(section, 'n_estimators')
    return RandomForestClassifier(criterion='entropy', random_state=RandomState(seed), max_depth=max_depth,
                                  n_estimators=n_estimators)

def load_logisticRegression(config, section):
    c = 1.0
    if config.has_option(section,'C'):
        c = float(config.get(section,'C'))
    return linear_model.LogisticRegression(C=c)

def load_classifier(config, section):
    clf = None
    classifier_type = config.get(section,'type')
    if classifier_type == 'LinearSVC':
        clf = load_linearSVC(config,section)
    elif classifier_type == 'SVC':
        clf = load_gaussianSVC(config, section)
    elif classifier_type == 'DecisionTree':
        clf = load_decisionTree(config, section)
    elif classifier_type == 'RandomForestClassifier':
        clf = load_randomForestClassifier(config, section)
    elif classifier_type == 'LogisticRegression':
        clf = load_logisticRegression(config,section)
    return clf

def build_pipeline(vect, clf, config, section):
    classifier_type = config.get(section,'type')
    if classifier_type == 'DecisionTreeClassifier' or classifier_type=='RandomForestClassifier':
        return Pipeline(steps=[('vect', vect),('sa',sklx.SparseToArray()),('clf',clf)])
    else:
        return Pipeline(steps=[('vect', vect),('clf',clf)])

def load_vectorizer(config, section):
    analyzer = 'word'
    if config.has_option(section, 'analyzer'):
        analyzer = config.get(section,'analyzer')

    binary = False
    if config.has_option(section,'binary'):
        binary = config.getboolean(section, 'binary')

    ngram_start = 1
    ngram_stop = 1
    if config.has_option(section, 'ngram_range_start'):
        ngram_start = config.getint(section, 'ngram_range_start')
    if config.has_option(section, 'ngram_range_stop'):
        ngram_stop = config.getint(section, 'ngram_range_stop')
    ngram_range = (ngram_start,ngram_stop)

    return CountVectorizer(input='content', decode_error='ignore', preprocessor=nlp.text_preprocessor,
                                 analyzer=analyzer, binary=binary, ngram_range=ngram_range)

def load_report(path):
    f = open(path,'r')
    text = reduce(lambda x,y: x+y, f.readlines(), "")
    f.close()
    return text

if __name__ == '__main__':
    # set common path variables
    region_keys = ['inner']

    # training/test data and output files
    label_file = 'data/input/SDS_PV2_combined/SDS_PV2_class_labels.txt'
    report_path = 'data/input/SDS_PV2_combined/reports'
    output_path = 'data/output/{0}'
    label_data = pd.read_csv(label_file)
    region_keys = label_data.columns[2:6]
    standard_out_file = output_path.format('Model_Persistence_Report.txt')
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
    print 'loading reports ..............'
    train_reports = [load_report('{0}/{1}.txt'.format(report_path,pid)) for pid in train_labels['pid']]
    test_reports = [load_report('{0}/{1}.txt'.format(report_path,pid)) for pid in test_labels['pid']]

    print 'loading configuration file .........'
    # load the configuration
    config_path = 'resources/config/models.ini'
    configuration = ConfigParser.ConfigParser()
    configuration.read(config_path)

    # instantiate, train, evaluate, & persist region models
    for key in region_keys:
        #load the vectorizer
        print 'loading vectorizer for {0} region .............'.format(key)
        vectorizer = load_vectorizer(configuration, '{0}_vectorizer'.format(key))

        print 'loading classifier for {0} region .............'.format(key)
        #load the classifier
        classifier = load_classifier(configuration, '{0}_classifier'.format(key))

        #create pipeline
        print 'fitting pipeline for {0} region ........'.format(key)
        pipeline = build_pipeline(vectorizer, classifier,configuration,'{0}_classifier'.format(key))
        pipeline.fit_transform(train_reports, train_labels[key])

        print 'Training and evaluating {0} region pipeline ...........'.format(key)
        #evaluate pipeline
        y_test_predicted = pipeline.predict(test_reports)
        pm = PerformanceMetrics(test_labels[key], y_test_predicted)
        printers.printsf('{0}{1} Region Performance{0}'.format(40*'-', key), standard_out_file)
        printers.printsf('Classifer Parameters', standard_out_file)
        for pn in sorted(pipeline.get_params()):
            printers.printsf("\t{0}: {1}".format(pn, pipeline.get_params()[pn]), standard_out_file)
        printers.printsf('Test Data Performance',standard_out_file)
        printers.printsfPerformanceMetrics(pm, standard_out_file)
        cm = sklearn.metrics.confusion_matrix(test_labels[key], y_test_predicted)
        printers.printTwoClassConfusion(cm, standard_out_file)

        print 'Persisting {0} region pipeline ................'.format(key)
        #persist the model
        output_classifier_path = output_path.format('{0}_classifier/'.format(key))
        if not os.path.exists(output_classifier_path):
            os.makedirs(output_classifier_path)

        clf_file = '{0}{1}'.format(output_classifier_path, '{0}_classifier.pkl'.format(key))
        joblib.dump(pipeline, clf_file)

        print '{0} region completed .................'.format(key)

    print 'Done'
