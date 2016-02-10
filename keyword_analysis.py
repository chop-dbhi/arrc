__author__ = 'Aaron J. Masino'

import numpy as np
from numpy.random import RandomState
import pandas as pd
from learn import printers, wrangle
from learn.metrics import PerformanceMetrics
import sklearn
from functools import reduce

def load_report(path):
    f = open(path,'r')
    text = reduce(lambda x,y: x+y, f.readlines(), "")
    f.close()
    return text

if __name__ == '__main__':

    keyword_file = './data/input/SDS_PV2_combined/keywords/keywords.txt'
    standard_out_file = './data/output/SDS_PV2_keyword_results.txt'
    region_keys = ['inner', 'middle', 'outer', 'mastoid']

    report_path = './data/input/SDS_PV2_combined/reports_single_find_impr'


    #load test set data - same set used for ML tests
    seed = 987654321
    # set the numpy random seed so results are reproducible
    rs = RandomState(987654321)

    # set common path variables
    label_file = './data/input/SDS_PV2_combined/SDS_PV2_class_labels.txt'

    # read data
    label_data = pd.read_csv(label_file)

    # partition the data
    pos_cases, neg_cases = wrangle.partion(label_data['doc_norm']==1, label_data, ratios=[0.8,0.2])
    train_mask = np.concatenate((pos_cases[0], neg_cases[0]))
    test_mask = np.concatenate((pos_cases[1], neg_cases[1]))
    rs.shuffle(train_mask)
    rs.shuffle(test_mask)
    train_labels = label_data.iloc[train_mask]
    test_labels = label_data.iloc[test_mask]
    # read in the text reports
    train_reports = [load_report('{0}/{1}_fi.txt'.format(report_path, pid)) for pid in train_labels['pid']]
    test_reports = [load_report('{0}/{1}_fi.txt'.format(report_path, pid)) for pid in test_labels['pid']]

    #import keywords
    keywords = {}
    with open(keyword_file, 'r') as f:
        key = ""
        for line in f.readlines():
            if line.startswith("#"):
                key = line[1:].strip('\n')
            else:
                l = keywords.get(key,[])
                v = line.split(",")[0]
                l.append(v)
                keywords[key] = l


    #create empty patient array to hold predicted values
    num_patients = len(test_labels)
    patients = np.empty((num_patients,), dtype=[('pid','S7'),('inner','i4'),('middle','i4'),('outer','i4'),('mastoid','i4')])

    #initialize patients array
    for k in region_keys:
        patients[k] = 0

    #get patient values based on icd9 codes
    cnt = 0
    for _ , row in test_labels.iterrows():
        pid = row['pid']
        patients['pid'][cnt] = pid
        report = test_reports[cnt]
        for region in region_keys:
            for keyword in keywords[region]:
                if keyword in report: patients[region][cnt] = 1
        cnt += 1


    #compare predicted and actual
    for k in region_keys:
        printers.printsf('{0}Analysis for {1} ear region{0}'.format(40*'-', k), standard_out_file)
        y_pred = patients[k]
        y_act = test_labels[k]
        pm = PerformanceMetrics(y_act, y_pred)
        printers.printsfPerformanceMetrics(pm, standard_out_file)
        cm = sklearn.metrics.confusion_matrix(y_act, y_pred)
        printers.printTwoClassConfusion(cm, standard_out_file)

