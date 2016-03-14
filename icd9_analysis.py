__author__ = 'Aaron J. Masino'
import numpy as np
from numpy.random import RandomState
import pandas as pd
from learn import printers, wrangle
from learn.metrics import PerformanceMetrics
import sklearn


if __name__ == '__main__':

    icd9_patient_file = './data/input/SDS_PV2_combined/icd9/SDS_PV2_icd9.csv'
    icd9_ear_file = './data/input/SDS_PV2_combined/icd9/ICD9CodeDescriptions.txt'
    standard_out_file = './data/output/SDS_PV2_icd9_results.txt'
    region_keys = ['inner', 'middle', 'outer', 'mastoid']

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

    #import icd9 ear codes
    icd9_ear = {}
    with open(icd9_ear_file, 'r') as f:
        key = ""
        for line in f.readlines():
            if line.startswith("#"):
                key = line[1:].strip('\n')
            else:
                l = icd9_ear.get(key,[])
                v = line.split(",")[0]
                l.append(v)
                icd9_ear[key] = l

    #import codes assigned to each patient
    patient_codes = {}
    with open(icd9_patient_file, 'r') as f:
        f.readline() #skip header
        for line in f.readlines():
            pid,code = line.split(',')
            code = code.strip('\r\n')
            l = patient_codes.get(pid,[])
            l.append(code)
            patient_codes[pid]=l

    #create empty patient array to hold predicted values
    num_patients = len(test_labels)
    patients = np.empty((num_patients,), dtype=[('pid','S7'),('inner','i4'),('middle','i4'),('outer','i4'),('mastoid','i4')])

    #initialize patients array
    for k in region_keys:
        patients[k] = 0

    #get predicted values based on icd9 codes
    cnt = 0
    for _ , row in test_labels.iterrows():
        pid = row['pid']
        patients['pid'][cnt] = pid
        codes = patient_codes[pid]
        for code in codes:
            if code in icd9_ear['inner']:patients['pid'][cnt] = 1
            if code in icd9_ear['middle']: patients['inner'][cnt] = 1
            if code in icd9_ear['outer']: patients['middle'][cnt] = 1
            if code in icd9_ear['mastoid']: patients['mastoid'][cnt] = 1
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
