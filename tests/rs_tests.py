import json
import unittest
import sys,os
sys.path.append('..')
import rs
import pandas as pd
from numpy.random import RandomState
import numpy as np
from learn import wrangle

__author__ = 'Aaron J. Masino'

class RSTestCase(unittest.TestCase):

    def load_report(self, path):
        f = open(path,'r')
        text = reduce(lambda x,y: x+y, f.readlines(), "")
        f.close()
        return text

    def setUp(self):
        rs.app.config['TESTING'] = True
        rs.configure_service('resources/config/test_config.cfg')
        self.app = rs.app.test_client()

    def tearDown(self):
        pass

    def test_classify_request(self):
        data = {
            'pid1':'Radiology report text 1',
            'pid2':'other radiology report text',
            'pid3':'and some more report text'
        }
        request_body = json.dumps(data)
        rv = self.app.post('/classify',data=request_body, content_type='application/json')
        rdata = json.loads(rv.data)
        self.assertEqual(len(data),len(rdata), rdata)
        self.assertEqual(type(data),type(rdata),type(rdata))
        for k,v in rdata.items():
            self.assertEqual(len(v),4,'Each return record should be a 4 tuple')

    def test_end2end_known_test_data(self):
        if rs.app.config['RUN_TESTS']:
            # training/test data and output files
            #label_file = '../data/input/SDS_PV2_combined/SDS_PV2_class_labels.txt'
            label_file = rs.app.config['LABEL_FILE']
            #self.report_path = '../data/input/SDS_PV2_combined/reports'
            #self.report_path = rs.app.config['TEXT_REPORT_DIR']
            label_data = pd.read_csv(label_file)
            key_start = int(rs.app.config['REGION_COL_START'])
            key_stop = int(rs.app.config['REGION_COL_STOP'])+1
            region_keys = label_data.columns[key_start:key_stop]
            # set the numpy random seed so results are reproducible
            randstate = RandomState(987654321)

            # partition the data
            pos_cases, neg_cases = wrangle.partion(label_data['doc_norm']==1, label_data, ratios=[0.8,0.2])
            test_mask = np.concatenate((pos_cases[1], neg_cases[1]))
            randstate.shuffle(test_mask)
            test_labels = label_data.iloc[test_mask]
            #report_path = '../data/input/SDS_PV2_combined/reports'
            report_path = rs.app.config['TEXT_REPORT_DIR']
            test_reports = [self.load_report('{0}/{1}.txt'.format(report_path, pid)) for pid in test_labels['pid']]
            min_acc = float(rs.app.config['MIN_ACCURACY'])

            # send reports individually as multiple requests
            accuracy = [0,0,0,0]
            region_labels = ['inner','middle', 'outer', 'mastoid']
            for idx, tpl in enumerate(zip(test_labels['pid'],test_reports)):
                data = {tpl[0]:tpl[1]}
                request_body = json.dumps(data)
                rv = self.app.post('/classify',data=request_body, content_type='application/json')
                rdata = json.loads(rv.data)
                for jdx,label in enumerate(region_labels):
                    act = test_labels[label].iloc[idx]
                    pred = rdata[tpl[0]][jdx]
                    if act == pred:
                        accuracy[jdx] += 1

            accuracy = [v/float(len(test_labels)) for v in accuracy]
            for v in accuracy:
                self.assertGreater(v, min_acc, 'Failed accuracy on individual post test {0}'.format(accuracy))

            #send reports in one batch requiest
            accuracy = [0,0,0,0]
            region_labels = ['inner','middle', 'outer', 'mastoid']
            data = {}
            for idx, tpl in enumerate(zip(test_labels['pid'],test_reports)):
                data[tpl[0]]=tpl[1]
            request_body = json.dumps(data)
            rv = self.app.post('/classify',data=request_body, content_type='application/json')
            rdata = json.loads(rv.data)
            for idx,pid in enumerate(test_labels['pid']):
                for jdx,label in enumerate(region_labels):
                    act = test_labels[label].iloc[idx]
                    pred = rdata[pid][jdx]
                    if act == pred:
                        accuracy[jdx] += 1

            accuracy = [v/float(len(test_labels)) for v in accuracy]
            for v in accuracy:
                self.assertGreater(v, min_acc, 'Failed accuracy on batch post test {0}'.format(accuracy))


if __name__ == '__main__':
    unittest.main()