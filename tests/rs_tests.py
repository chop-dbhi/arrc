import json
import unittest
import sys,os
sys.path.append('..')
import rs

__author__ = 'Aaron J. Masino'

class RSTestCase(unittest.TestCase):

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

if __name__ == '__main__':
    unittest.main()