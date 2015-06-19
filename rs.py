import json
import os
import sklearn
from sklearn.externals import joblib
from flask import Flask,request, session, g, redirect, url_for, \
    abort, render_template, flash

__author__ = 'Aaron J. Masino'

#CREATE THE FLASK APP
app = Flask(__name__)

#CREATE MODEL VARIABLES
inner_clf = None
middle_clf = None
outer_clf = None
mastoid_clf = None

def configure_service(config_file):
    app.config.from_pyfile(config_file)
    inner_clf = joblib.load(app.config['INNER_PKL'])

# SERVICE REQUESTS
@app.route('/classify', methods=['POST'])
def classify():
    data = request.get_json()
    if data:
        rd = {}
        for pid,text in data.items():
            #classify text
            inner = inner_clf.predict(text)[0]
            rd[pid] = (inner,1,1,0)
    return json.dumps(rd)

if __name__ == '__main__':
    configure_service('app_config.cfg')
    app.run()