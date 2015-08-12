import os
import sys
import json
from sklearn.externals import joblib
from flask import Flask, request

__author__ = 'Aaron J. Masino'

# CREATE THE FLASK APP
app = Flask(__name__)

# CREATE MODEL VARIABLES
__inner_clf__ = None
__middle_clf__ = None
__outer_clf__ = None
__mastoid_clf__ = None


def configure_service(config_file):
    global __inner_clf__
    global __middle_clf__
    global __outer_clf__
    global __mastoid_clf__

    app.config.from_pyfile(config_file)

    __inner_clf__ = joblib.load(app.config['INNER_PKL'])
    __middle_clf__ = joblib.load(app.config['MIDDLE_PKL'])
    __outer_clf__ = joblib.load(app.config['OUTER_PKL'])
    __mastoid_clf__ = joblib.load(app.config['MASTOID_PKL'])


# SERVICE REQUESTS
@app.route('/classify', methods=['POST'])
def classify():
    data = request.get_json()

    if data:
        rd = {}

        for pid, text in data.items():
            # classify text
            inner = __inner_clf__.predict([text])[0]
            middle = __middle_clf__.predict([text])[0]
            outer = __outer_clf__.predict([text])[0]
            mastoid = __mastoid_clf__.predict([text])[0]
            rd[pid] = (inner, middle, outer, mastoid)

        return json.dumps(rd)


if __name__ == '__main__':
    host = None
    port = None
    debug = os.environ.get('DEBUG') is not None

    args = sys.argv[1:]

    if len(args) > 0:
        addr = args[0].split(':', 1)

        host = addr[0]

        if len(addr) > 1:
            port = int(addr[1])

    configure_service('resources/config/app_config.cfg')

    app.run(host=host, port=port, debug=debug)
