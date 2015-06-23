**A**udgenDB **R**adiology **R**eport **C**lassification (arrc)

## Model Selection
The [*model_analysis.py*](https://github.com/chop-dbhi/arrc/blob/master/model_analysis.py) file analyizes a number of classification models using the [scikit-learn API](http://scikit-learn.org/stable/) and the [NLTK API](http://www.nltk.org/). It performs a grid search over several model hyper-parameters applying k-fold cross validation to select the best models. Performance is subsequently evaluated on a hold-out test set. 

As written, the [*model_analysis.py*](https://github.com/chop-dbhi/arrc/blob/master/model_analysis.py) file requries a label file that contains column headers *pid*, *doc_norm*, *inner*, *middle*, *outer*, and *mastoid* in that order. The *pid* column is a unique identifier for that corresponds to a text report file named *pid.txt*. The *doc_norm* column is binary valued and indicates if *pid.txt* is contains NO abnormalities (0) or at least one abnormality (1). The *inner*, *middle*, *outer*, and *mastoid* columns are also binary valued and indicate if *pid.txt* contains an no abnormality (0) or at least one abnormality (1) in the inner, middle, outer ear or mastoid regions respectively. 

### Training & Test Data
Training & test data can be obtained from the [AudGenDB project](http://audgendb.chop.edu/) via the [AudGenDB application](https://audgendb.chop.edu/app/login/). 

## Model Persistence 
To avoid re-training the classification models every time the REST service is started (see below) the selected classification models can be persisted via pickling. The [*model_persist.py*](https://github.com/chop-dbhi/arrc/blob/master/model_persist.py) will train and pickle the models using the hyper-parameters specified in the *INSTALL_DIR/resources/config/models.ini* file. See this [sample file](https://github.com/chop-dbhi/arrc/wiki/Sample-models.ini). 

## REST Service
A simple [Flask](http://flask.pocoo.org/) REST service is availble as an example document labeling service. The REST service utilizes four scikit-learn based classifiers to classify radiology text reports relative to presence/absence of an abnormality in the inner, middel, outer, or mastoid ear regions. 

### Configuration
A configuration file is expected as INSTALL_DIR/resources/config/app_config.cfg where INSTALL_DIR is the directory containing this source code. The configuration file has four entries that are the paths to the model persistance files. See this [sample file](https://github.com/chop-dbhi/arrc/wiki/Sample-REST-service-configuration-file)

### Usage
Start the REST service:

    python rs.py

The service has a single endpoint at http://HOST/classify where HOST is where the application is hosted. If running locally with Flask, HOST is http://localhost:5000 by default. This endpoint handles only POST requests. The body of the request must be JSON formated with entries 

    {"id1":"text1", "id2":"text2"}
    
The return result is also JSON formated in the form

    {"id1":[v1,v2,v3,v4], "id2":[v1,v2,v3,v4]}
    
where v1-v4 are binary (0,1) values corresponding to the classifications for the inner, middle, outer, and mastoid regions, respectively. 
