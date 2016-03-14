__author__ = 'Aaron J. Masino'

import time
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV


def grid_analysis(pipeline, parameters, train_input, train_labels, cv = None):
    if not cv:
        cv = StratifiedKFold(train_labels, n_folds=5, shuffle=False)
    grid_search = GridSearchCV(pipeline, parameters, scoring='accuracy', cv=cv, refit=True, verbose = 1, n_jobs=2)
    print("Performing grid search...")
    tic = time.clock()
    grid_search.fit(train_input,train_labels)
    toc = time.clock()
    print("Grid search complete in {0} sec".format(toc-tic))
    return grid_search

class SparseToArray:
    '''transforms sparse array to normal array'''
    def fit(self,X,y, **params):
        #does nothing
        return self

    def transform(self,sparseX):
        return sparseX.toarray()

    def get_params(self,deep):
        return {}
