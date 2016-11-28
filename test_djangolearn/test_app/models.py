# -*- coding: utf-8 -*-

from sklearn import svm
from djangolearn.models import SciKitLearnModel
from sklearn import datasets
from sklearn import linear_model



class IrisSVMModel(SciKitLearnModel):

    def train(self, x, y):
        clf = svm.SVC()
        clf.fit(x, y)
        self.store(clf)

    def evaluate(self, x):
        # this is bad don't load the model at every run
        restored = self.restore()
        restored_results = restored.predict(x)
        return restored_results


class LinearRegressionModel(SciKitLearnModel):

    def train(self, x, y):
        lr = linear_model.LinearRegression()
        lr.fit(x, y)
        self.store(lr)

    def evaluate(self, x):
        # this is bad don't load the model at every run
        restored = self.restore()
        restored_results = restored.predict(x)
        return restored_results
