#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals

from django.test import TestCase
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn import linear_model
from test_djangolearn.test_app.models import IrisSVMModel, LinearRegressionModel

iris = datasets.load_iris()

boston = datasets.load_boston()


class InheritanceTestCase(TestCase):
    """Test that a model inheriting from SciKitLearnModel works"""
    def setUp(self):
        clf = svm.SVC()
        X, y = iris.data, iris.target
        clf.fit(X, y)
        self.expected_results = clf.predict(X[0:1])

    def test_model_inheritance_basic(self):
        X, y = iris.data, iris.target

        model = IrisSVMModel.objects.create()
        model.train(X, y)
        results = model.evaluate(X[0:1])
        self.assertEquals(self.expected_results, results)



class MultipleObjectsInheritanceTestCase(TestCase):
    """Test that we segregate the storage layer correctly for different models"""
    def setUp(self):
        # For SVN
        clf = svm.SVC()
        X, y = iris.data, iris.target
        clf.fit(X, y)
        self.svm_expected_results = clf.predict(X[0:1])

        # For Clustering
        lr = linear_model.LinearRegression()
        lr.fit(boston.data, boston.target)

        self.linear_expected = lr.predict(boston.data)


    def test_multiple_model_basic(self):
        """ Test that we don't mess up two models"""
        X, y = iris.data, iris.target

        svm_model = IrisSVMModel.objects.create()
        svm_model.train(X, y)

        lr_model = LinearRegressionModel.objects.create()
        lr_model.train(boston.data, boston.target)

        svm_results = svm_model.evaluate(X[0:1])
        self.assertEquals(self.svm_expected_results, svm_results)


        lr_results = lr_model.evaluate(boston.data)
        # Pick one sample at random
        self.assertEquals(self.linear_expected[10], lr_results[10])
