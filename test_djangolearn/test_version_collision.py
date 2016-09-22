#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals

from django.test import TestCase
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn import datasets

from test_djangolearn.test_app.models import IrisSVMModel
from djangolearn.models import SciKitLearnModel
from djangolearn.exceptions import FeaturesVersionCollisionException, FrameworkVersionCollisionException

iris = datasets.load_iris()

class FrameworkCollisionTestCase(TestCase):
    """Test that we pickip a framework collision"""
    def setUp(self):
        clf = svm.SVC()
        X, y = iris.data, iris.target
        clf.fit(X, y)
        self.expected_results = clf.predict(X[0:1])

    def test_framework_collision(self):
        X, y = iris.data, iris.target

        model = IrisSVMModel.objects.create()
        model.train(X, y)

        model.scikit_version = '0.0000'
        model.save()

        with self.assertRaises(FrameworkVersionCollisionException):
            results = model.evaluate(X[0:1])


class DataFrameworkCollisionTestCase(TestCase):
    """Test that we pickip a framework collision"""
    def setUp(self):
        self.clf = svm.SVC()
        X, y = iris.data, iris.target
        self.clf.fit(X, y)
        self.expected_results = self.clf.predict(X[0:1])

    def test_data_framework_collision(self):
        scikit_model = SciKitLearnModel.objects.create()
        scikit_model.store(self.clf, features_version='1')

        with self.assertRaises(FeaturesVersionCollisionException):
            restored = scikit_model.restore(features_version='3')


    def test_data_framework_bypass_one(self):

        scikit_model = SciKitLearnModel.objects.create()
        # We should bypass validation because we did not provide a version
        scikit_model.store(self.clf)

        try:
            restored = scikit_model.restore(features_version='3')
        except (FeaturesVersionCollisionException):
            self.fail()


    def test_data_framework_bypass_two(self):

        scikit_model = SciKitLearnModel.objects.create()
        scikit_model.store(self.clf, features_version='1')

        try:
            # We should bypass validation because we did not provide a version
            restored = scikit_model.restore()
        except (FeaturesVersionCollisionException):
            self.fail()
