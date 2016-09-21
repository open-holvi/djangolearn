# encoding: utf-8

from __future__ import absolute_import, division, print_function, unicode_literals

from django.test import TestCase
from sklearn import svm
from sklearn import datasets

from test_djangolearn.test_app.models import IrisModel

iris = datasets.load_iris()

class InheritanceTestCase(TestCase):
    """Test that a model inheriting from SciKitLearnModel works"""
    def setUp(self):
        clf = svm.SVC()
        X, y = iris.data, iris.target
        clf.fit(X, y)
        self.expected_results = clf.predict(X[0:1])

    def test_model_storage_init(self):
        X, y = iris.data, iris.target

        model = IrisModel.objects.create()
        model.train()
        results = model.evaluate(X[0:1])
        self.assertEquals(self.expected_results, results)
