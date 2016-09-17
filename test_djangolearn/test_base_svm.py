# encoding: utf-8

from __future__ import absolute_import, division, print_function, unicode_literals

from django.test import TestCase
from sklearn import svm
from sklearn import datasets

from djangolearn.models import SciKitLearnModel

class SVMBaseTestCase(TestCase):
    def setUp(self):
        self.clf = svm.SVC()
        iris = datasets.load_iris()
        self.X, self.y = iris.data, iris.target
        self.clf.fit(self.X, self.y)
        self.expected_results = self.clf.predict(self.X[0:1])

    def test_model_storage_init(self):
        scikit_model = SciKitLearnModel.objects.create()
        scikit_model.store(self.clf)

        restored = scikit_model.restore()
        restored_results = restored.predict(self.X[0:1])
        self.assertEquals(self.expected_results, restored_results)
