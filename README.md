# Django-Learn
### A simplistic SciKitLearn Machine learning model persistence layer for django.

[![Build Status](https://travis-ci.org/open-holvi/djangolearn.svg?branch=master)](https://travis-ci.org/open-holvi/djangolearn) [![Codacy Badge](https://api.codacy.com/project/badge/Grade/bd14b53e401548eb8324afa414f15f9c)](https://www.codacy.com/app/sergio_4/djangolearn?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=open-holvi/djangolearn&amp;utm_campaign=Badge_Grade)

## Basic Usage
```py
from sklearn import svm
from sklearn import datasets
from djangolearn.models import SciKitLearnModel

iris = datasets.load_iris()

class IrisModel(SciKitLearnModel):

    def train(self, X, Y):
        clf = svm.SVC()

        clf.fit(X, Y)
        self.store(clf)

    def evaluate(self, x):
        # This is an example. Don't do this at request time
        # Loading the model takes a bit
        restored = self.restore()
        restored_results = restored.predict(x)

X, y = iris.data, iris.target

instance = IrisModel.objects.create()
instance.train(X, y)
instance.evaluate(X[0:1])
```

## Storage:

Django Learn uses [Django Storages](https://github.com/jschneier/django-storages)

If you want to specify a different storage system than the default one,
specify it through the setting:

```py
DJANGOLEARN_STORAGE = 'storages.backends.s3boto.S3BotoStorage'
```

Note that you will need to do the configuration management mentioned in the
Django Storage page.

## What's under the hood?

One simple solution would be to Pickle the model. [That's quite insecure](http://pyvideo.org/pycon-us-2014/pickles-are-for-delis-not-software.html).
For that reason we use joblib to store the models. There's a bit of validation sauce,
like making sure that the scikit version is the same as the one used to
train the system. There is also the option to specify the version of the
code/library used to pre-process the data, so that train data and evaluation
data have the same features.

## Security
Every object de-serialization is dangerous. Only store models you trust and
keep your model storage secure.
We try to make basic validations of the models, but python can only go so far. Use at your own risk.

## Performance
This is supposed to be an MVP for small jobs, and for playing around with machine learning in Django.
Loading models at request time is not recomemdend.

## Requirements
* Python 2.7 3.4 3.5
* Django > 1.7 (tests being deprecated for 1.7)
* SciKitLearn >= 0.15
