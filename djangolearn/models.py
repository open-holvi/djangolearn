# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import pickle
import sklearn
import logging
import shutil
try:
    from tempfile import TemporaryDirectory
except:
    from .py3_utils import TemporaryDirectory

from django.core.files.storage import default_storage

from django.db import models
from django.core.files import File
from django.conf import settings

from djangolearn.exceptions import (
    NotMachineLearningModelException, FrameworkVersionCollisionException,
    FeaturesVersionCollisionException, ModelNotStoredException)

from sklearn.base import BaseEstimator
from sklearn.externals import joblib

logger = logging.getLogger(__name__)

storage_method = getattr(settings, "DJANGOLEARN_STORAGE", None)

if not storage_method:
    logger.debug("DjangoLearn using default django storage method!")
    storage_method = default_storage
else:
    logger.debug("DjangoLearn using custom django storage method %s!" % storage_method)
    storage_method = __import__(storage_method, fromlist=[''])


def model_storage_path(instance, filename):
    # file will be uploaded to MEDIA_ROOT/model/ModelID
    return 'djangolearn/{0}/v{1}__{2}'.format(
            instance.model_handle.id, instance.version, filename)

class MachineLearningModelFileStorage(models.Model):
    """
    A model storage is the method of storing the machine learning model.
    It uses Django FileField for this purpuse, and leverages django-storages
    to indicate to where should the model be saved. Eg. S3/Azure/google
    """

    # File identifier (eg. file name)
    identifier = models.CharField(
        max_length=128, blank=False, null=False)

    # Is header (SciKitLearn uses a file as a header and sequencial numpy
    # arrays. The header has to be specified as the entrypoint)
    is_header = models.BooleanField(default=False)

    # Is active is used in case a model is retrained, and the storage updated.
    active = models.BooleanField(default=True)

    # Is active is used in case a model is retrained, and the storage updated.
    # Version is here to fetch previous versions of the trained model
    version = models.IntegerField(default=1)

    # The handle of the machine learning model the storage belongs to
    model_handle = models.ForeignKey(
        'MachineLearningModel', null=False, related_name='storage_set')

    # The underlying framework version of the model
    framework_version = models.CharField(
        max_length=128, blank=False, null=False)

    # The version of the features framework used to get features
    features_version = models.CharField(
        max_length=128, blank=False, null=True)

    # The file containg the binaries.
    payload = models.FileField(
        upload_to=model_storage_path,
        storage=storage_method)

    def get_payload(self):
        return self.payload.read()


class MachineLearningModelSerialiser(object):
    """A serialiser that takes a machine learning model and sends to storage"""

    def __init__(self, model_object, framework_version, features_version, *args, **kwargs):
        self.model_object = model_object
        self.framework_version = framework_version
        self.features_version = features_version

    def store(self, model, *args, **kwargs):
        raise NotImplementedError("")

    def restore(self, model, *args, **kwargs):
        raise NotImplementedError("")


class ScikitJobLibModelSerialiser(MachineLearningModelSerialiser):
    """
    A serialiser that takes a SciKitLearn model and serialises it via
    JobLib (since Pickle is very insecure). It then uses storage_method
    to persistently store the binaries.
    """

    storage_model = MachineLearningModelFileStorage

    def __init__(self, model_object, *args, **kwargs):
        super(ScikitJobLibModelSerialiser, self).__init__(model_object, *args, **kwargs)

    def store(self, trained_model):

        # hardcoded for now
        model_file_name = 'model.pkl'

        assert '/' not in model_file_name


        prev_storage_obj = self.storage_model.objects.filter(
            active=True,
            model_handle=self.model_object,
        )

        if prev_storage_obj:
            version = prev_storage_obj.first().version
            prev_storage_obj.update(active=False)
            logger.debug(
                "Found previously stored model of version %s" %
                version)
            version = version + 1
        else:
            version = 1

        # Create tempdir
        with TemporaryDirectory() as tmp_dir:
            # save model & get list of all files
            file_names = joblib.dump(trained_model, tmp_dir + '/' + model_file_name)
            # upload files

            for file_name in file_names:
                with open(file_name, mode='ab+') as file_content:
                    file_name = file_name.split('/')[-1]
                    if file_name == model_file_name:
                        is_header = True
                    else:
                        is_header = False

                    storage_obj = self.storage_model.objects.create(
                        version=version,
                        identifier=file_name,
                        payload=File(file_content),
                        model_handle=self.model_object,
                        is_header=is_header,
                        framework_version=self.framework_version,
                        features_version=self.features_version,
                    )


    def restore(self, version=None):
        # hardcoded for now
        model_file_name = 'model.pkl'
        assert '/' not in model_file_name

        # get all files
        if version:
            files = self.storage_model.objects.filter(
                model_handle=self.model_object,
                version=version)
        else:
            files = self.storage_model.objects.filter(
                model_handle=self.model_object,
                active=True)

        if not files:
            raise ModelNotStoredException(
                "No model stored for %s!" % self.model_object)
        else:
            f = files.first()
            # Warning - We will only compare if these arguments were provided
            # If not, validation will be bypassed
            if f.framework_version and self.framework_version and f.framework_version != self.framework_version:
                raise FrameworkVersionCollisionException(
                    "Cannot load a model from a different framework version. "
                    "please retrain the model!")

            elif f.features_version and self.features_version and f.features_version != self.features_version:
                raise FeaturesVersionCollisionException(
                    "Cannot (Or should not) load a model trained with "
                    "different features."
                )

        # Create tempdir
        with TemporaryDirectory() as tmp_dir:

            for storage_file in files:

                logger.debug("Reconstucting %s" % tmp_dir+'/'+storage_file.identifier)
                fs_file = open(tmp_dir+'/'+storage_file.identifier, mode='ab+')
                fs_file.write(storage_file.get_payload())
                fs_file.seek(0)

            # reconstruct
            clf = joblib.load(tmp_dir + '/' + model_file_name)

        return clf


class MachineLearningModel(models.Model):
    # This is a meta class.
    serialiser = MachineLearningModelSerialiser
    loaded_model = None

    model_class_name = models.CharField(
        max_length=128, blank=False, null=False)

    create_time = models.DateTimeField(auto_now_add=True)
    update_time = models.DateTimeField(auto_now=True)

    class Meta:
        get_latest_by = "create_time"
        ordering = ('-create_time',)

    def store(self, model, features_version=None, framework_version=None, *args, **kwargs):

        if not self.id:
            try:
                self.refresh_from_db()
            except:
                # compatibility with Django < 1.8
                self.save()

        # Lets double check:
        if not self.id:
            raise Exception(
                "Cannot store a model without an ID.",
                "please reload the model")

        serialiser_instance = self.serialiser(
            self,
            features_version=features_version,
            framework_version=framework_version)
        serialiser_instance.store(model)

    def restore(self, features_version=None, framework_version=None, version=None):
        serialiser_instance = self.serialiser(
            self,
            features_version=features_version,
            framework_version=framework_version)
        self.loaded_model = serialiser_instance.restore(version=version)
        return self.loaded_model


class SciKitLearnModel(MachineLearningModel):
    """A SciKitLearn Machine learning model persistance model"""

    serialiser = ScikitJobLibModelSerialiser

    def __init__(self, *args, **kwargs):
        super(SciKitLearnModel, self).__init__(*args, **kwargs)
        self.scikit_version = sklearn.__version__

    def store(self, model, *args, **kwargs):
        logger.info("Storing Model %s..." % self)

        if not issubclass(model.__class__, BaseEstimator):
            raise NotMachineLearningModelException(
                "Machine Learning Model is not a SciKitLearn Model")

        super(SciKitLearnModel, self).store(
            model, framework_version=self.scikit_version, *args, **kwargs)

    def restore(self, *args, **kwargs):
        logger.info("Restoring Model %s..." % self)

        result = super(SciKitLearnModel, self).restore(
            framework_version=self.scikit_version,
            *args, **kwargs)

        if not issubclass(result.__class__, BaseEstimator):
            raise NotMachineLearningModelException(
                "Restored model is not a SciKitLearn Model")

        return result

    def __str__(self):
        return "SciKit Learn Machine Learning Model %s" % (
                self.id
            )
