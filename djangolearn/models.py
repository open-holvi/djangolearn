import pickle
import sklearn
import logging
import tempfile
import shutil
from django.core.files.storage import default_storage


from django.db import models
from django.core.files import File
from django.conf import settings

from djangolearn.exceptions import (
    NotMachineLearningModelException, FrameworkVersionCollisionException,
    DataProcessingVersionCollisionException, ModelNotStoredException)

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
    version = models.IntegerField (default=1)

    # The handle of the machine learning model the storage belongs to
    model_handle = models.ForeignKey(
        'MachineLearningModel', null=False, related_name='storage_set')

    # The file containg the binaries.
    payload = models.FileField(
        upload_to=model_storage_path,
        storage=storage_method)

    def get_payload(self):
        return self.payload.read()


class MachineLearningModelSerialiser(object):
    """A serialiser that takes a machine learning model and sends to storage"""

    def __init__(self, model_object, *args, **kwargs):
        self.model_object = model_object

    def store(self, model):
        raise NotImplementedError("")

    def restore(self, model):
        raise NotImplementedError("")


class ScikitJobLibModelSerialiser(MachineLearningModelSerialiser):
    """
    A serialiser that takes a SciKitLearn model and serialises it via
    JobLib (since Pickle is very insecure). It then uses storage_method
    to persistently store the binaries.
    """

    storage_method = MachineLearningModelFileStorage

    def __init__(self, model_object, *args, **kwargs):
        super(ScikitJobLibModelSerialiser, self).__init__(model_object, *args, **kwargs)

    def store(self, trained_model):

        # hardcoded for now
        model_file_name = 'model.pkl'

        assert '/' not in model_file_name

        # Create tempdir
        tmp_dir = tempfile.mkdtemp()

        # save model & get list of all files
        file_names = joblib.dump(trained_model, tmp_dir + '/' + model_file_name)
        # upload files

        prev_storage_obj = self.storage_method.objects.filter(
            active=True,
            model_handle=self.model_object,
        )

        if prev_storage_obj:
            version = prev_storage_obj.first().version
            prev_storage_obj.update(active=False)
            version = version + 1
        else:
            version = 1

        for file_name in file_names:
            with open(file_name, 'r') as file_content:
                file_name = file_name.split('/')[-1]
                if file_name == model_file_name:
                    is_header = True
                else:
                    is_header = False

                storage_obj = self.storage_method.objects.create(
                    version=version,
                    identifier=file_name,
                    payload=File(file=file_content, name=file_name),
                    model_handle=self.model_object,
                    is_header=is_header,
                )

        shutil.rmtree(tmp_dir)

    def restore(self, version=None):
        # hardcoded for now
        model_file_name = 'model.pkl'
        assert '/' not in model_file_name

        # get all files
        if version:
            files = self.storage_method.objects.filter(
                model_handle=self.model_object,
                version=version)
        else:
            files = self.storage_method.objects.filter(
                model_handle=self.model_object,
                active=True)

        if not files:
            raise ModelNotStoredException(
                "No model stored for %s!" % self.model_object)

        # Create tempdir
        tmp_dir = tempfile.mkdtemp()

        for storage_file in files:

            logger.debug("Reconstucting %s" % tmp_dir+'/'+storage_file.identifier)
            fs_file = open(tmp_dir+'/'+storage_file.identifier,'ab+')
            fs_file.write(storage_file.get_payload())
            fs_file.seek(0)

        clf = joblib.load(tmp_dir + '/' + model_file_name)
        # reconstruct
        shutil.rmtree(tmp_dir)
        return clf


class MachineLearningModel(models.Model):
    # This is a meta class.
    serialiser = MachineLearningModelSerialiser
    loaded_model = None

    model_class_name = models.CharField(
        max_length=128, blank=False, null=False)

    data_library_version = models.CharField(
        max_length=128, blank=True, default='')

    def save(self, *args, **kwargs):
        if not self.data_library_version:
            logger.warning(
                "You haven't specified the data transformation library version. "
                "It's good practice to make sure traning data and "
                "evaluation data are pre-processed by the same methods!")

        super(MachineLearningModel, self).save(*args, **kwargs)

    def store(self, model, data_library_version=None, *args, **kwargs):
        serialiser_instance = self.serialiser(self)
        serialiser_instance.store(model)

    def restore(self, version=None, data_library_version=None):

        if not self.data_library_version:
            logger.warning(
                "Model does not have a stored data library version. "
                "Bypassing validation...")

        elif self.data_library_version and not data_library_version:
            logger.warning(
                "Data processing library version not specified... "
                "Assuming installed version is %s. Proceed at your own risk"
                % self.data_library_version)

        elif self.data_library_version != data_library_version:
            raise DataProcessingVersionCollisionException(
                "Data processing library version %s of stored model %s does not"
                " match installed version %s" %
                (self.data_library_version, data_library_version)
            )

        serialiser_instance = self.serialiser(self)
        self.loaded_model = serialiser_instance.restore(version=version)
        return self.loaded_model


class SciKitLearnModel(MachineLearningModel):
    """A SciKitLearn Machine learning model persistance model"""

    serialiser = ScikitJobLibModelSerialiser

    scikit_version = models.CharField(
        max_length=128, blank=False, null=False)

    def __init__(self, *args, **kwargs):
        super(SciKitLearnModel, self).__init__(*args, **kwargs)
        self.scikit_version = sklearn.__version__

    def store(self, model, *args, **kwargs):
        logger.info("Storing Model %s..." % self)
        if not issubclass(model.__class__, BaseEstimator):
            raise NotMachineLearningModelException(
                "Machine Learning Model is not a SciKitLearn Model")
        super(SciKitLearnModel, self).store(
            model, self.data_library_version, *args, **kwargs)

    def restore(self, data_library_version=None, *args, **kwargs):
        logger.info("Restoring Model %s..." % self)
        if self.scikit_version != sklearn.__version__:
            raise FrameworkVersionCollisionException(
                "Scikit learn version %s of stored model %s does not match "
                "installed version %s " %
                (self.scikit_version, self, sklearn.__version__)
            )
        result = super(SciKitLearnModel, self).restore(
            data_library_version, *args, **kwargs)

        if not issubclass(result.__class__, BaseEstimator):
            raise NotMachineLearningModelException(
                "Restored model is not a SciKitLearn Model")
        return result

    def __str__(self):
        return "SciKit Learn Machine Learning Model with version %s" % (
                self.scikit_version
            )
