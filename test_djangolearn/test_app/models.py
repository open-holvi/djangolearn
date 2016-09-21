from sklearn import svm
from djangolearn import SciKitLearnModel
from sklearn import datasets

class IrisModel(SciKitLearnModel):

    def train(self):
        iris = datasets.load_iris()
        clf = svm.SVC()
        X, y = iris.data, iris.target
        clf.fit(X, y)
        self.store(clf)

    def evaluate(self, x):
        # this is bad don't load the model at every run
        restored = self.restore()
        restored_results = restored.predict(x)
        return restored_results
