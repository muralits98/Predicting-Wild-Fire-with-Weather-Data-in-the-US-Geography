import numpy
import xgboost
import sklearn.base
import sklearn.tree
import sklearn.linear_model
import sklearn.kernel_approximation
import sklearn.model_selection

class XDSClassifier(sklearn.base.BaseEstimator,
                    sklearn.base.ClassifierMixin):
    def __init__(self, n_ensemble = 5, n_components = 100):
        self.n_ensemble = n_ensemble
        self.n_components = n_components
    def fit(self, X, y):
        self.estimators = []
        self.transformer = sklearn.kernel_approximation\
            .Nystroem(n_components = self.n_components)
        pos_ind, neg_ind = y > 0, y == 0
        X_pos, y_pos = X[pos_ind], y[pos_ind]
        X_neg, y_neg = X[neg_ind], y[neg_ind]
        self.estimators.append(xgboost.XGBClassifier(n_estimators = 25)
            .fit(X, y))
        for i in range(self.n_ensemble):
            X_neg_cur, _, y_neg_cur, _ =\
            sklearn.model_selection.train_test_split(X_neg,
                y_neg, train_size = len(y_pos))
            X_tr = numpy.concatenate((X_pos, X_neg_cur), axis = 0)
            X_tr_kernel = self.transformer.fit_transform(X_tr)
            y_tr = numpy.concatenate((y_pos, y_neg_cur), axis = 0)
            est =\
            sklearn.linear_model.SGDClassifier().fit(X_tr_kernel, y_tr)
            self.estimators.append(est)
            est =\
            sklearn.tree.DecisionTreeClassifier()\
                .fit(X_tr, y_tr)
            self.estimators.append(est)

        return self
    def predict(self, X):
        X_transformed = self.transformer.fit_transform(X)
        return numpy.mean([
            j.predict(X if not i % 2 else X_transformed)
            for i, j in enumerate(self.estimators)], axis = 0) > 0.5
