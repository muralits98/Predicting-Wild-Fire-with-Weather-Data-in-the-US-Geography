import numpy
import xgboost
import sklearn.ensemble
import sklearn.model_selection

class RandomForestClassifierBS(sklearn.ensemble.RandomForestClassifier):
    def __init__(self, n_boost = 5, n_estimators="warn",
                 criterion='gini', max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                 max_features='auto', max_leaf_nodes=None,
                 min_impurity_decrease=0.0, min_impurity_split=None,
                 bootstrap=True, oob_score=False, n_jobs=None,
                 random_state=None, verbose=0, warm_start=False,
                 class_weight=None):
        super().__init__(n_estimators=n_estimators,
                criterion=criterion, max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                min_weight_fraction_leaf=min_weight_fraction_leaf,
                max_features=max_features,
                max_leaf_nodes=max_leaf_nodes,
                min_impurity_decrease=min_impurity_decrease,
                min_impurity_split=min_impurity_split,
                bootstrap=bootstrap,
                oob_score=oob_score,
                n_jobs=n_jobs,
                random_state=random_state,
                verbose=verbose,
                warm_start=warm_start,
                class_weight=class_weight)
        self.n_boost = n_boost
        self.estimators = []
    def fit(self, X, y, sample_weight = None):
        pos_ind, neg_ind = y > 0, y == 0
        X_pos, y_pos = X[pos_ind], y[pos_ind]
        X_neg, y_neg = X[neg_ind], y[neg_ind]
        for i in range(self.n_boost):
            X_neg_cur, _, y_neg_cur, _ =\
            sklearn.model_selection.train_test_split(X_neg,
                y_neg, train_size = len(y_pos) + 1)
            X_tr = numpy.concatenate((X_pos, X_neg_cur), axis = 0)
            y_tr = numpy.concatenate((y_pos, y_neg_cur), axis = 0)
            est =\
            super().fit(X_tr, y_tr, sample_weight)
            self.estimators.append(est)
        return self
    def predict(self, X):
        return numpy.mean([
            sklearn.ensemble.RandomForestClassifier.predict(i, X)
            for i in self.estimators], axis = 0) > 0.5
    def predict_proba(self, X):
        return numpy.mean([
            sklearn.ensemble.RandomForestClassifier.predict_proba(i, X)
            for i in self.estimators], axis = 0)
    def predict_log_proba(self, X):
        return numpy.mean([
            sklearn.ensemble.RandomForestClassifier.predict_log_proba(i, X)
            for i in self.estimators], axis = 0)

class XGBClassifierBS(xgboost.XGBClassifier):
    def __init__(self, n_boost = 5, max_depth=3, learning_rate=0.1,
        n_estimators=100,
        verbosity=1, silent=None, objective='binary:logistic',
        booster='gbtree',
        n_jobs=1, nthread=None, gamma=0, min_child_weight=1, max_delta_step=0,
        subsample=1, colsample_bytree=1, colsample_bylevel=1,
        colsample_bynode=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
        base_score=0.5, random_state=0, seed=None, missing=None, **kwargs):
        super().__init__(max_depth=max_depth, learning_rate=learning_rate,
            n_estimators=n_estimators, verbosity=verbosity, silent=silent,
            objective=objective, booster=booster, n_jobs=n_jobs,
            nthread=nthread, gamma=gamma, min_child_weight=min_child_weight,
            max_delta_step=max_delta_step, subsample=subsample,
            colsample_bytree=colsample_bytree,
            colsample_bylevel=colsample_bylevel,
            colsample_bynode=colsample_bynode, reg_alpha=reg_alpha,
            reg_lambda=reg_lambda, scale_pos_weight=scale_pos_weight,
            base_score=base_score, random_state=random_state, seed=seed,
            missing=missing, **kwargs)
        self.n_boost = n_boost
        self.estimators = []
    def fit(self, X, y, sample_weight = None):
        pos_ind, neg_ind = y > 0, y == 0
        X_pos, y_pos = X[pos_ind], y[pos_ind]
        X_neg, y_neg = X[neg_ind], y[neg_ind]
        for i in range(self.n_boost):
            X_neg_cur, _, y_neg_cur, _ =\
            sklearn.model_selection.train_test_split(X_neg,
                y_neg, train_size = len(y_pos) + 1)
            X_tr = numpy.concatenate((X_pos, X_neg_cur), axis = 0)
            y_tr = numpy.concatenate((y_pos, y_neg_cur), axis = 0)
            est =\
            super().fit(X_tr, y_tr, sample_weight)
            self.estimators.append(est)
        return self
    def predict(self, X):
        return numpy.mean([
            xgboost.XGBClassifier.predict(i, X)
            for i in self.estimators], axis = 0) > 0.5
    def predict_proba(self, X):
        return numpy.mean([
            xgboost.XGBClassifier.predict_proba(i, X)
            for i in self.estimators], axis = 0)
    def predict_log_proba(self, X):
        return numpy.mean([
            xgboost.XGBClassifier.predict_log_proba(i, X)
            for i in self.estimators], axis = 0)
