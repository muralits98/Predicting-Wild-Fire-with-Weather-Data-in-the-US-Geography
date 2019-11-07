import numpy
import pandas
import sqlalchemy
import sklearn
import sklearn.ensemble
import sklearn.svm
import sklearn.preprocessing
import sklearn.metrics
import xgboost
import matplotlib.pyplot

template = '''
SELECT
    Humidity, Pressure, Temperature, WeatherDescription, WindDirection,
    WindSpeed, Fire
FROM
    {}Fire
WHERE
    Year < 2016 AND
    Humidity IS NOT NULL AND
    Pressure IS NOT NULL AND
    Temperature IS NOT NULL AND
    WeatherDescription IS NOT NULL AND
    WindDirection IS NOT NULL AND
    WindSpeed IS NOT NULL   
'''

latex_template = '''
\\begin{{table}}[H]
    \\begin{{tabular}}{{|r|r|r|r|}}
        \\hline
        Classifier &Meta Parameter &Training Accuracy
        &Test Accuracy\\\\
        \\hline
        &n\\_estimators &\multicolumn{{2}}{{|r|}}{{}}\\\\
        \\hline
        XGBClassifier &{} &{:.4f} &{:.4f}\\\\
        \\hline
        RandomForestClassifier &{} &{:.4f} &{:.4f}\\\\
        \\hline
        &C &\multicolumn{{2}}{{|r|}}{{}}\\\\
        \\hline
        SVC &{:.4f} &{:.4f} &{:.4f}\\\\
        \\hline
    \\end{{tabular}}
\\end{{table}}
'''
engine = sqlalchemy.create_engine("sqlite:///../../data/data.sqlite")
connection = engine.connect()
cities = pandas.read_sql_query("SELECT Name FROM Cities", connection)
enc = sklearn.preprocessing.OneHotEncoder(sparse = False)

import sys
job = -1
if (len(sys.argv) == 2):
    try:
        job = int(sys.argv[1])
    except:
        job = -1
with open("log.csv", "w") as log:
    log.write("City,Train,Test\n")
    for i in range(len(cities)):
        data = pandas.read_sql(template.format(cities.iloc[i, 0]),
                               connection)
        one_hot = enc.fit_transform(numpy.array(data["WeatherDescription"])
                                    .reshape(-1, 1))
        X = numpy.concatenate((numpy.array(data.iloc[:, [0, 1, 2, 4, 5]]),
                               one_hot), axis = 1)
        y = numpy.array(data["Fire"])
        ind_1 = y == 1
        ind_0 = y == 0
        X_1 = X[ind_1]
        y_1 = y[ind_1]
        if (not y_1.shape[0]):
            continue
        X_sel_1, X_hof_1, y_sel_1, y_hof_1 =\
        sklearn.model_selection.train_test_split(X_1, y_1, test_size = 0.2)
        X_0 = X[ind_0]
        y_0 = y[ind_0]
        X_sel_0, X_hof_0, y_sel_0, y_hof_0 =\
        sklearn.model_selection.train_test_split(X_0, y_0,
        test_size = y_hof_1.shape[0])
        X_sel_0_tr, X_sel_0_drop, y_sel_0_tr, y_sel_0_drop =\
        sklearn.model_selection.train_test_split(X_sel_0, y_sel_0,
            train_size = y_sel_1.shape[0])
        X_sel = numpy.concatenate((X_sel_0_tr, X_sel_1), axis = 0)
        y_sel = numpy.concatenate((y_sel_0_tr, y_sel_1), axis = 0)
        X_hof = numpy.concatenate((X_hof_0, X_hof_1), axis = 0)
        y_hof = numpy.concatenate((y_hof_0, y_hof_1), axis = 0)
        print(f"{y_sel_1.shape[0]}/{y_sel.shape[0]}")
        print(f"{y_hof_1.shape[0]}/{y_hof.shape[0]}")
        log.write("{},{}/{},{}/{}\n".format(cities.iloc[i, 0],
            y_sel_1.shape[0], y_sel.shape[0],
            y_hof_1.shape[0], y_hof.shape[0]))
        max_score = 0
        par = 0
        scores = numpy.zeros(100)
        for j in range(1, 101):
            clf = xgboost.XGBClassifier(n_estimators = j)
            score = numpy.mean(sklearn.model_selection.cross_validate(
                    clf, X_sel, y_sel, scoring = "accuracy", cv = 5,
                    n_jobs = job)
                    ["test_score"])
            print(cities.iloc[i, 0], "XGBClassifier", j, score)
            scores[j - 1] = score
            if (score > max_score):
                par = j
                max_score = score
        matplotlib.pyplot.plot(scores)
        matplotlib.pyplot.xlabel("n_estimator")
        matplotlib.pyplot.ylabel("accuracy")
        matplotlib.pyplot.title(f"{cities.iloc[i, 0]} XGBClassifier")
        matplotlib.pyplot.savefig(
            f"plot/{cities.iloc[i, 0]}XGBClassifier.pdf")
        matplotlib.pyplot.clf()
        max_score_xgb = max_score
        par_xgb = par
        clf = xgboost.XGBClassifier(n_estimators = par)
        clf = clf.fit(X_sel, y_sel)
        y_hat = clf.predict(X_hof)
        test_score = sklearn.metrics.accuracy_score(y_hat, y_hof)
        print(f"{cities.iloc[i, 0]} XGBClassifier {test_score}")
        test_score_xgb = test_score
        max_score = 0
        par = 0
        for j in range(1, 101):
            clf = sklearn.ensemble.RandomForestClassifier(n_estimators = j)
            score = numpy.mean(sklearn.model_selection.cross_validate(
                    clf, X_sel, y_sel, scoring = "accuracy", cv = 5,
                    n_jobs = job)
                    ["test_score"])
            print(cities.iloc[i, 0], "RandomForestClassifier", j, score)
            scores[j - 1] = score
            if (score > max_score):
                par = j
                max_score = score
        matplotlib.pyplot.plot(scores)
        matplotlib.pyplot.xlabel("n_estimator")
        matplotlib.pyplot.ylabel("accuracy")
        matplotlib.pyplot.title(
            f"{cities.iloc[i, 0]} RandomForestClassifier")
        matplotlib.pyplot.savefig(
            f"plot/{cities.iloc[i, 0]}RandomForestClassifier.pdf")
        matplotlib.pyplot.clf()
        max_score_rf = max_score
        par_rf = par
        clf =\
        sklearn.ensemble.RandomForestClassifier(n_estimators = par)
        clf = clf.fit(X_sel, y_sel)
        y_hat = clf.predict(X_hof)
        test_score = sklearn.metrics.accuracy_score(y_hat, y_hof)
        print(f"{cities.iloc[i, 0]} RandomForestClassifier {test_score}")
        test_score_rf = test_score
        max_score = 0
        par = 0
        par_range = numpy.linspace(0.5, 1, 100)
        for j in range(100):
            clf = sklearn.svm.SVC(C = par_range[i], gamma = "scale")
            score = numpy.mean(sklearn.model_selection.cross_validate(
                    clf, X_sel, y_sel, scoring = "accuracy", cv = 5,
                    n_jobs = job)
                    ["test_score"])
            print(cities.iloc[i, 0], "SVC", j, score)
            scores[j] = score
            if (score > max_score):
                par = par_range[j]
                max_score = score
        matplotlib.pyplot.plot(scores)
        matplotlib.pyplot.xlabel("C")
        matplotlib.pyplot.ylabel("accuracy")
        matplotlib.pyplot.title(f"{cities.iloc[i, 0]} SVC")
        matplotlib.pyplot.savefig(
            f"plot/{cities.iloc[i, 0]}SVC.pdf")
        matplotlib.pyplot.clf()
        max_score_svc = max_score
        par_svc = par
        clf =\
        sklearn.svm.SVC(C = par, gamma = "scale")
        clf = clf.fit(X_sel, y_sel)
        y_hat = clf.predict(X_hof)
        test_score = sklearn.metrics.accuracy_score(y_hat, y_hof)
        print(f"{cities.iloc[i, 0]} SVC {test_score}")
        test_score_svc = test_score
        with open(f"tex/{cities.iloc[i, 0]}.tex", "w") as fout:
            fout.write(latex_template.format(par_xgb, cities.iloc[i, 0],
                max_score_xgb, test_score_xgb, par_rf,
                max_score_rf, test_score_rf, par_svc,
                max_score_svc, test_score_svc))
connection.close()

