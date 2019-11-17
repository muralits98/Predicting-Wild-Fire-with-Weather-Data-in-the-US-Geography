import util

import numpy
import pandas
import sqlalchemy
import sklearn
import sklearn.preprocessing
import sklearn.metrics
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
    \\caption{{{}}}
    \\centering
    \\begin{{tabular}}{{|r|r|r|r|r|}}
        \\hline
        Classifier &\\multicolumn{{2}}{{|r|}}{{Meta Parameter}}
        &Training Accuracy
        &Test Accuracy\\\\
        \\hline
        &n\\_bootstrap &n\\_estimators &\\multicolumn{{2}}{{|r|}}{{}}\\\
        \\hline
        XGBClassifier &{} &{} &{:.4f} &{:.4f}\\\\
        \\hline
        RandomForestClassifier &{} &{} &{:.4f} &{:.4f}\\\\
        \\hline
    \\end{{tabular}}
\\end{{table}}
'''
engine = sqlalchemy.create_engine("sqlite:///../../../data/data.sqlite")
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
    X_sel = numpy.concatenate((X_sel_0, X_sel_1), axis = 0)
    y_sel = numpy.concatenate((y_sel_0, y_sel_1), axis = 0)
    X_hof = numpy.concatenate((X_hof_0, X_hof_1), axis = 0)
    y_hof = numpy.concatenate((y_hof_0, y_hof_1), axis = 0)
    max_score_rf = 0
    par_rf = (0, 0)
    for j in range(1, 21):
        for k in range(1, 101):
            clf = util.RandomForestClassifierBS(n_bootstrap = j,
                n_estimators = k)
            score = numpy.mean(sklearn.model_selection.cross_validate(
                clf, X_sel, y_sel, scoring = "accuracy", cv = 5,
                n_jobs = job)["test_score"])
            print(cities.iloc[i, 0], "RandomForestClassifierBS",
                j, k, score)
            if (score > max_score_rf):
                par_rf = (j, k)
                max_score = score
    clf = util.RandomForestClassifierBS(n_bootstrap = par_rf[0],
        n_estimators = par_rf[1])
    clf.fit(X_sel, y_sel)
    y_hat = clf.predict(X_hof)
    test_score_rf = sklearn.metrics.accuracy_score(y_hat, y_hof)
    print(
        f"{cities.iloc[i, 0]} RandomForestClassifierBS {j} {k} {test_score_rf}")
    max_score_xgb = 0
    par_xgb = (0, 0)
    for j in range(1, 21):
        for k in range(1, 101):
            clf = util.XGBClassifierBS(n_bootstrap = j,
                n_estimators = k)
            score = numpy.mean(sklearn.model_selection.cross_validate(
                clf, X_sel, y_sel, scoring = "accuracy", cv = 5,
                n_jobs = job)["test_score"])
            print(cities.iloc[i, 0], "XGBClassifierBS",
                j, k, score)
            if (score > max_score_xgb):
                par_xgb = (j, k)
                max_score_xgb = score
    clf = util.XGBClassifierBS(n_bootstrap = par_xgb[0],
        n_estimators = par_xgb[1])
    clf.fit(X_sel, y_sel)
    y_hat = clf.predict(X_hof)
    test_score_xgb = sklearn.metrics.accuracy_score(y_hat, y_hof)
    print(f"{cities.iloc[i, 0]} XGBClassifierBS {j} {k} {test_score_xgb}")
    with open(f"tex/{cities.iloc[i, 0]}.tex", "w") as fout:
        fout.write(latex_template.format(par_xgb[0],
            par_xgb[1], max_score_xgb, test_score_xgb,
            par_rf[0], par_rf[1], max_score_rf, test_score_rf))
        
