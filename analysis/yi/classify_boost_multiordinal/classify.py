import util
import multiordinal

import numpy
import pandas
import sqlalchemy
import sklearn
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
        &Training
        &Training\\\\
        \\hline
        &n\\_bootstrap &n\\_estimators
        &bal\\_acc
        &bal\\_acc\\\\
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

import sys
job = -1
if (len(sys.argv) == 2):
    try:
        job = int(sys.argv[1])
    except:
        job = -1

colors = ["red", "green", "blue", "yellow", "magenta", "cyan", "orange",
          "purple", "indigo", "gold"]
for i in range(len(cities)):
    data = pandas.read_sql(template.format(cities.iloc[i, 0]),
                            connection)
    wd = numpy.array(data["WeatherDescription"])
    many_hot = numpy.zeros(len(wd) * 54)
    multiordinal.transform(wd, many_hot)
    many_hot = many_hot.reshape(len(wd), 54)
    X = numpy.concatenate((numpy.array(data.iloc[:, [0, 1, 2, 4, 5]]),
                           many_hot), axis = 1)
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
    scores = numpy.zeros(100)
    lines = []
    labels = []
    for j in range(1, 11):
        for k in range(1, 101):
            clf = util.RandomForestClassifierBS(n_boost = j,
                n_estimators = k)
            score = numpy.mean(sklearn.model_selection.cross_validate(
                clf, X_sel, y_sel, scoring = "balanced_accuracy", cv = 20,
                n_jobs = job)["test_score"])
            scores[k - 1] = score
            print(cities.iloc[i, 0], "RandomForestClassifierBS",
                j, k, score)
            if (score > max_score_rf):
                par_rf = (j, k)
                max_score_rf = score
        line = matplotlib.pyplot.plot(scores, color = colors[j - 1])
        lines.append(line)
        labels.append(f"n_boost = {j}")
    matplotlib.pyplot.xlabel("n_estimator")
    matplotlib.pyplot.ylabel("balanced accuracy")
    matplotlib.pyplot.legend(lines, labels)
    matplotlib.pyplot.title(f"{cities.iloc[i, 0]} XGBClassifierBS")
    matplotlib.pyplot.savefig(
        f"plot/{cities.iloc[i, 0]}XGBClassifierBS.pdf")
    matplotlib.pyplot.clf()

    clf = util.RandomForestClassifierBS(n_boost = par_rf[0],
        n_estimators = par_rf[1])
    clf.fit(X_sel, y_sel)
    y_hat = clf.predict(X_hof)
    test_score_rf = sklearn.metrics.balanced_accuracy_score(y_hat, y_hof)
    print(
        f"{cities.iloc[i, 0]} RandomForestClassifierBS {j} {k} {test_score_rf}")
    max_score_xgb = 0
    par_xgb = (0, 0)
    lines = []
    labels = []
    for j in range(1, 11):
        for k in range(1, 101):
            clf = util.XGBClassifierBS(n_boost = j,
                n_estimators = k)
            score = numpy.mean(sklearn.model_selection.cross_validate(
                clf, X_sel, y_sel, scoring = "balanced_accuracy", cv = 20,
                n_jobs = job)["test_score"])
            scores[k - 1] = score
            print(cities.iloc[i, 0], "XGBClassifierBS",
                j, k, score)
            if (score > max_score_xgb):
                par_xgb = (j, k)
                max_score_xgb = score
        line = matplotlib.pyplot.plot(scores, color = colors[j - 1])
        lines.append(line)
        labels.append(f"n_boost = {j}")
    matplotlib.pyplot.xlabel("n_estimator")
    matplotlib.pyplot.ylabel("balanced accuracy")
    matplotlib.pyplot.legend(lines, labels)
    matplotlib.pyplot.title(f"{cities.iloc[i, 0]} RandomForestClassifierBS")
    matplotlib.pyplot.savefig(
        f"plot/{cities.iloc[i, 0]}XGBClassifierBS.pdf")
    matplotlib.pyplot.clf()
    clf = util.XGBClassifierBS(n_boost = par_xgb[0],
        n_estimators = par_xgb[1])
    clf.fit(X_sel, y_sel)
    y_hat = clf.predict(X_hof)
    test_score_xgb = sklearn.metrics.balanced_accuracy_score(y_hat, y_hof)
    print(f"{cities.iloc[i, 0]} XGBClassifierBS {j} {k} {test_score_xgb}")
    with open(f"tex/{cities.iloc[i, 0]}.tex", "w") as fout:
        fout.write(latex_template.format(cities.iloc[i, 0],
            par_xgb[0], par_xgb[1], max_score_xgb, test_score_xgb,
            par_rf[0], par_rf[1], max_score_rf, test_score_rf))
        
