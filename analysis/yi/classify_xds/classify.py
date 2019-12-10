import util
import multiordinal

import numpy
import pandas
import sqlalchemy
import sklearn
import sklearn.metrics
import sklearn.cluster
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
    \\begin{{tabular}}{{|r|r|r|r|r|r|r|r|r|}}
        \\hline
        Classifier &Meta Parameter
        &\\multicolumn{{4}}{{|r|}}{{Training}}
        &\\multicolumn{{2}}{{|r|}}{{Test}}\\\\
        \\hline
        &n\\_ensemble
        &acc
        &bal\\_acc
        &prec
        &rec
        &acc
        &bal\\_acc
        &prec
        &rec\\\\
        \\hline
        EnsembleClassifier &{} &{:.4f} &{:.4f} &{:.4f} &{:.4f}
        &{:.4f} &{:.4f} &{:.4f} &{:.4f}\\\\
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

fig1 = matplotlib.pyplot.figure()
fig2 = matplotlib.pyplot.figure()
fig3 = matplotlib.pyplot.figure()
fig4 = matplotlib.pyplot.figure()
ax1 = fig1.add_subplot(111)
ax2 = fig2.add_subplot(111)
ax3 = fig3.add_subplot(111)
ax4 = fig4.add_subplot(111)

enc = sklearn.preprocessing.OneHotEncoder(sparse = False, categories = "auto")

for i in range(len(cities)):
    data = pandas.read_sql(template.format(cities.iloc[i, 0]),
                            connection)
    wd = numpy.array(data["WeatherDescription"])
    many_hot = numpy.zeros(len(wd) * 54)
    multiordinal.transform(wd, many_hot)
    many_hot = many_hot.reshape(len(wd), 54)
    X = numpy.concatenate((numpy.array(data.iloc[:, [0, 1, 2, 4, 5]]),
                           many_hot), axis = 1)
    clustering = sklearn.cluster.DBSCAN(n_jobs = job).fit(X)
    clustering_one_hot = enc.fit_transform(clustering.labels_
        .reshape(-1, 1))
    X = numpy.concatenate((X, clustering_one_hot), axis = 1)
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
    max_score = 0
    train_acc = 0
    train_prec = 0
    train_rec = 0
    par = 0 
    scores = numpy.zeros(100)
    accs = numpy.zeros(100)
    precs = numpy.zeros(100)
    recs = numpy.zeros(100)
    for j in range(1, 101):
        clf = util.XDSClassifier(n_ensemble = j,
                                 n_components = 10)
        score = numpy.mean(sklearn.model_selection.cross_validate(
            clf, X_sel, y_sel, scoring = "balanced_accuracy", cv = 20,
            error_score = "raise",
            n_jobs = job)["test_score"])
        acc = numpy.mean(sklearn.model_selection.cross_validate(
            clf, X_sel, y_sel, scoring = "accuracy", cv = 20,
            n_jobs = job)["test_score"])
        prec = numpy.mean(sklearn.model_selection.cross_validate(
            clf, X_sel, y_sel, scoring = "precision", cv = 20,
            n_jobs = job)["test_score"])
        rec = numpy.mean(sklearn.model_selection.cross_validate(
            clf, X_sel, y_sel, scoring = "recall", cv = 20,
            n_jobs = job)["test_score"])
        scores[j - 1] = score
        accs[j - 1] = acc
        precs[j - 1] = prec
        recs[j - 1] = rec
        print(cities.iloc[i, 0], "XDSClassifier",
            j, acc, score)
        if (score > max_score):
            par = j
            max_score = score
            train_acc = acc
            train_prec = prec
            train_rec = rec
    ax1.plot(scores)
    ax2.plot(accs)
    ax3.plot(precs)
    ax4.plot(recs)

    ax1.set_xlabel("n_ensemble")
    ax1.set_ylabel("balanced accuracy")
    ax2.set_xlabel("n_ensemble")
    ax2.set_ylabel("accuracy")
    ax3.set_xlabel("n_ensemble")
    ax3.set_ylabel("precision")
    ax4.set_xlabel("n_ensemble")
    ax4.set_ylabel("recall")
    fig1.suptitle(f"{cities.iloc[i, 0]} XDSClassifier")
    fig1.savefig(
        f"plot/{cities.iloc[i, 0]}XDSClassifier_1.pdf")
    fig2.suptitle(f"{cities.iloc[i, 0]} XDSClassifier")
    fig2.savefig(
        f"plot/{cities.iloc[i, 0]}XDSClassifier_2.pdf")
    fig3.suptitle(f"{cities.iloc[i, 0]} XDSClassifier")
    fig3.savefig(
        f"plot/{cities.iloc[i, 0]}XDSClassifier_3.pdf")
    fig4.suptitle(f"{cities.iloc[i, 0]} XDSClassifier")
    fig4.savefig(
        f"plot/{cities.iloc[i, 0]}XDSClassifier_4.pdf")
    ax1.clear()
    ax2.clear()
    ax3.clear()
    ax4.clear()
    clf = util.XDSClassifier(n_ensemble = par,
                             n_components = 10)
    clf.fit(X_sel, y_sel)
    y_hat = clf.predict(X_hof)
    test_score = sklearn.metrics.balanced_accuracy_score(y_hat, y_hof)
    test_acc = sklearn.metrics.accuracy_score(y_hat, y_hof)
    test_prec = sklearn.metrics.precision_score(y_hat, y_hof)
    test_rec = sklearn.metrics.recall_score(y_hat, y_hof)
    print(f"{cities.iloc[i, 0]} XDSClassifier {par} {test_score}")
    with open(f"tex/{cities.iloc[i, 0]}.tex", "w") as fout:
        fout.write(latex_template.format(cities.iloc[i, 0],
            par, train_acc, max_score,
            train_prec, train_rec, 
            test_acc, test_score,
            test_prec, test_rec))
        
