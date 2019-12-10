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
    \\begin{{tabular}}{{|r|r|r|r|r|r|r|r|r|}}
        \\hline
        Classifier &\\multicolumn{{2}}{{|r|}}{{Meta Parameter}}
        &\\multicolumn{{4}}{{|r|}}{{Training}}
        &\\multicolumn{{2}}{{|r|}}{{Test}}\\\\
        \\hline
        &n\\_boost &n\\_estimators
        &acc
        &bal\\_acc
        &prec
        &rec
        &acc
        &bal\\_acc
        &prec
        &rec\\\\
        \\hline
        XGBClassifier &{} &{} &{:.4f} &{:.4f} &{:.4f} &{:.4f}
        &{:.4f} &{:.4f} &{:.4f} &{:.4f}\\\\
        \\hline
        RandomForestClassifier &{} &{} &{:.4f} &{:.4f} &{:.4f} &{:.4f}
        &{:.4f} &{:.4f} &{:.4f} &{:.4f}\\\\
        \\hline
    \\end{{tabular}}
\\end{{table}}
'''
engine = sqlalchemy.create_engine("sqlite:///../../../data/data.sqlite")
connection = engine.connect()


import sys
job = -1

if (len(sys.argv) < 2):
    print("Error")
    exit(1)

cities = sys.argv[1:]

colors = ["red", "green", "blue", "yellow", "magenta", "cyan", "orange",
          "purple", "indigo", "gold"]
fig1 = matplotlib.pyplot.figure()
fig2 = matplotlib.pyplot.figure()
fig3 = matplotlib.pyplot.figure()
fig4 = matplotlib.pyplot.figure()
ax1 = fig1.add_subplot(111)
ax2 = fig2.add_subplot(111)
ax3 = fig3.add_subplot(111)
ax4 = fig4.add_subplot(111)

for i in range(len(cities)):
    data = pandas.read_sql(template.format(cities[i]),
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
    train_acc_rf = 0
    train_prec_rf = 0
    train_rec_rf = 0
    par_rf = (0, 0)
    scores = numpy.zeros(100)
    accs = numpy.zeros(100)
    precs = numpy.zeros(100)
    recs = numpy.zeros(100)

    for j in range(1, 11):
        for k in range(1, 101):
            clf = util.RandomForestClassifierBS(n_boost = j,
                n_estimators = k)
            score = numpy.mean(sklearn.model_selection.cross_validate(
                clf, X_sel, y_sel, scoring = "balanced_accuracy", cv = 20,
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
            scores[k - 1] = score
            accs[k - 1] = acc
            precs[k - 1] = prec
            recs[k - 1] = rec
            print(cities[i], "RandomForestClassifierBS",
                j, k, acc, score)
            if (score > max_score_rf):
                par_rf = (j, k)
                max_score_rf = score
                train_acc_rf = acc
                train_prec_rf = prec
                train_rec_rf = rec
        label = f"n_boost = {j}"
        ax1.plot(scores, color = colors[j - 1], label = label)
        ax2.plot(accs, color = colors[j - 1], label = label)
        ax3.plot(precs, color = colors[j - 1], label = label)
        ax4.plot(recs, color = colors[j - 1], label = label)

    ax1.set_xlabel("n_estimator")
    ax1.set_ylabel("balanced accuracy")
    ax1.legend()
    ax2.set_xlabel("n_estimator")
    ax2.set_ylabel("accuracy")
    ax2.legend()
    ax3.set_xlabel("n_estimator")
    ax3.set_ylabel("precision")
    ax3.legend()
    ax4.set_xlabel("n_estimator")
    ax4.set_ylabel("recall")
    ax4.legend()
    fig1.suptitle(f"{cities[i]} RandomForestClassifierBS")
    fig1.savefig(
        f"plot_part/{cities[i]}RandomForestClassifierBS_1.pdf")
    fig2.suptitle(f"{cities[i]} RandomForestClassifierBS")
    fig2.savefig(
        f"plot_part/{cities[i]}RandomForestClassifierBS_2.pdf")
    fig3.suptitle(f"{cities[i]} RandomForestClassifierBS")
    fig3.savefig(
        f"plot_part/{cities[i]}RandomForestClassifierBS_3.pdf")
    fig4.suptitle(f"{cities[i]} RandomForestClassifierBS")
    fig4.savefig(
        f"plot_part/{cities[i]}RandomForestClassifierBS_4.pdf")
    ax1.clear()
    ax2.clear()
    ax3.clear()
    ax4.clear()

    clf = util.RandomForestClassifierBS(n_boost = par_rf[0],
        n_estimators = par_rf[1])
    clf.fit(X_sel, y_sel)
    y_hat = clf.predict(X_hof)
    test_score_rf = sklearn.metrics.balanced_accuracy_score(y_hat, y_hof)
    test_acc_rf = sklearn.metrics.accuracy_score(y_hat, y_hof)
    test_prec_rf = sklearn.metrics.precision_score(y_hat, y_hof)
    test_rec_rf = sklearn.metrics.recall_score(y_hat, y_hof)
    print(
        f"{cities[i]} RandomForestClassifierBS {j} {k} {test_score_rf}")
    max_score_xgb = 0
    train_acc_xgb = 0
    train_prec_xgb = 0
    train_rec_xgb = 0
    par_xgb = (0, 0)

    for j in range(1, 11):
        for k in range(1, 101):
            clf = util.XGBClassifierBS(n_boost = j,
                n_estimators = k)
            score = numpy.mean(sklearn.model_selection.cross_validate(
                clf, X_sel, y_sel, scoring = "balanced_accuracy", cv = 20,
                n_jobs = job)["test_score"])
            acc = numpy.mean(sklearn.model_selection.cross_validate(
                clf, X_sel, y_sel, scoring = "accuracy", cv = 20,
                n_jobs = job)["test_score"])
            prec = numpy.mean(sklearn.model_selection.cross_validate(
                clf, X_sel, y_sel, scoring = "precision", cv = 20,
                n_jobs = job)["test_score"])
            rec = numpy.mean(sklearn.model_selection.cross_validate(
                clf, X_sel, y_sel, scoring = "precision", cv = 20,
                n_jobs = job)["test_score"])
            scores[k - 1] = score
            accs[k - 1] = acc
            precs[k - 1] = prec
            recs[k - 1] = rec
            print(cities[i], "XGBClassifierBS",
                j, k, score)
            if (score > max_score_xgb):
                par_xgb = (j, k)
                max_score_xgb = score
                train_acc_xgb = acc
                train_prec_xgb = prec
                train_rec_xgb = rec
        label = f"n_boost = {j}"
        ax1.plot(scores, color = colors[j - 1], label = label)
        ax2.plot(accs, color = colors[j - 1], label = label)
        ax3.plot(precs, color = colors[j - 1], label = label)
        ax4.plot(recs, color = colors[j - 1], label = label)

    ax1.set_xlabel("n_estimator")
    ax1.set_ylabel("balanced accuracy")
    ax1.legend()
    ax2.set_xlabel("n_estimator")
    ax2.set_ylabel("accuracy")
    ax2.legend()
    ax3.set_xlabel("n_estimator")
    ax3.set_ylabel("precision")
    ax3.legend()
    ax4.set_xlabel("n_estimator")
    ax4.set_ylabel("recall")
    ax4.legend()
    fig1.suptitle(f"{cities[i]} XGBClassifierBS")
    fig1.savefig(
        f"plot_part/{cities[i]}XGBClassifierBS_1.pdf")
    fig2.suptitle(f"{cities[i]} XGBClassifierBS")
    fig2.savefig(
        f"plot_part/{cities[i]}XGBClassifierBS_2.pdf")
    fig3.suptitle(f"{cities[i]} XGBClassifierBS")
    fig3.savefig(
        f"plot_part/{cities[i]}XGBClassifierBS_3.pdf")
    fig4.suptitle(f"{cities[i]} XGBClassifierBS")
    fig4.savefig(
        f"plot_part/{cities[i]}XGBClassifierBS_4.pdf")
    ax1.clear()
    ax2.clear()
    ax3.clear()
    ax4.clear()

    clf = util.XGBClassifierBS(n_boost = par_xgb[0],
        n_estimators = par_xgb[1])
    clf.fit(X_sel, y_sel)
    y_hat = clf.predict(X_hof)
    test_score_xgb = sklearn.metrics.balanced_accuracy_score(y_hat, y_hof)
    test_acc_xgb = sklearn.metrics.accuracy_score(y_hat, y_hof)
    test_prec_xgb = sklearn.metrics.precision_score(y_hat, y_hof)
    test_rec_xgb = sklearn.metrics.recall_score(y_hat, y_hof)
    print(f"{cities[i]} XGBClassifierBS {j} {k} {test_score_xgb}")
    with open(f"tex_part/{cities[i]}.tex", "w") as fout:
        fout.write(latex_template.format(cities[i],
            par_xgb[0], par_xgb[1], train_acc_xgb, max_score_xgb,
            train_prec_xgb, train_rec_xgb, 
            test_acc_xgb, test_score_xgb,
            test_prec_xgb, test_rec_xgb,
            par_rf[0], par_rf[1], train_acc_rf, max_score_rf,
            train_prec_rf, train_rec_rf,
            test_acc_rf, test_score_rf,
            test_prec_rf, test_rec_rf))
        
