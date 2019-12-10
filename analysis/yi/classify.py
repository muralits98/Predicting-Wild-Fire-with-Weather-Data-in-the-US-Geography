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
    \\caption{{{}}}
    \\centering
    \\begin{{tabular}}{{|r|r|r|r|r|r|r|r|r|}}
        \\hline
        Classifier &Meta Parameter
        &\\multicolumn{{4}}{{|r|}}{{Training}}
        &\\multicolumn{{2}}{{|r|}}{{Test}}\\\\
        \\hline
        &n\\_estimators
        &acc
        &bal\\_acc
        &prec
        &rec
        &acc
        &bal\\_acc
        &prec
        &rec\\\\
        \\hline
        XGBClassifier &{} &{:.4f} &{:.4f} &{:.4f} &{:.4f}
        &{:.4f} &{:.4f} &{:.4f} &{:.4f}\\\\
        \\hline
        RandomForestClassifier &{} &{:.4f} &{:.4f} &{:.4f} &{:.4f}
        &{:.4f} &{:.4f} &{:.4f} &{:.4f}\\\\
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

fig1 = matplotlib.pyplot.figure()
fig2 = matplotlib.pyplot.figure()
fig3 = matplotlib.pyplot.figure()
fig4 = matplotlib.pyplot.figure()
ax1 = fig1.add_subplot(111)
ax2 = fig2.add_subplot(111)
ax3 = fig3.add_subplot(111)
ax4 = fig4.add_subplot(111)

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

        max_score_xgb = 0
        train_acc_xgb = 0
        train_prec_xgb = 0
        train_rec_xgb = 0
        par_xgb = 0
        scores = numpy.zeros(100)
        accs = numpy.zeros(100)
        precs = numpy.zeros(100)
        recs = numpy.zeros(100)

        for j in range(1, 101):
            clf = xgboost.XGBClassifier(n_estimators = j)
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
            scores[j - 1] = score
            accs[j - 1] = acc
            precs[j - 1] = prec
            recs[j - 1] = rec
            print(cities.iloc[i, 0], "XGBClassifier",
                j, acc, score)
            if (score > max_score_xgb):
                par_xgb = j
                max_score_xgb = score
        label = f"n_boost = {j}"
        ax1.plot(scores)
        ax2.plot(accs)
        ax3.plot(precs)
        ax4.plot(recs)

        ax1.plot(scores)
        ax2.plot(accs)
        ax3.plot(precs)
        ax4.plot(recs)

        ax1.set_xlabel("n_estimator")
        ax1.set_ylabel("balanced accuracy")
        ax2.set_xlabel("n_estimator")
        ax2.set_ylabel("accuracy")
        ax3.set_xlabel("n_estimator")
        ax3.set_ylabel("precision")
        ax4.set_xlabel("n_estimator")
        ax4.set_ylabel("recall")
        fig1.suptitle(f"{cities.iloc[i, 0]} XGBClassifier")
        fig1.savefig(
            f"plot/{cities.iloc[i, 0]}XGBClassifier_1.pdf")
        fig2.suptitle(f"{cities.iloc[i, 0]} XGBClassifier")
        fig2.savefig(
            f"plot/{cities.iloc[i, 0]}XGBClassifier_2.pdf")
        fig3.suptitle(f"{cities.iloc[i, 0]} XGBClassifier")
        fig3.savefig(
            f"plot/{cities.iloc[i, 0]}XGBClassifier_3.pdf")
        fig4.suptitle(f"{cities.iloc[i, 0]} XGBClassifier")
        fig4.savefig(
            f"plot/{cities.iloc[i, 0]}XGBClassifier_4.pdf")
        ax1.clear()
        ax2.clear()
        ax3.clear()
        ax4.clear()

        clf = xgboost.XGBClassifier(n_estimators = par_xgb)
        clf = clf.fit(X_sel, y_sel)
        y_hat = clf.predict(X_hof)
        test_score_xgb = sklearn.metrics.balanced_accuracy_score(y_hat, y_hof)
        test_acc_xgb = sklearn.metrics.accuracy_score(y_hat, y_hof)
        test_prec_xgb = sklearn.metrics.precision_score(y_hat, y_hof)
        test_rec_xgb = sklearn.metrics.recall_score(y_hat, y_hof)

        max_score_rf = 0
        train_acc_rf = 0
        train_prec_rf = 0
        train_rec_rf = 0
        par_rf = 0
        for j in range(1, 101):
            clf = sklearn.ensemble.RandomForestClassifier(n_estimators = j)
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
            scores[j - 1] = score
            accs[j - 1] = acc
            precs[j - 1] = prec
            recs[j - 1] = rec
            print(cities.iloc[i, 0], "RandomForestClassifier",
                j, acc, score)
            if (score > max_score_rf):
                par_rf = j
                max_score_rf = score
        ax1.plot(scores)
        ax2.plot(accs)
        ax3.plot(precs)
        ax4.plot(recs)


        ax1.set_xlabel("n_estimator")
        ax1.set_ylabel("balanced accuracy")
        ax2.set_xlabel("n_estimator")
        ax2.set_ylabel("accuracy")
        ax3.set_xlabel("n_estimator")
        ax3.set_ylabel("precision")
        ax4.set_xlabel("n_estimator")
        ax4.set_ylabel("recall")
        fig1.suptitle(f"{cities.iloc[i, 0]} RandomForestClassifier")
        fig1.savefig(
            f"plot/{cities.iloc[i, 0]}RandomForestClassifier_1.pdf")
        fig2.suptitle(f"{cities.iloc[i, 0]} RandomForestClassifier")
        fig2.savefig(
            f"plot/{cities.iloc[i, 0]}RandomForestClassifier_2.pdf")
        fig3.suptitle(f"{cities.iloc[i, 0]} RandomForestClassifier")
        fig3.savefig(
            f"plot/{cities.iloc[i, 0]}RandomForestClassifier_3.pdf")
        fig4.suptitle(f"{cities.iloc[i, 0]} RandomForestClassifier")
        fig4.savefig(
            f"plot/{cities.iloc[i, 0]}RandomForestClassifier_4.pdf")
        ax1.clear()
        ax2.clear()
        ax3.clear()
        ax4.clear()

        clf =\
        sklearn.ensemble.RandomForestClassifier(n_estimators = par_rf)
        clf = clf.fit(X_sel, y_sel)
        y_hat = clf.predict(X_hof)
        test_score_rf = sklearn.metrics.balanced_accuracy_score(y_hat, y_hof)
        test_acc_rf = sklearn.metrics.accuracy_score(y_hat, y_hof)
        test_prec_rf = sklearn.metrics.precision_score(y_hat, y_hof)
        test_rec_rf = sklearn.metrics.recall_score(y_hat, y_hof)
        # max_score = 0
        # par = 0
        # par_range = numpy.linspace(0.5, 1, 100)
        # for j in range(100):
        #     clf = sklearn.svm.SVC(C = par_range[i], gamma = "scale")
        #     score = numpy.mean(sklearn.model_selection.cross_validate(
        #             clf, X_sel, y_sel, scoring = "accuracy", cv = 5,
        #             n_jobs = job)
        #             ["test_score"])
        #     print(cities.iloc[i, 0], "SVC", j, score)
        #     scores[j] = score
        #     if (score > max_score):
        #         par = par_range[j]
        #         max_score = score
        # matplotlib.pyplot.plot(scores)
        # matplotlib.pyplot.xlabel("C")
        # matplotlib.pyplot.ylabel("accuracy")
        # matplotlib.pyplot.title(f"{cities.iloc[i, 0]} SVC")
        # matplotlib.pyplot.savefig(
        #     f"plot/{cities.iloc[i, 0]}SVC.pdf")
        # matplotlib.pyplot.clf()
        # max_score_svc = max_score
        # par_svc = par
        # clf =\
        # sklearn.svm.SVC(C = par, gamma = "scale")
        # clf = clf.fit(X_sel, y_sel)
        # y_hat = clf.predict(X_hof)
        # test_score = sklearn.metrics.accuracy_score(y_hat, y_hof)
        # print(f"{cities.iloc[i, 0]} SVC {test_score}")
        # test_score_svc = test_score
        with open(f"tex/{cities.iloc[i, 0]}.tex", "w") as fout:
            fout.write(latex_template.format(cities.iloc[i, 0],
                par_xgb, train_acc_xgb, max_score_xgb,
                train_prec_xgb, train_rec_xgb, 
                test_acc_xgb, test_score_xgb,
                test_prec_xgb, test_rec_xgb,
                par_rf, train_acc_rf, max_score_rf,
                train_prec_rf, train_rec_rf,
                test_acc_rf, test_score_rf,
                test_prec_rf, test_rec_rf))
connection.close()

