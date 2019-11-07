import numpy
import pandas
import sqlalchemy
import sklearn
import sklearn.model_selection
import sklearn.preprocessing
import tpot

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
    \caption{{{}}}
    \centering
    \\begin{{tabular}}{{|r|r|}}
        Cross Validation Accuracy
        &Accuracy on a Separate Test Set
        \\hline
        {:.4f} &{:.4f}\\\\
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
with open("log_tpot.csv", "w") as log:
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
        print(f"{cities.iloc[i, 0]}")
        print(f"{y_sel_1.shape[0]}/{y_sel.shape[0]}")
        print(f"{y_hof_1.shape[0]}/{y_hof.shape[0]}")
        log.write("{},{}/{},{}/{}\n".format(cities.iloc[i, 0],
            y_sel_1.shape[0], y_sel.shape[0],
            y_hof_1.shape[0], y_hof.shape[0]))
        clf = tpot.TPOTClassifier(n_jobs = job)
        clf.fit(X_sel, y_sel)
        max_score = clf.score(X_sel, y_sel)
        test_score = clf.score(X_hof, y_hof)
        print(f"{cities.iloc[i, 0]} {max_score} {test_score}")
        with open(f"tex/{cities.iloc[i, 0]}_tpot.tex", "w") as fout:
            fout.write(latex_template.format(cities.iloc[i, 0], max_score,
                test_score))
connection.close()

