import numpy
import pandas
import sqlalchemy
import sklearn
import sklearn.ensemble
import sklearn.preprocessing
import sklearn.metrics
import xgboost

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
\\begin{table}[H]
    \caption\{{}\}
    \centering
    \\begin{tabular}{|r|r|r|r|r|}
\\end{table}
'''
engine = sqlalchemy.create_engine("sqlite:///../../data/data.sqlite")
connection = engine.connect()
cities = pandas.read_sql_query("SELECT Name FROM Cities", connection)
enc = sklearn.preprocessing.OneHotEncoder(sparse = False)

with open("res.csv", "w") as fout:
    fout.write("City,Type,Par,Acc,N\n")
    for i in range(len(cities)):
        data = pandas.read_sql(template.format(cities.iloc[i, 0]),
                               connection)
        one_hot = enc.fit_transform(numpy.array(data["WeatherDescription"])
                                    .reshape(-1, 1))
        X = numpy.concatenate((numpy.array(data.iloc[:, [0, 1, 2, 4, 5]]),
                               one_hot), axis = 1)
        y = numpy.array(data["Fire"])
        ind_1 = y == 1
        '''
        max_score = 0
        par = 0
        est_type = "xgboost.XGBClassifier"
        for j in range(1, 101):
            clf = xgboost.XGBClassifier(n_estimators = j)
            score = numpy.mean(sklearn.model_selection.cross_validate(
                    clf, X_sel, y_sel, scoring = "accuracy", cv = 3,
                    n_jobs = -1)
                    ["test_score"])
            print(cities.iloc[i, 0], est_type, j, score)
            if (score > max_score):
                par = j
                max_score = score
        for j in range(1, 101):
            clf = sklearn.ensemble.RandomForestClassifier(n_estimators = j)
            score = numpy.mean(sklearn.model_selection.cross_validate(
                    clf, X_sel, y_sel, scoring = "accuracy", cv = 3,
                    n_jobs = -1)
                    ["test_score"])
            print(cities.iloc[i, 0], est_type, j, score)
            if (score > max_score):
                par = j
                max_score = score
                est_type = "sklearn.ensemble.RandomForestClassifier"
        for j in range(1, 101):
            clf = sklearn.ensemble.AdaBoostClassifier(n_estimators = j)
            score = numpy.mean(sklearn.model_selection.cross_validate(
                    clf, X_sel, y_sel, scoring = "accuracy", cv = 3,
                    n_jobs = -1)
                    ["test_score"])
            print(cities.iloc[i, 0], est_type, j, score)
            if (score > max_score):
                par = j
                max_score = score
                est_type = "sklearn.ensemble.AdaBoostClassifier"
        test_score = 0
        if (est_type == "xgboost.XGBClassifier"):
            clf = xgboost.XGBClassifier(n_estimators = par)
            clf = clf.fit(X_sel, y_sel)
            y_hat = clf.predict(X_hof)
            test_score = sklearn.metrics.accuracy_score(y_hat, y_hof)
        elif (est_type == "sklearn.ensemble.RandomForestClassifier"):
            clf =\
            sklearn.ensemble.RandomForestClassifier(n_estimators = par)
            clf = clf.fit(X_sel, y_sel)
            y_hat = clf.predict(X_hof)
            test_score = sklearn.metrics.accuracy_score(y_hat, y_hof)
        else:
            clf = sklearn.ensemble.AdaBoostClassifier(n_estimators = par)
            clf = clf.fit(X_sel, y_sel)
            y_hat = clf.predict(X_hof)
            test_score = sklearn.metrics.accuracy_score(y_hat, y_hof)
        print(cities.iloc[i, 0], est_type, par, test_score)
        fout.write(f"{cities.iloc[i, 0]},{est_type},{par},{test_score},")
        fout.write(f"{y_hof.shape[0]}\n")
        '''
connection.close()

