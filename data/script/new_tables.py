import csv
import sqlite3
import time

script = '''
DROP TABLE IF EXISTS {0};
CREATE TABLE IF NOT EXISTS {0} (
    Year TEXT NOT NULL,
    DOY INTEGER NOT NULL,
    HM TEXT NOT NULL,
    Humidity REAL,
    Pressure REAL,
    Temperature REAL,
    WeatherDescription TEXT,
    WindDirection REAL,
    WindSpeed REAL,
    PRIMARY KEY (Year, DOY, HM)
);
DROP TABLE IF EXISTS {0}Fire;
CREATE TABLE IF NOT EXISTS {0}Fire (
    Year TEXT NOT NULL,
    DOY INTEGER NOT NULL,
    HM TEXT NOT NULL,
    Humidity REAL,
    Pressure REAL,
    Temperature REAL,
    WeatherDescription TEXT,
    WindDirection REAL,
    WindSpeed REAL,
    Fire Integer,
    PRIMARY KEY (Year, DOY, HM)
);
'''

template = '''
INSERT INTO {} VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
'''

template_fire = '''
INSERT INTO {}Fire VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
'''

fns = ["humidity.csv", "pressure.csv", "temperature.csv",
       "weather_description.csv", "wind_direction.csv", "wind_speed.csv"]

script_city = '''
DROP TABLE IF EXISTS Cities;
CREATE TABLE IF NOT EXISTS Cities (
    Name TEXT PRIMARY KEY,
    Latitude REAL,
    Longitude REAL
);
'''

template_city = '''
INSERT INTO Cities VALUES (?, ?, ?)
'''

def main(args):
    flag = True
    limit = 0 
    if (len(args) == 2):
        flag = False
        try:
            limit = int(args[1])
        except:
            return 1
    conn = sqlite3.connect("../data_raw.sqlite")
    c = conn.cursor()
    cities = []
    cordinates = []
    with open("../weather/city_attributes.csv") as fin:
        flag = False
        for i in csv.reader(fin):
            if (flag and len(args) < 2):
                print(i[0])
                c.executescript(script.format(i[0].replace(' ', '')))
                cities.append(i[0].replace(' ', ''))
                cordinates.append((float(i[2]), float(i[3])))
            else:
                flag = True
    if (len(args) < 2):
        c.executescript(script_city)    
        for i in zip(cities, cordinates):
            c.execute(template_city, (i[0], i[1][0], i[1][1]))        

    file_handles = [open(f"../weather/{i}") for i in fns]
    readers = [csv.reader(i) for i in file_handles]

    for i in readers:
        i.__next__()
    i = 0
    while 1:
        try:
            data = [i.__next__() for i in readers]
            if (i < limit):
                print(i)
                i += 1
                continue
            dt_str = data[0][0]
            dt = time.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
            year = time.strftime("%Y", dt)
            doy = int(time.strftime("%j", dt))
            hm = time.strftime("%H%M", dt)
            for i, j in enumerate(cities):
                print(dt_str, j)
                val = (year, doy, hm,
                       float(data[0][i + 1]) if data[0][i + 1] else None,
                       float(data[1][i + 1]) if data[1][i + 1] else None,
                       float(data[2][i + 1]) if data[2][i + 1] else None,
                       data[3][i + 1]        if data[3][i + 1] else None,
                       float(data[4][i + 1]) if data[4][i + 1] else None,
                       float(data[5][i + 1]) if data[5][i + 1] else None)
                c.execute(template.format(j), val)
                c.execute(template_fire.format(j),
                          val + (0,))
                conn.commit()
        except Exception as e:
            print(e)
            break
    for i in file_handles:
        i.close()
    conn.close()
    return 0

if __name__ == "__main__":
    import sys
    ret = main(sys.argv)
    exit(ret)

