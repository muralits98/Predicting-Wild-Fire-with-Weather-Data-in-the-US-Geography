import sqlite3
conn = sqlite3.connect("../data.sqlite")
c = conn.cursor()
c.execute('''
SELECT name FROM sqlite_master WHERE type = 'table' AND
name NOT LIKE '%\_%' ESCAPE '\\' AND
name <> 'KNN' AND name NOT LIKE 'Spatial%' AND
name NOT LIKE 'Elementary%'
''')
names = c.fetchall()
for j, i in enumerate(names):
    try:
        print(i[0])
        c.execute(f"select * from {i[0]}")
        with open(f"../db_dump/{i[0]}.csv", "w") as fout:
            l = c.fetchone()
            while l:
                fout.write(",".join(map(str, l)))
                fout.write("\n")
                l = c.fetchone()
    except:
        print(f"error in {i[0]}")
conn.close()
