import csv
import matplotlib.pyplot

with open("../data/all.csv", "r") as fin:
    r = csv.reader(fin)
    l = list(r)
    bins = [i[0] for i in l[1:]]
    weight = [int(i[1]) for i in l[1:]]
    matplotlib.pyplot.tight_layout()
    matplotlib.pyplot.bar(bins, weight)
    matplotlib.pyplot.xticks(rotation = 90)
    matplotlib.pyplot.title("Number of Fires by Hour of Day")
    matplotlib.pyplot.xlabel("HM")
    matplotlib.pyplot.ylabel("Number of Fires")
    matplotlib.pyplot.savefig("../plot/byhour_all.pdf",
                              bbox_inches = "tight")

matplotlib.pyplot.clf()

with open("../data/sf.csv", "r") as fin:
    r = csv.reader(fin)
    l = list(r)
    bins = [i[0] for i in l[1:]]
    weight = [int(i[1]) for i in l[1:]]
    matplotlib.pyplot.tight_layout()
    matplotlib.pyplot.bar(bins, weight)
    matplotlib.pyplot.xticks(rotation = 90)
    matplotlib.pyplot.title("Number of Fires by Hour of Day")
    matplotlib.pyplot.xlabel("HM")
    matplotlib.pyplot.ylabel("Number of Fires")
    matplotlib.pyplot.savefig("../plot/byhour_sf.pdf",
                              bbox_inches = "tight")

matplotlib.pyplot.clf()

with open("../data/ny.csv", "r") as fin:
    r = csv.reader(fin)
    l = list(r)
    bins = [f"0{i}00" for i in range(10)] + [i[0] for i in l[1:]]
    weight = [0 for i in range(10)] + [int(i[1]) for i in l[1:]]
    matplotlib.pyplot.tight_layout()
    matplotlib.pyplot.bar(bins, weight)
    matplotlib.pyplot.xticks(rotation = 90)
    matplotlib.pyplot.title("Number of Fires by Hour of Day")
    matplotlib.pyplot.xlabel("HM")
    matplotlib.pyplot.ylabel("Number of Fires")
    matplotlib.pyplot.savefig("../plot/byhour_ny.pdf",
                              bbox_inches = "tight")

matplotlib.pyplot.clf()
