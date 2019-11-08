let cities = [
    {
        name: "Albuquerque",
        x: 200,
        y: 600
    }, {
        name: "Houston",
        x: 330,
        y: 690
    }, {
        name: "Montreal",
        x: 540,
        y: 460
    }, {
        name: "SanAntonio",
        x: 290,
        y: 680
    }, {
        name: "Atlanta",
        x: 450,
        y: 630
    }, {
        name: "Indianapolis",
        x: 460,
        y: 550
    }, {
        name: "Nashville",
        x: 420,
        y: 600
    }, {
        name: "SanDiego",
        x: 100,
        y: 600
    }, {
        name: "Charlotte",
        x: 500,
        y: 610
    }, {
        name: "Jacksonville",
        x: 490,
        y: 670
    }, {
        name: "NewYork",
        x: 550,
        y: 520
    }, {
        name: "SanFrancisco",
        x: 60,
        y: 510
    }, {
        name: "Chicago",
        x: 410,
        y: 520
    }, {
        name: "KansasCity",
        x: 350,
        y: 560
    }, {
        name: "Philadelphia",
        x: 540,
        y: 540
    }, {
        name: "Seattle",
        x: 100,
        y: 385
    }, {
        name: "Dallas",
        x: 320,
        y: 640
    }, {
        name: "LasVegas",
        x: 120,
        y: 560
    }, {
        name: "Phoenix",
        x: 150,
        y: 600
    }, {
        name: "Toronto",
        x: 490,
        y: 490
    }, {
        name: "Denver",
        x: 230,
        y: 540
    }, {
        name: "LosAngeles",
        x: 90,
        y: 570
    }, {
        name: "Pittsburgh",
        x: 480,
        y: 540
    }, {
        name: "Vancouver",
        x: 110,
        y: 360
    }, {
        name: "Detroit",
        x: 460,
        y: 520
    }, {
        name: "Miami",
        x: 500,
        y: 730
    }, {
        name: "Portland",
        x: 100,
        y: 410
    }, {
        name: "Minneapolis",
        x: 360,
        y: 480
    }, {
        name: "SaintLouis",
        x: 380,
        y: 570
    }
];

let ind = 0;
let time = null;
let inp = null;
let max_ind = 28440;
let timer = null;

window.onload = () => {
    con = d3.select("#console").text("Loading, please wait");
    let promises = [];
    for (i of cities) {
        promises.push(d3.json(`data/${i.name}Fire.json`, (err, data) => {
            console.log(err);
        }));
    }
    Promise.all(
        promises
    ).then((data) => {
        for (let i = 0; i < data.length; i++) {
            cities[i].data = data[i];
            cities[i].img = d3.select("#main")
                .append("image")
                .attr("href", "res/fire.png")
                .attr("x", cities[i].x)
                .attr("y", cities[i].y)
                .attr("width", 25)
                .attr("height", 25)
                .style("visibility", data[i][ind].Fire ? "visible" : "hidden");
            let tab = d3.select("#info")
                .append("table")
                .style("grid-row-start", i / 2 + 1)
                .style("grid-row-end", i / 2 + 1)
                .style("grid-column-start", i % 2 + 1)
                .style("grid-column-end", i % 2 + 1);
            tab.append("tr")
                .append("th")
                .text(cities[i].name)
                .append("th");
            let r = tab.append("tr")
                .append("td")
                .text("Temperature")
            cities[i].table = {}
            cities[i].table.Temperature = r.append("td")
                .text(data[i][ind].Temperature);
            r = tab.append("tr")
                .append("td")
                .text("Humidity")
            cities[i].table.Humidity = r.append("td")
                .text(data[i][ind].Humidity);
            r = tab.append("tr")
                .append("td")
                .text("Pressure")
            cities[i].table.Pressure = r.append("td")
                .text(data[i][ind].Pressure);
            r = tab.append("tr")
                .append("td")
                .text("Weather Description")
            cities[i].table.WindDescription = r.append("td")
                .text(data[i][ind].WeatherDescription);
            r = tab.append("tr")
                .append("td")
                .text("Wind Direction")
            cities[i].table.WindDirection = r.append("td")
                .text(data[i][ind].WindDirection);
            r = tab.append("tr")
                .append("td")
                .text("Wind Speed")
            cities[i].table.WindSpeed = r.append("td")
                .text(data[i][ind].WindSpeed);
        }
        inp = d3.select("#inp").attr("value", ind);
        time = d3.select("#time")
            .append("table")
            .append("tr")
            .append("td")
            .text("Time")
            .append("td")
            .text(`${cities[0].data[ind].Year} ${cities[0].data[ind].DOY} ${cities[0].data[ind].HM}`);
        con.text("Ready");
        d3.selectAll("button").attr("disabled", null);
    });
};

function main() {
    if (!timer) {
        con.text("Running");
        timer = setInterval(() => {
            next();
            if (ind >= max_ind) {
                clearInterval(timer);
            }
        }, 200);
    }
}

function next() {
    if (ind >= max_ind) return;
    ind++;
    inp.attr("value", ind);
    time.text(`${cities[0].data[ind].Year} ${cities[0].data[ind].DOY} ${cities[0].data[ind].HM}`)
    for (let i of cities) {
        i.img.style("visibility", i.data[ind].Fire ? "visible" : "hidden");
        for (let j in i.table) {
            i.table[j].text(i.data[ind][j])
        }
    }
}

function previous() {
    if (ind < 0) return;
    ind--;
    inp.attr("value", ind);
    time.text(`${cities[0].data[ind].Year} ${cities[0].data[ind].DOY} ${cities[0].data[ind].HM}`)
    for (let i of cities) {
        i.img.style("visibility", i.data[ind].Fire ? "visible" : "hidden");
        for (let j in i.table) {
            i.table[j].text(i.data[ind][j])
        }
    }
}

function reset() {
    ind = 0;
    inp.attr("value", ind);
    time.text(`${cities[0].data[ind].Year} ${cities[0].data[ind].DOY} ${cities[0].data[ind].HM}`)
    for (let i of cities) {
        i.img.style("visibility", i.data[ind].Fire ? "visible" : "hidden");
        for (let j in i.table) {
            i.table[j].text(i.data[ind][j])
        }
    }
}

function stop() {
    if (timer) {
        con.text("Ready");
        clearInterval(timer);
        timer = null;
    }
}

function move() {
    let val = parseInt(document.getElementById("inp").value);
    if (val >= 0 && val <= max_ind) {
        ind = val;
        time.text(`${cities[0].data[ind].Year} ${cities[0].data[ind].DOY} ${cities[0].data[ind].HM}`)
        for (let i of cities) {
            i.img.style("visibility", i.data[ind].Fire ? "visible" : "hidden");
            for (let j in i.table) {
                i.table[j].text(i.data[ind][j])
            }
        }
    } else {
        alert(`Invalid! Must be between 0 and ${max_ind}`);
    }
}
