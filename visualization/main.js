let cities = [
    {
        name: "Albuquerque",
        data: Albuquerque,
        x: 200,
        y: 600
    }, {
        name: "Houston",
        data: Houston,
        x: 330,
        y: 690
    }, {
        name: "Montreal",
        data: Montreal,
        x: 540,
        y: 460
    }, {
        name: "SanAntonio",
        data: SanAntonio,
        x: 290,
        y: 680
    }, {
        name: "Atlanta",
        data: Atlanta,
        x: 450,
        y: 630
    }, {
        name: "Indianapolis",
        data: Indianapolis,
        x: 460,
        y: 550
    }, {
        name: "Nashville",
        data: Nashville,
        x: 420,
        y: 600
    }, {
        name: "SanDiego",
        data: SanDiego,
        x: 100,
        y: 600
    }, {
        name: "Charlotte",
        data: Charlotte,
        x: 500,
        y: 610
    }, {
        name: "Jacksonville",
        data: Jacksonville,
        x: 490,
        y: 670
    }, {
        name: "NewYork",
        data: NewYork,
        x: 550,
        y: 520
    }, {
        name: "SanFrancisco",
        data: SanFrancisco,
        x: 60,
        y: 510
    }, {
        name: "Chicago",
        data: Chicago,
        x: 410,
        y: 520
    }, {
        name: "KansasCity",
        data: KansasCity,
        x: 350,
        y: 560
    }, {
        name: "Philadelphia",
        data: Philadelphia,
        x: 540,
        y: 540
    }, {
        name: "Seattle",
        data: Seattle,
        x: 100,
        y: 385
    }, {
        name: "Dallas",
        data: Dallas,
        x: 320,
        y: 640
    }, {
        name: "LasVegas",
        data: LasVegas,
        x: 120,
        y: 560
    }, {
        name: "Phoenix",
        data: Phoenix,
        x: 150,
        y: 600
    }, {
        name: "Toronto",
        data: Toronto,
        x: 490,
        y: 490
    }, {
        name: "Denver",
        data: Denver,
        x: 230,
        y: 540
    }, {
        name: "LosAngeles",
        data: LosAngeles,
        x: 90,
        y: 570
    }, {
        name: "Pittsburgh",
        data: Pittsburgh,
        x: 480,
        y: 540
    }, {
        name: "Vancouver",
        data: Vancouver,
        x: 110,
        y: 360
    }, {
        name: "Detroit",
        data: Detroit,
        x: 460,
        y: 520
    }, {
        name: "Miami",
        data: Miami,
        x: 500,
        y: 730
    }, {
        name: "Portland",
        data: Portland,
        x: 100,
        y: 410
    }, {
        name: "Minneapolis",
        data: Minneapolis,
        x: 360,
        y: 480
    }, {
        name: "SaintLouis",
        data: SaintLouis,
        x: 380,
        y: 570
    }
];

let ind = 0;
let time = null;
let max_ind = 28440;
let timer = null;

window.onload = () => {
    time = d3.select("#time")
        .append("table")
        .append("tr")
        .append("td")
        .text("Time")
        .append("td")
        .text(`${cities[0].data[ind].Year} ${cities[0].data[ind].DOY} ${cities[0].data[ind].HM}`);
    let j = 0
    for (let i of cities) {
        i.img = d3.select("#main")
            .append("image")
            .attr("href", "res/fire.png")
            .attr("x", i.x)
            .attr("y", i.y)
            .attr("width", 20)
            .attr("height", 20)
            .style("visibility", i.data[ind].Fire ? "visible" : "hidden");
        let tab = d3.select("#info")
            .append("table")
            .style("grid-row-start", j / 2 + 1)
            .style("grid-row-end", j / 2 + 1)
            .style("grid-column-start", j % 2 + 1)
            .style("grid-column-end", j % 2 + 1);
        tab.append("tr")
            .append("th")
            .text(i.name)
            .append("th");
        let r = tab.append("tr")
            .append("td")
            .text("Temperature")
        i.table = {}
        i.table.Temperature = r.append("td")
            .text(i.data[ind].Temperature);
        r = tab.append("tr")
            .append("td")
            .text("Humidity")
        i.table.Humidity = r.append("td")
            .text(i.data[ind].Humidity);
        r = tab.append("tr")
            .append("td")
            .text("Pressure")
        i.table.Pressure = r.append("td")
            .text(i.data[ind].Pressure);
        r = tab.append("tr")
            .append("td")
            .text("Weather Description")
        i.table.WindDescription = r.append("td")
            .text(i.data[ind].WeatherDescription);
        r = tab.append("tr")
            .append("td")
            .text("Wind Direction")
        i.table.WindDirection = r.append("td")
            .text(i.data[ind].WindDirection);
        r = tab.append("tr")
            .append("td")
            .text("Wind Speed")
        i.table.WindSpeed = r.append("td")
            .text(i.data[ind].WindSpeed);
        j++;
    }
};

function main() {
    timer = setInterval(() => {
        next();
        if (ind >= max_ind) {
            clearInterval(timer);
        }
    }, 200)
}

function next() {
    if (ind >= max_ind) return;
    ind++;
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
        clearInterval(timer);
    }
}