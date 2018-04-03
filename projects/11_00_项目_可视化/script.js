function draw(data) {

    var margin = 75,
        width = 900 - margin,
        height = 400 - margin;

    var svg = d3.select("body")
                .append("svg")
                .attr("width", width + margin)
                .attr("height", height + margin)
                .append("g")
                .attr("class", "chart");

    var mychart = new dimple.chart(svg, data);
    mychart.addMeasureAxis("x", "avg")
           .title = "Average"
           .overrideMin = 0
           .overrideMax = 0.3;
    mychart.addMeasureAxis("y", "HR")
           .title = "Home Runs"
           .overrideMin = 0
           .overrideMax = 600;
    mychart.addSeries(
        ["name", "handedness", "HR", "avg", "height", "weight"],
        dimple.plot.bubble
    );
    mychart.draw();

    var buttonNames = ["All", "Left", "Right", "Both"];

    d3.select("#Bttn")
        .selectAll("input")
        .data(buttonNames)
        .enter()
        .append("input")
        .attr("type", "button")
        .attr("class", "button")
        .attr("id", function(d) {
            return d;
        })
        .attr("value", function(d) {
            return d;
        });

    d3.select("#All").on("click", function(d) {
        data_select = d;
        mychart.data = data;
        mychart.draw(1000);
    });

    function filterbttn(cls_id, val) {
        d3.select(cls_id).on("click", function(d) {
            data_select = d;
            mychart.data = dimple.filterData(data, "handedness", val);
            mychart.draw(1000);
        });
    };

    filterbttn("#Left", "L");

    filterbttn("#Right", "R");

    filterbttn("#Both", "B");

};