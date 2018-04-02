function draw(data) {

    var margin = 75,
        width = 1000 - margin,
        height = 600 - margin;

    var svg = d3.select("body")
                .append("svg")
                .attr("width", width + margin)
                .attr("height", height + margin)
                .append("g")
                .attr("class", "chart");

    var myChart = new dimple.chart(svg, data);
    
	var x = myChart.addMeasureAxis("x", "height");
	x.title = '身高';
	debugger;
	var y = myChart.addMeasureAxis("y", "weight");
	y.title = '体重';
	myChart.addSeries(
        ["name", "handedness", "HR", "avg"],
        dimple.plot.bubble
    );
    myChart.draw();
};