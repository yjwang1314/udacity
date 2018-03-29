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
    
	myChart.addMeasureAxis("x", "height")
		   .overrideMin = 60
           .overrideMax = 160; 
	myChart.addMeasureAxis("y", "weight")
		   .overrideMin = 100
           .overrideMax = 400;
	//x.dateParseFormat = "%Y";
	//x.tickFormat = "%Y";
	//x.timeInterval = 4;
	//myChart.addSeries(null, dimple.plot.line);
	//myChart.addSeries(null, dimple.plot.scatter);
	//myChart.draw();
	myChart.addSeries(
        ["name", "handedness", "HR", "avg"],
        dimple.plot.bubble
    );
    myChart.draw();


};