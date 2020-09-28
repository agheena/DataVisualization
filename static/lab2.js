var height = 450;
var width = 1000;
var fontFamily = 'Vedana'

select_category = new Array('Select sub-task');
task_category = new Array('Select sub-task', 'PCA_SCREE', 'PCA_SCREE_RANDOM', 'PCA_SCREE_STRAT', 'ORIGINAL_SCREE', 'RANDOM_SCREE', 'STRATIFIED_SCREE');
task_category_names = new Array('Select sub-task', "Intrinsic Dim Original", "Intrinsic Dim Random", "Intrinsic Dim Stratified", "Original Scree plot", "Random Scree plot", "Stratified Scree plot");
tasks = new Array('Select sub-task', 'PCA_ORIGINAL_SAMPLING', 'PCA_RANDOM_SAMPLING', 'PCA_ADAPTIVE_SAMPLING', 'MDS_EUCLIDEAN_ORIGINAL_SAMPLING', 'MDS_EUCLIDEAN_RANDOM_SAMPLING',
    'MDS_EUCLIDEAN_ADAPTIVE_SAMPLING', 'MDS_CORRELATION_ORIGINAL_SAMPLING', 'MDS_CORRELATION_RANDOM_SAMPLING', 'MDS_CORRELATION_ADAPTIVE_SAMPLING',
    'SCATTER_MATRIX_ORIGINAL_SAMPLING', 'SCATTER_MATRIX_RANDOM_SAMPLING', 'SCATTER_MATRIX_ADAPTIVE_SAMPLING');
task_names = new Array('Select sub-task', 'PCA Original Sampling', 'PCA Random Sampling', 'PCA Adaptive Sampling', 'MDS Euclidean Original Sampling', 'MDS Euclidean Random Sampling', 'MDS Euclidean Adaptive Sampling',
    'MDS Correlation Original Sampling', 'MDS Correlation Random Sampling', 'MDS Correlation Adaptive Sampling', 'Scatter Matrix Original Sampling', 'Scatter Matrix Random Sampling', 'Scatter Matrix Adaptive Sampling');

populateSelect();

$(function() {
    $('#map1 ').change(function() {
        populateSelect();
    });
});

function populateSelect() {
    map1 = $('#map1 ').val();
    console.log(map1);
    $('#map2').html('');

    if (map1 == 'Task 1') {
        task_category.forEach(function(t, i) {
            $('#map2').append('<option value=' + t + '>' + task_category_names[i] + '</option>');
        });
    }

    if (map1 == 'Task 2') {
        tasks.forEach(function(t, i) {
            $('#map2').append('<option value=' + t + '>' + task_names[i] + '</option>');
        });
    }

    if (map1 == 'Select Task') {
        select_category.forEach(function(t, i) {
            $('#map2').append('<option> ' + t + '</option>');
        });
    }
}

function draw_scree_plot(eigen_values, chart_title) {
    console.log("Inside draw_scree_plot");
    console.log(eigen_values);

    var data = JSON.parse(eigen_values);
    d3.select('#chart').remove();

    var margin = { top: 20, right: 20, bottom: 30, left: 60 };
    var width = 1366 - margin.left - margin.right;
    var height = 500 - margin.top - margin.bottom;

    var chart_width = 800;
    var chart_height = height + margin.top + margin.bottom;

    var x = d3.scale.linear().domain([1, data.length + 0.5]).range([0, chart_width - 120]);
    var y = d3.scale.linear().domain([0, d3.max(data)]).range([height, 0]);



    var xAxis = d3.svg.axis().scale(x).orient("bottom");
    var yAxis = d3.svg.axis().scale(y).orient("left");

    var markerX
    var markerY
    var color = d3.scale.category10();

    var line = d3.svg.line()
        .x(function(d, i) {
            if (i == 3) {
                markerX = x(i);
                markerY = y(d)
                console.log("markercircle", markerX + " " + markerY);
            }
            return x(i);
        })
        .y(function(d) {
            console.log('line', d);
            return y(d);
        })

    // Add an SVG element with the desired dimensions and margin.
    var graph = d3.select("body").append("svg")
        .attr("id", "chart")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom + 10)
        .append("g")
        .attr("transform", "translate(250,10)");

    // create yAxis
    graph.append("g")
        .attr("class", "x_axis")
        .attr("transform", "translate(110," + height + ")")
        .call(xAxis);

    // Add the y-axis to the left
    graph.append("g")
        .attr("class", "y_axis")
        .attr("transform", "translate(100,0)")
        .call(yAxis);

    graph.append("path")
        .attr("d", line(data))
        .attr("transform", "translate(215,0)")
        .attr("fill", "none")
        .attr("stroke", color(1))
        .attr("stroke-width", "3px")

    graph.append("circle")
        .attr("cx", markerX)
        .attr("cy", markerY)
        .attr("r", 8)
        .attr("transform", "translate(215,0)")
        .style("fill", "red");

    graph.append("text")
        .attr("class", "axis_label")
        .attr("text-anchor", "middle")
        .attr("transform", "translate(" + (50) + "," + (height / 2) + ")rotate(-90)")
        .style("font-size", "12px")
        .style("font-weight", "bold")
        .text("Eigen Values");

    graph.append("text")
        .attr("class", "axis_label")
        .attr("text-anchor", "middle")
        .attr("transform", "translate(" + (chart_width / 2) + "," + (chart_height) + ")")
        .style("font-size", "12px")
        .style("font-weight", "bold")
        .text("Principle Components");

    graph.append("text")
        .attr("x", (width / 3))
        .attr("y", 0 + (margin.top / 5))
        .attr("text-anchor", "middle")
        .style("font-size", "16px")
        .style("text-decoration", "underline")
        .style("font-weight", "bold")
        .text(chart_title);
}


//plotting bar chart for all the categorical attributes selected
function drawbarchart(result, chart_title) {
        // d3.select('#chart').remove();
    d3.select('#chart').remove();

    var variance = JSON.parse(result.explained_variance);


    // d3.select('svg').remove();

    var margin = {
            top: 50,
            right: 50,
            bottom: 80,
            left: 100
        },
        width = 800 - margin.left - margin.right,
        height = 600 - margin.top - margin.bottom;

    //d3.selectAll('#chart').remove()

    //svg defined for the bar chart and setting width and height attribute

    var svg = d3.select("body")
        .append("svg")
        .attr("id", "chart")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
        .append("g")
        .attr("transform",
            "translate(" + margin.left + "," + margin.top + ")");

    //counting the number of times a particular data appears and assigning it to an object with both count and attribute values
    //d3.select("#chart").append("div").attr("class", "myToolTip");
    var tooltip = d3.select("#chart").append("div").attr("class", "myToolTip");
    //d3.select("body").append("svg").attr("id", "chart").append("div").attr("class", "myToolTip");

    var obj = {};
    var finalArr = []

    var j = result.components
    forsum = 0
    for (var i = 1; i <= result.components; i++) {
        obj = new Object()
        obj["comp"] = i
        j = j - 1

        obj["vari"] = variance[j]
        forsum = forsum + variance[j]
        obj["cum_sum"] = forsum
        console.log(j)

        finalArr.push(obj)
    }

    // for (var j = 0; j < textdata.length; j++) {
    //     obj = new Object()
    //     obj["selectedAttribute"] = textdata[j]
    //     obj["frequency"] = count[j]
    //     finalArr.push(obj)

    // }

    //defining xscale
    var xScale = d3.scale.ordinal()
        .rangeRoundBands([0, width], .2)
        .domain(finalArr
            .map(function(a) {
                return a.comp;
            }));

    svg.append("g")
        .attr("transform", "translate(0," + height + ")")
        .call(d3.svg.axis()
            .scale(xScale)
            .orient("bottom"))
        .selectAll("text")
        .attr("y", 0)
        .attr("x", 5)
        .attr("dy", "2em")
        .style("text-anchor", "start");
    //labeling variable name on x-axis
    svg.append("text")
        .attr("transform",
            "translate(" + (width / 2) + " ," +
            (height + margin.top + 20) + ")")
        .style("text-anchor", "middle")
        .text("Principle Components");

    // Creating Y axis
    var yScale = d3.scale.linear()
        .domain([0, d3.max(finalArr, function(d) {
            //Math.max(parseInt(d.vari), parseInt(d.cum_sum));
          
            return Math.max(parseInt(d.vari), parseInt(d.cum_sum));;
        })])
        .rangeRound([height, 0]);
    svg.append("g")
        .call(d3.svg.axis()
            .scale(yScale)
            .orient("left"));
    //labeling variable count on y-axis

    svg.append("text")
        .attr("transform", "rotate(-90)")
        .attr("y", 0 - margin.left)
        .attr("x", 0 - (height / 2))
        .attr("dy", "1em")
        .style("text-anchor", "middle")
        .text("Variance Explained (%)");


    // Creating Bars
    svg.selectAll("bars")
        .data(finalArr)
        .enter()
        .append("rect")
        .attr("x", function(a) {
            return xScale(a.comp);
        })
        .attr("y", function(a) {
            return yScale(a.vari);
        })
        .attr("width", xScale.rangeBand())
        .attr("height", function(a) {
            return height - yScale(a.vari);
        })
        .attr("fill", "#69b3a2")
        .on("mousemove", function(a) {

            tooltip
                .style("left", (d3.event.pageX - 50) + "px")
                .style("top", (d3.event.pageY - 70) + "px")
                .style("display", "inline-block")
                .html((a.comp) + "<br>" + yScale(a.vari));
            d3.select(this).style('fill', 'orange')
                .style("transform", "scale(1.1,1.1)")
                .style("transform-origin", "50% 50%");
        }).on("mouseout", function(d) {
            d3.select(this).style('fill', '#69b3a2')
                .style("transform", "scale(1,1)")
            tooltip.style("display", "none");
        });

    svg.selectAll(".text")
        .data(finalArr)
        .enter().append("text")
        .attr("class", "label")
        .attr("text-anchor", "middle")
        .attr("x", function(d) { return xScale(d.comp) + xScale.rangeBand() / 2; })
        .attr("y", function(d) { return yScale(d.vari) - 5; })
        .text(function(d) { return Math.floor(d.vari * 100) / 100; });
    //svg.append("text")
    //     .attr("class", "label")
    //     //y position of the label is halfway down the bar
    //     .attr("y", function(d) {
    //         return yScale(d.vari) + y.rangeBand() / 2 + 4;
    //     })
    //     //x position is 3 pixels to the right of the bar
    //     .attr("x", function(d) {
    //         return xScale(d.comp) + 3;
    //     })
    //     .text(function(d) {
    //         return d.value;
    //     });
    // svg.selectAll(".text")
    //     .data(finalArr)
    //     .enter()
    //     .append("text")
    //     .attr("class", "label")
    //     .attr("x", (function(d) { return xScale(d.comp) + xScale.rangeBand() / 2; }))
    //     .attr("y", function(d) { return yScale(d.vari) + 1; })
    //     .attr("dy", ".75em")
    //     .text(function(d) { return d.vari; });






    var line = d3.svg.line()
        .x(function(d, i) {
            return xScale(d.comp);
        })
        .y(function(d) {
            return yScale(d.cum_sum);
        });

    svg.append("path")
        .datum(finalArr)
        .attr("class", "line")
        .attr("d", line)
        .style("fill", "none")
        .style("stroke", "steelblue")
        .style("stoke-width", "3px");
    svg.append("text")
        .attr("x", (width / 3))
        .attr("y", 0 + (margin.top / 3))
        .attr("text-anchor", "middle")
        .style("font-size", "16px")
        .style("text-decoration", "underline")
        .style("font-weight", "bold")
        .text(chart_title);
}







function drawScatterPlotMatrix(sData, rs, chart_title) {
    d3.select('#chart').remove();

    var jdata = JSON.parse(sData);
        //  To get column names of most weighted attributes/columns
    var ftrNames = Object.keys(jdata[0]);
    var width = 960,
        size = 230,
        padding = 20;

    var x = d3.scale.linear()
        .range([padding / 2, size - padding / 2]);

    var y = d3.scale.linear()
        .range([size - padding / 2, padding / 2]);

    var xAxis = d3.svg.axis()
        .scale(x)
        .orient("bottom")
        .ticks(6);

    var yAxis = d3.svg.axis()
        .scale(y)
        .orient("left")
        .ticks(6);


    data = {};
    var color = d3.scale.category10();
    data[ftrNames[0]] = [];
    data[ftrNames[1]] = [];
    data[ftrNames[2]] = [];
    data[ftrNames[3]] = [];

    for (var i = 0; i < Object.keys(jdata).length; ++i) {

        data[ftrNames[0]].push(Math.floor(jdata[i][ftrNames[0]] * 100) / 100);
        data[ftrNames[1]].push(Math.floor(jdata[i][ftrNames[1]] * 100) / 100);
        data[ftrNames[2]].push(Math.floor(jdata[i][ftrNames[2]] * 100) / 100);
        data[ftrNames[3]].push(Math.floor(jdata[i][ftrNames[3]] * 100) / 100);
  
    }



    var domainByFtr = {},
        ftrNames = d3.keys(data).filter(function(d) { return d !== "clusterid"; }),
        n = ftrNames.length;

    xAxis.tickSize(size * n);
    yAxis.tickSize(-size * n);
    //ftrNames = d3.keys()
    ftrNames.forEach(function(ftrName) {
        domainByFtr[ftrName] = d3.extent(d3.values(data[ftrName]));
    });

    var svg = d3.select("body").append("svg")
        .attr('id', 'chart')
        .attr("width", size * n + padding)
        .attr("height", size * n + padding)
        .append("g")
        .attr("transform", "translate(" + padding + "," + padding / 2 + ")");

    svg.selectAll(".x.axis")
        .data(ftrNames)
        .enter().append("g")
        .attr("class", "x axis")
        .attr("transform", function(d, i) { return "translate(" + (n - i - 1) * size + ",0)"; })
        .each(function(d) {
            x.domain(domainByFtr[d]);
            d3.select(this).call(xAxis);
        });

    svg.selectAll(".y.axis")
        .data(ftrNames)
        .enter().append("g")
        .attr("class", "y axis")
        .attr("transform", function(d, i) { return "translate(0," + i * size + ")"; })
        .each(function(d) {
            y.domain(domainByFtr[d]);
            d3.select(this).call(yAxis);
        });

    svg.append("text")
        .attr("x", (width / 2.8))
        .attr("y", 0 + (5))
        .attr("text-anchor", "middle")
        .style("font-size", "16px")
        .style("text-decoration", "underline")
        .style("font-weight", "bold")
        .text(chart_title);

    var cell = svg.selectAll(".cell")
        .data(cross(ftrNames, ftrNames))
        .enter().append("g")
        .attr("class", "cell")
        .attr("transform", function(d) { return "translate(" + (n - d.i - 1) * size + "," + d.j * size + ")"; })
        .each(plot);

    cell.filter(function(d) { return d.i === d.j; }).append("text")
        .attr("x", padding)
        .attr("y", padding)
        .attr("dy", ".71em")
        .text(function(d) { return d.x; });


    function plot(p) {
        var cell = d3.select(this);
        x.domain(domainByFtr[String(p.x)]);
        y.domain(domainByFtr[String(p.y)]);
        cell.append("rect")
            .attr("class", "frame")
            .attr("x", padding / 2)
            .attr("y", padding / 2)
            .attr("width", size - padding)
            .attr("height", size - padding);

        first_comp = data[String(p.x)];
        second_comp = data[String(p.y)];
        result_array = []
        second = d3.values(second_comp)
        cluster = data['clusterid']
        d3.values(first_comp).forEach(function(item, index) {
            temp_map = {};
            temp_map["x"] = item;
            temp_map["y"] = second[index];
            temp_map["clusterid"] = cluster[index];
            result_array.push(temp_map);
        });

        cell.selectAll("circle")
            .data(result_array)
            .enter().append("circle")
            .attr("cx", function(d) { return x(d.x); })
            .attr("cy", function(d) { return y(d.y); })
            .attr("r", 4)
            .style("fill", function(d) { return rs ? color("blue") : color(d.clusterid); });
    }
}

function cross(a, b) {
    var c = [],
        n = a.length,
        m = b.length,
        i, j;
    for (i = -1; ++i < n;)
        for (j = -1; ++j < m;) c.push({ x: a[i], i: i, y: b[j], j: j });
    return c;
}

function drawScatter(sData, rs, chart_title) {
    console.log("In draw scatter plot")
    d3.select('#chart').remove();
    var data = JSON.parse(sData);
    // var clust = JSON.parse(sData.clusterid);
   
    var array = [];
    var min = 0,
        max = 0;
    //  To get column names of most weighted attributes/columns
    ftrNames = Object.keys(data[0]);



    for (var i = 0; i < Object.keys(data).length; ++i) {
        obj = {}
        obj.x = Math.floor((data[i][0]) * 100) / 100;
        obj.y = Math.floor((data[i][1]) * 100) / 100;
        obj.clusterid = data[i]['clusterid']
            
        obj.ftr1 = data[i][ftrNames[2]]
        obj.ftr2 = data[i][ftrNames[3]]
        array.push(obj);
            // console.log(data['clusterid'][i]);
    }
    data = array;

    var margin = { top: 20, right: 20, bottom: 30, left: 40 },
        width = 960 - margin.left - margin.right,
        height = 500 - margin.top - margin.bottom;

    var xValue = function(d) { return d.x; },
        xScale = d3.scale.linear().range([0, width]),
        xMap = function(d) { return xScale(xValue(d)); },
        xAxis = d3.svg.axis().scale(xScale).orient("bottom");

    var yValue = function(d) { return d.y; },
        yScale = d3.scale.linear().range([height, 0]),
        yMap = function(d) { return yScale(yValue(d)); },
        yAxis = d3.svg.axis().scale(yScale).orient("left");

    var cValue
    if (rs) {
        cValue = function(d) { return d.clusteridx; }
    } else {
        cValue = function(d) { return d.clusterid; }
    }
    var color = d3.scale.category10();

    var svg = d3.select("body").append("svg")
        .attr('id', 'chart')
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
        .append("g")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

    var tooltip = d3.select("body").append('div').style('position', 'absolute');

    xScale.domain([d3.min(data, xValue) - 1, d3.max(data, xValue) + 1]);
    yScale.domain([d3.min(data, yValue) - 1, d3.max(data, yValue) + 1]);

    svg.append("g")
        .attr("transform", "translate(0," + height + ")")
        .attr("class", "x_axis")
        .call(xAxis)
        .append("text")
        .attr("class", "label")
        .attr("y", -6)
        .attr("x", width)
        .text("Compoenent 1")
        .style("text-anchor", "end");

    svg.append("g")
        .attr("class", "y_axis")
        .call(yAxis)
        .append("text")
        .attr("class", "label")
        .attr("y", 6)
        .attr("transform", "rotate(-90)")
        .attr("dy", ".71em")
        .text("Compoenent 2")
        .style("text-anchor", "end");

    svg.selectAll(".dot")
        .data(data)
        .enter().append("circle")
        .attr("class", "dot")
        .attr("cx", xMap)
        .attr("r", 3.5)
        .attr("cy", yMap)
        .style("fill", function(d) { return color(cValue(d)); })
        .on("mouseover", function(d) {
            tooltip.transition().style('opacity', .9).style(
                    'font-family', fontFamily).style('color', 'steelblue')
                // tooltip.html(ftrNames[2] + " : " + d.ftr1 + ", " + ftrNames[3] + " : " + d.ftr2)
            tooltip.html(d.x + ", " + d.y)
                .style("top", (d3.event.pageY - 28) + "px")
                .style("left", (d3.event.pageX + 5) + "px");
        })
        .on("mouseout", function(d) {
            tooltip.transition()
                .duration(500)
                .style("opacity", 0);
            tooltip.html('');
        });

    svg.append("text")
        .attr("x", (width / 2))
        .attr("y", 0 + (margin.top / 2))
        .attr("text-anchor", "middle")
        .style("font-size", "16px")
        .style("text-decoration", "underline")
        .style("font-weight", "bold")
        .text(chart_title);
}

function mapSelect() {
    console.log('mapselect')
    var dropdown = document.getElementById("map2");
    var selectedValue = dropdown.options[dropdown.selectedIndex].value;
    if (selectedValue == -1) {
        // Do nothing
    } else if (selectedValue == "PCA_ORIGINAL_SAMPLING") {
        console.log(selectedValue)
        get_map('/pca_original', true, false, false, false, 'PCA with Original Sampling');
    } else if (selectedValue == "PCA_RANDOM_SAMPLING") {
        console.log(selectedValue)
        get_map('/pca_random', true, false, false, false, 'PCA with Random Sampling');
    } else if (selectedValue == "PCA_ADAPTIVE_SAMPLING") {
        get_map('/pca_adaptive', false, false, false, false, 'PCA with Adaptive Sampling');
    } else if (selectedValue == "MDS_EUCLIDEAN_ORIGINAL_SAMPLING") {
        get_map('/mds_euclidean_original', true, false, false, false, 'MDS via Euclidean distance on Original Samples');
    } else if (selectedValue == "MDS_EUCLIDEAN_RANDOM_SAMPLING") {
        get_map('/mds_euclidean_random', true, false, false, false, 'MDS via Euclidean distance on Random Samples');
    } else if (selectedValue == "MDS_EUCLIDEAN_ADAPTIVE_SAMPLING") {
        get_map('/mds_euclidean_adaptive', false, false, false, false, 'MDS via Euclidean distance on Adaptive Samples');
    } else if (selectedValue == "MDS_CORRELATION_ORIGINAL_SAMPLING") {
        get_map('/mds_correlation_original', true, false, false, false, 'MDS via Correlation distance on Original Samples');
    } else if (selectedValue == "MDS_CORRELATION_RANDOM_SAMPLING") {
        get_map('/mds_correlation_random', true, false, false, false, 'MDS via Correlation distance on Random Samples');
    } else if (selectedValue == "MDS_CORRELATION_ADAPTIVE_SAMPLING") {
        get_map('/mds_correlation_adaptive', false, false, false, false, 'MDS via Correlation distance on Adaptive Sampels');
    } else if (selectedValue == "SCATTER_MATRIX_ORIGINAL_SAMPLING") {
        get_map('/scatter_matrix_original', true, true, false, false, 'Scatterplot Matrix on three highest attributes on Original Samples');
    } else if (selectedValue == "SCATTER_MATRIX_RANDOM_SAMPLING") {
        get_map('/scatter_matrix_random', true, true, false, false, 'Scatterplot Matrix on three highest attributes on Random Samples');
    } else if (selectedValue == "SCATTER_MATRIX_ADAPTIVE_SAMPLING") {
        get_map('/scatter_matrix_adaptive', false, true, false, false, 'Scatterplot Matrix on three highest attributes on Adaptive Samples');
    } else if (selectedValue == "PCA_SCREE") {
        console.log(selectedValue)
        get_map('/pca_scree', false, false, true, false, 'Scree plot to find intrinsic dimensioanlity for Original Data');
    } else if (selectedValue == "PCA_SCREE_RANDOM") {
        get_map('/pca_scree_rs', false, false, true, false, 'Scree plot to find intrinsic dimensioanlity for Random Data');
    } else if (selectedValue == "PCA_SCREE_STRAT") {
        console.log(selectedValue)
        get_map('/pca_scree_ss', false, false, true, false, 'Scree plot to find intrinsic dimensioanlity for Stratified data');
    } else if (selectedValue == "ORIGINAL_SCREE") {
        get_map('/original_Scree', false, false, false, true, 'Scree plot before sampling');
    } else if (selectedValue == "RANDOM_SCREE") {
        get_map('/random_scree', false, false, false, true, 'Scree plot after Random sampling');
    } else if (selectedValue == "STRATIFIED_SCREE") {
        get_map('/stratified_scree', false, false, false, true, 'Scree plot after Stratified sampling');
    }


    d3.select('#scree').remove();
}

function get_map(url, rs, matrix, isScree, isBar, chart_title) {
    $.ajax({
        type: 'GET',
        url: url,
        contentType: 'application/json; charset=utf-8',
        xhrFields: {
            withCredentials: false
        },
        headers: {},
        success: function(result) {
            if (matrix) {
                drawScatterPlotMatrix(result, rs, chart_title);
            } else if (!matrix && !isBar) {
                drawScatter(result, rs, chart_title);
            }

            if (isScree) {
                draw_scree_plot(result, chart_title)
            }
            if (isBar) {
                drawbarchart(result, chart_title)
            }
        },
        error: function(result) {
            $("#error").html(result);
        }
    });
}