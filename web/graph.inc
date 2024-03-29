<?php

// functions to display dynamic graphs
// i.e. that let you zoom and pan
// see https://www.d3-graph-gallery.com/graph/interactivity_zoom.html
//

DEPRECATED

// get x/y min/max/name from a .csv file
//
function get_extrema($file) {
    $fin = fopen($file, "r");
    if (!$fin) die("no file $file");
    $i = 0;
    while ($line = fgets($fin)) {
        if ($i == 0) {
            $x = explode(",", $line);
            $xname = $x[0];
            $yname = $x[1];
            $i = 1;
            continue;
        }
        sscanf($line, "%f,%f", $x, $y);
        if ($i == 1) {
            $i = 2;
            $xmin = $xmax = $x;
            $ymin = $ymax = $y;
            continue;
        }
        if ($x < $xmin) $xmin = $x;
        if ($x > $xmax) $xmax = $x;
        if ($y < $ymin) $ymin = $y;
        if ($y > $ymax) $ymax = $y;
    }
    if ($i<2) die("empty file $file");
    return [$xmin, $xmax, $ymin, $ymax, $xname, $yname];
}

// call this first
//
function zoom_init() {
    echo '<script src="https://d3js.org/d3.v4.js"></script>
';
}

// for now you can only have one graph per page.
// To change this you'd have to append a per-graph string
// to all the javascript variables below.

$x = <<<EOT
<!-- Create a div where the circle will take place -->
<div id="%s"></div>


<script>

// set the dimensions and margins of the graph
var margin = {top: 10, right: 30, bottom: 40, left: 60},
    width = %d - margin.left - margin.right,
    height = %d - margin.top - margin.bottom;

// append the SVG object to the body of the page
var SVG = d3.select("#%s")
  .append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
    .style("background-color", "white")
  .append("g")
    .attr("transform",
          "translate(" + margin.left + "," + margin.top + ")");

//Read the data
d3.csv("%s", function(data) {

  // Add X axis
  var x = d3.scaleLinear()
    .domain([%d, %d])
    .range([ 0, width ]);
  var xAxis = SVG.append("g")
    .attr("transform", "translate(0," + height + ")")
    .call(d3.axisBottom(x));

  SVG.append("text")             
      .attr("transform",
            "translate(" + (width/2) + " ," + 
                           (height + margin.top + 25) + ")")
      .style("text-anchor", "middle")
      .text("%s");

  // Add Y axis
  var y = d3.%s()
    .domain([%d, %d])
    .range([ height, 0]);
  var yAxis = SVG.append("g")
    .call(d3.axisLeft(y));

  // text label for the y axis
  SVG.append("text")
      .attr("transform", "rotate(-90)")
      .attr("y", 0 - margin.left)
      .attr("x",0 - (height / 2))
      .attr("dy", "1em")
      .style("text-anchor", "middle")
      .text("%s");      

  // Add a clipPath: everything out of this area won't be drawn.
  var clip = SVG.append("defs").append("SVG:clipPath")
      .attr("id", "clip")
      .append("SVG:rect")
      .attr("width", width )
      .attr("height", height )
      .attr("x", 0)
      .attr("y", 0);

  // Create the scatter variable: where both the circles and the brush take place
  var scatter = SVG.append('g')
    .attr("clip-path", "url(#clip)")

  // Add circles
  scatter
    .selectAll("circle")
    .data(data)
    .enter()
    .append("circle")
      .attr("cx", function (d) { return x(d.%s); } )
      .attr("cy", function (d) { return y(d.%s); } )
      .attr("r", 3)
      .style("fill", "#6183b9")
      .style("opacity", 0.5)

  // Set the zoom and Pan features: how much you can zoom, on which part, and what to do when there is a zoom
  var zoom = d3.zoom()
      .scaleExtent([.5, 20])  // This control how much you can unzoom (x0.5) and zoom (x20)
      .extent([[0, 0], [width, height]])
      .on("zoom", updateChart);

  // This add an invisible rect on top of the chart area. This rect can recover pointer events: necessary to understand when the user zoom
  SVG.append("rect")
      .attr("width", width)
      .attr("height", height)
      .style("fill", "none")
      .style("pointer-events", "all")
      .attr('transform', 'translate(' + margin.left + ',' + margin.top + ')')
      .call(zoom);
  // now the user can zoom and it will trigger the function called updateChart

  // A function that updates the chart when the user zoom and thus new boundaries are available
  function updateChart() {

    // recover the new scale
    var newX = d3.event.transform.rescaleX(x);
    var newY = d3.event.transform.rescaleY(y);

    // update axes with these new boundaries
    xAxis.call(d3.axisBottom(newX))
    yAxis.call(d3.axisLeft(newY))

    // update circle position
    scatter
      .selectAll("circle")
      .attr('cx', function(d) {return newX(d.%s)})
      .attr('cy', function(d) {return newY(d.%s)});
  }

})
</script>
EOT;

function zoom_graph(
    $file,              // URL of .csv data file
    $width, $height,    // graph size in pixels
    $xlabel, $ylabel,
        // get the following from .csv file using get_extrema()
    $xname, $yname,             // column names
    $xmin, $xmax, $ymin, $ymax,  // data extrema
    $ylogscale
) {
    global $x;
    static $i=0;

    $divname = "div$i";
    $i++;

    echo sprintf(
        $x, $divname, $width, $height, $divname, $file,
        $xmin, $xmax, $xlabel,
        $ylogscale?"scaleLog":"scaleLinear",
        $ymin, $ymax, $ylabel,
        $xname, $yname, $xname, $yname
    );
}

//zoom_graph("foo", 40, 40, "x", "y", "x","y",0,10,0,10,false);
?>
