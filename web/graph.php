<?php

// show a zoomable graph of a .csv file specified by path

require_once("graph.inc");

function main($path, $type) {
    echo '
        <!DOCTYPE html>
        <meta charset="utf-8">
    ';

    $url = "https://setiathome.berkeley.edu/panoseti/$path";
    list($xmin, $xmax, $ymin, $ymax, $xname, $yname) = get_extrema($path);

    // kludge
    if ($type == "rms") {
        $yname = "rms";
    }

    zoom_init();

    zoom_graph(
        $url,
        1000, 600,
        "Frame number", "Value",
        $xname, $yname,
        $xmin, $xmax,
        $ymin, $ymax
    );
    echo "file: $path";
}

$path = $_GET['path'];
$type = $_GET['type'];

// security checks
//
if (strstr($path, "..")) die("");
if ($path[0] == "/") die("");
main($path, $type);

?>
