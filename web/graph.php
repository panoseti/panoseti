<?php

// show a zoomable graph of a .csv file specified by path

require_once("graph.inc");

function main($path) {
    echo '
        <!DOCTYPE html>
        <meta charset="utf-8">
    ';

    $url = "https://setiathome.berkeley.edu/panoseti/$path";
    list($xmin, $xmax, $ymin, $ymax, $xname, $yname) = get_extrema($path);

    zoom_init();

    zoom_graph(
        $url,
        $xname, $yname,
        1000, 600,
        $xmin, $xmax,
        $ymin, $ymax
    );
}

$path = $_GET['path'];

// security checks
//
if (strstr($path, "..")) die("");
if ($path[0] == "/") die("");
main($path);

?>
