<?php

// show a zoomable graph of a .csv file specified by path

require_once("panoseti.inc");
require_once("graph.inc");
require_once("pulse.inc");

function main($file, $module, $pixel, $type, $dur) {
    page_head(pulse_title($type));

    $path = pulse_file_name($file, $module, $pixel, $type, $dur);

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
    echo "<p>file: $file";
    echo "<p>module: $module";
    echo "<p>pixel: $pixel";
    $d = 2<<$dur;
    if ($type != "value") {
        echo "<p>pulse duration: $d";
    }
    page_tail();
}

$file = $_GET['file'];
$module = (int)$_GET['module'];
$pixel = (int)$_GET['pixel'];
$type = $_GET['type'];
$dur = (int)$_GET['dur'];

// security checks
//
if (strstr($file, "..")) die("");
if ($file[0] == "/") die("");

main($file, $module, $pixel, $type, $dur);

?>
