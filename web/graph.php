<?php

// show a zoomable graph of a .csv file specified by path
//
// Also, since graphs with lots of points are slow to draw in Javascript,
// you can zoom and move in the data itself.

require_once("panoseti.inc");
require_once("graph.inc");
require_once("pulse.inc");

function main($action, $start, $n, $file, $module, $pixel, $type, $dur) {
    page_head(pulse_title($type));

    $path = pulse_file_name($file, $module, $pixel, $type, $dur);

    if ($action == 'zoom_in') {
        if ($n) {
            $start += (int)($n/2);
            $n = (int)($n/10);
        } else {
            $n = count(file($path));
            $start = (int)($n/2);
            $n = (int)($n/10);
        }
    } else if ($action == 'zoom_out') {
        $n *= 10;
        $start -= (int)($n/2);
        if ($start < 2) $start = 2;
    } else if ($action == 'left') {
        $start -= $n;
        if ($start < 2) $start = 2;
    } else if ($action == 'right') {
        $start += $n;
    }

    if ($n) {
        $last = $start+$n;
        $cmd = sprintf("sed -n '1p;%d,%dp;%dq' %s > %s",
            $start, $last, $last+1, $path, "tmp/graph.csv"
        );
        system($cmd);
        $path = "tmp/graph.csv";
    }

    $url = "https://setiathome.berkeley.edu/panoseti/$path";
    list($xmin, $xmax, $ymin, $ymax, $xname, $yname) = get_extrema($path);

    // kludge
    if ($type == "stddev") {
        $yname = "stddev";
    }

    zoom_init();

    $ylogscale = false;
    if ($type == "value_hist") {
        $xtitle = "Pixel value";
        $ytitle = "log(#Pixels)";
        $ylogscale = true;
    } else {
        $xtitle = "Seconds";
        $ytitle = "Mean intensity";
        $ymin -= 10;
        $ymax += 10;
    }
    $xmin -= 1;
    $xmax += 1;

    zoom_graph(
        $url,
        1200, 600,
        $xtitle, $ytitle,
        $xname, $yname,
        $xmin, $xmax,
        $ymin, $ymax,
        $ylogscale
    );

    // show zoom/pan buttons
    //
    $my_url = "graph.php?file=$file&module=$module&pixel=$pixel&type=$type&dur=$dur&start=$start&n=$n";
    echo "<br>";
    $btn = 'class="btn btn-primary btn-sm"';
    if ($n && $start > 0) {
        echo "<a $btn href=$my_url&action=left><<</a> &nbsp";
    }

    echo "<a $btn href=$my_url&action=zoom_in>Zoom in 10X</a>";
    if ($n) {
        echo "&nbsp; <a $btn href=$my_url&action=zoom_out>Zoom out 10X</a>";
        echo "&nbsp; <a $btn href=$my_url&action=right>>></a>";
    }
    echo "<br><br>";

    echo "<p>File: <a href=data_file.php?name=$file>$file</a>";
    echo "<p>Dome: $module";
    echo "<p>Pixel: $pixel";
    $d = 1<<$dur;
    if ($type != "value") {
        echo "<p>Number of frames integrated: $d";
    }
    page_tail();
}

$file = get_str('file');
$module = get_int('module');
$pixel = get_int('pixel');
$type = get_str('type');
$dur = get_int('dur');
$action = get_str('action');
$start = get_int('start');
$n = get_int('n');

check_filename($file);

main($action, $start, $n, $file, $module, $pixel, $type, $dur);

?>
