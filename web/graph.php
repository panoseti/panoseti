<?php

ini_set('display_errors', 1);

// show a zoomable graph of a .csv file specified by path
//
// Also, since graphs with lots of points are slow to draw in Javascript,
// you can zoom and move in the data itself.

require_once("panoseti.inc");
require_once("graph.inc");
require_once("pulse.inc");

function make_url($file, $pixel, $type, $dur, $start, $n) {
    return "graph.php?file=$file&pixel=$pixel&type=$type&dur=$dur&start=$start&n=$n";
}

function main($start, $n, $file, $pixel, $type, $dur) {
    page_head(pulse_title($type));

    $path = pulse_file_name($file, $pixel, $type, $dur);

    if ($n) {
        $last = $start+$n;
        $cmd = sprintf("sed -n '1p;%d,%dp;%dq' %s > %s",
            $start, $last, $last+1, $path, "tmp/graph.csv"
        );
        system($cmd);
        $path = "tmp/graph.csv";
    }

    $url = "https://visigoth.ucolick.org/$path";
    [$xmin, $xmax, $ymin, $ymax, $xname, $yname] = get_extrema($path);

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
        $dy = $ymax - $ymin;
        $ymin -= $dy/100;
        $ymax += $dy/100;
    }
    $dx = $xmax-$xmin;
    $xmin -= $dx/100;
    $xmax += $dx/100;

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
    echo "<br>";
    $btn = 'class="btn btn-primary btn-sm"';

    // left
    //
    if ($n && $start > 2) {
        $new_start = $start - $n;
        if ($new_start<2) $new_start = 2;
        $url = make_url($file, $pixel, $type, $dur, $new_start, $n);
        echo "<a $btn href=$url><<</a> &nbsp";
    }

    // zoom in
    //
    if ($n) {
        $new_start += (int)($n/2);
        $new_n = (int)($n/10);
    } else {
        $m = count(file($path));
        $new_start = (int)($m/2);
        $new_n = (int)($m/10);
    }
    $url = make_url($file, $pixel, $type, $dur, $new_start, $new_n);
    echo "<a $btn href=$url>Zoom in 10X</a>";

    // zoom out
    //
    if ($n) {
        $new_n = $n*10;
        $new_start = $start - (int)($new_n/2);
        if ($new_start < 2) $new_start = 2;
        $url = make_url($file, $pixel, $type, $dur, $new_start, $new_n);
        echo "&nbsp; <a $btn href=$url>Zoom out 10X</a>";

    // right
    //
        $new_start = $start + $n;
        $url = make_url($file, $pixel, $type, $dur, $new_start, $n);
        echo "&nbsp; <a $btn href=$url>>></a>";
    }
    echo "<br><br>";

    echo "<p>File: $file";
    echo "<p>Pixel: $pixel";
    $d = 1<<$dur;
    if ($type != "value") {
        echo "<p>Number of frames integrated: $d";
    }
    page_tail();
}

$file = get_str('file');
$pixel = get_int('pixel');
$type = get_str('type');
$dur = get_int('dur');
$start = get_int('start');
$n = get_int('n');

check_filename($file);

main($start, $n, $file, $pixel, $type, $dur);

?>
