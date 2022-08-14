<?php

ini_set('display_errors', 1);

require_once("panoseti.inc");
require_once("pulse.inc");

// show pulse info for a given file/module/pixel
//

function show_file($file, $pixel, $type, $dur) {
    $title = pulse_title($type);
    $path = pulse_file_name($file, $pixel, $type, $dur);
    $size = 0;
    if (file_exists($path)) {
        $size = filesize($path);
    }
    if ($size == 0) {
        echo "<li>$title: none\n";
        return;
    }
    $url = sprintf('graph.php?file=%s&pixel=%d&type=%s&dur=%d',
        $file, $pixel, $type, $dur
    );
    echo "<li><a href=$url>$title</a> ($size bytes)";
}

function sigma_form($file, $pixel, $dur) {
    echo "<li><form action=pulse_sigma.php>Pulses above
        <input type=text name=nsigma size=4> sigma
        <input type=hidden name=file value=$file>
        <input type=hidden name=pixel value=$pixel>
        <input type=hidden name=dur value=$dur>
        <input type=submit class=\"btn-sm btn-primary\" name=list value=list>
        </form>
    ";
}

function main($file, $pixel) {
    page_head("Software pulse info", LOGIN_OPTIONAL);
    $x = explode('/', $file);
    echo sprintf("Run: %s", $x[0]);
    echo sprintf("<p>File: %s", $x[1]);
    echo "<p>Pixel: $pixel\n";
    show_file($file, $pixel, "value_hist", 0);
    for ($i=0; $i<16; $i++) {
        $x = 1<<$i;
        echo "<h3>Pulse duration $x</h3>\n";
        echo "<ul>";
        show_file($file, $pixel, "all", $i);
        show_file($file, $pixel, "mean", $i);
        show_file($file, $pixel, "stddev", $i);
        show_file($file, $pixel, "thresh", $i);
        sigma_form($file, $pixel, $i);
        echo "</ul>";
    }
    page_tail();
}

$file = get_str("file");
$pixel = get_int("pixel");
check_filename($file);
main($file, $pixel);

?>
