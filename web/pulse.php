<?php

require_once("panoseti.inc");
require_once("pulse.inc");

// show pulse info for a given file/pixel
//

function show_file($file, $module, $pixel, $type, $dur) {
    $title = pulse_title($type);
    $path = pulse_file_name($file, $module, $pixel, $type, $dur);
    $size = 0;
    if (file_exists($path)) {
        $size = filesize($path);
    }
    if ($size == 0) {
        echo "<li>$title: none\n";
        return;
    }
    $url = sprintf('graph.php?file=%s&module=%d&pixel=%d&type=%s&dur=%d',
        $file, $module, $pixel, $type, $dur
    );
    echo "<li><a href=$url>$title</a> ($size bytes)";
}

function main($file, $module, $pixel) {
    page_head("Pulses for file $file");
    echo "<p>Module: $module\n";
    echo "<p>Pixel: $pixel\n";
    show_file($file, $module, $pixel, "value", 0);
    show_file($file, $module, $pixel, "value_hist", 0);
    for ($i=2; $i<16; $i++) {
        $x = 1<<$i;
        echo "<h3>Pulse duration $x</h3>\n";
        echo "<ul>";
        show_file($file, $module, $pixel, "all", $i);
        show_file($file, $module, $pixel, "mean", $i);
        show_file($file, $module, $pixel, "rms", $i);
        show_file($file, $module, $pixel, "thresh", $i);
        echo "</ul>";
    }
    page_tail();
}

$file = $_GET["file"];
$module = $_GET["module"];
$pixel = $_GET["pixel"];

main($file, $module, $pixel);

?>
