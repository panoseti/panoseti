<?php

// show pulse info for a given file/pixel
//

function show_file($path, $title, $type) {
    if (!file_exists($path)) return;
    $size = filesize($path);
    if ($size == 0) return;
    echo "<p><a href=graph.php?path=$path&type=$type>$title</a> ($size bytes)";
}

function main($file, $pixel) {
    $dir = "pulse_out/$file/$pixel";
    for ($i=0; $i<16; $i++) {
        $x = 2<<$i;
        echo "<h2>Pulse duration $x</h2>\n";
        $all = "pulse_out/$file/$pixel/all_$i";
        $stats = "pulse_out/$file/$pixel/stats_$i";
        $pulse = "pulse_out/$file/$pixel/pulse_$i";
        show_file($all, "All pulses", "all");
        show_file($stats, "Mean", "mean");
        show_file($stats, "RMS", "rms");
        show_file($pulse, "Pulses above threshold", "thresh");
    }
}

$file = $_GET["file"];
$pixel = $_GET["pixel"];

main($file, $pixel);

?>
