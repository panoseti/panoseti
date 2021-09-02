<?php
require_once("panoseti.inc");
require_once("pulse.inc");

function show_list($file, $module, $pixel, $dur, $nsigma) {
    page_head("Pulses above $nsigma sigma");
    $fname = pulse_file_name($file, $module, $pixel, 'all', $dur);
    $lines = file($fname);
    start_table("table-striped");
    table_header("time (sec)", "mean intensity", "sigma");
    foreach ($lines as $line) {
        $x = explode(',', $line);
        $s = (double)$x[2];
        if ($s > $nsigma) {
            table_row($x[0], $x[1], $s);
        }
    }
    end_table();
    page_tail();
}

function main() {
}

$file = get_str("file");
$module = get_int("module");
$pixel = get_int("pixel");
$dur = get_int("dur");
$nsigma = (double)get_str("nsigma");

show_list($file, $module, $pixel, $dur, $nsigma);

?>
