<?php

require_once("bootstrap.inc");

function img_pulse_show() {
    $x = 'Pixels: ';
    foreach (scandir(ANALYSIS_ROOT."/$run/$fname") as $pixel) {
        if ($pixel[0] == ".") continue;
        if (!is_numeric($pixel)) continue;
        $url = "pulse.php?file=$run/$fname&pixel=$pixel";
        $x .= "&nbsp;&nbsp <a href=$url>$pixel</a>\n";
    }
    row2("Pulse info", $x);
}

function img_pulse_form($run) {
    form_start("img_pulse_run.php");
    form_input_hidden('run', $run);
    form_input_text("# of duration octaves", "nlevels");
    form_input_text("Size of window for mean/stddev", "win_size");
    form_input_text("Threshold", "thresh");
    form_input_text(
        "Seconds<br><small>Analyze first N seconds of files</small>",
        "seconds"
    );
    form_input_text("Pixels<br><small>comma-separated list of pixels (0-1023)</small>", "pixels");
    form_checkboxes("All pixels", [['all_pixels', '', false]]);
    form_checkboxes(
        "Record all<br><small>Write all samples to a separate file (can be huge)</small>",
        [['record_all', '', false]]
    );
    form_submit("OK");
    form_end();
}

function img_pulse_file_path($run, $analysis_dir, $module_dir, $pixel_dir, $type, $dur) {
    $x = "analysis/$run/img_pulse/$analysis_dir/$module_dir/$pixel_dir/$type";
    switch($type) {
    case "value":
    case "value_hist":
        break;
    default:
        $x .= "_$dur";
    }
    return $x;
}

function img_pulse_title($type) {
    if ($type == 'value') return "Pixel values";
    if ($type == 'value_hist') return "Pixel value histogram";
    if ($type == 'all') return "Intensity";
    if ($type == 'mean') return "Local mean intensity";
    if ($type == 'stddev') return "Local stddev";
    if ($type == 'thresh') return "Pulses > 3 sigma above mean intensity";
    return "bad type";
}

?>
