<?php

require_once("bootstrap.inc");

// fields in output files
define('OFFSET_SAMPLE', 0);
define('OFFSET_VALUE',  1);
define('OFFSET_MEAN',   2);
define('OFFSET_STDDEV', 3);
define('OFFSET_NSIGMA', 4);
define('OFFSET_PIXEL',  5);

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
        [['log_all', '', false]]
    );
    form_submit("OK");
    form_end();
}

function img_pulse_params_str($params) {
    $p = $params->pixels?implode(',', $params->pixels):'';
    return sprintf(
        "#levels: %d<br>Stats window size: %d<br>Threshold: %f<br>Seconds: %f<br>Pixels: %s<br>All pixels: %s<br>Log all: %s",
        $params->nlevels, $params->win_size, $params->thresh, $params->seconds,
        $p, $params->all_pixels?"Yes":"No",
        $params->log_all?"Yes":"No"
    );
}

?>
