<?php

require_once("bootstrap.inc");

// fields in output files
define('OFFSET_SAMPLE', 0);
define('OFFSET_VALUE',  1);
define('OFFSET_MEAN',   2);
define('OFFSET_STDDEV', 3);
define('OFFSET_NSIGMA', 4);
define('OFFSET_PIXEL',  5);

function ph_coincidence_form($vol, $run) {
    form_start("ph_coincidence_run.php");
    form_input_hidden('vol', $vol);
    form_input_hidden('run', $run);
    form_input_text("Max frame time difference between modules", "max_time_diff");
    form_input_text(
        "Max ADC minimum threshold<br><small>Minimum threshold of the max adc pixel value for a frame's inclusion in the analysis</small>",
        "threshold_max_adc");
    form_input_text("Modules<br><small>space-separated list of comma-separated pairs module IDs </small>", "modules");
    form_checkboxes("All modules", [['all_modules', '', false]]);
    form_submit("OK");
    form_end();
}

function ph_coincidence_params_str($params) {
    $m = $params->modules?implode(',', $params->modules):'';
    return sprintf(
        "Max time diff: %f<br>Threshold: %f<br>Modules: %s<br>All Modules: %s",
        $params->max_time_diff, $params->threshold,
        $m, $params->all_modules?"Yes":"No",
    );
}

?>
