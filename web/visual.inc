<?php

require_once("bootstrap.inc");

function visual_form($run) {
    form_start("visual_run.php");
    form_input_hidden('run', $run);
    form_input_text(
        "Seconds<br><small>Analyze first N seconds of files</small>",
        "seconds"
    );
    form_submit("OK");
    form_end();
}

function visual_params_str($params) {
    return sprintf("Seconds: %f", $params->seconds);
}

?>
