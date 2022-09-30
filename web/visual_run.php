<?php

// do a visualization run;
// form handler for visual.inc:visual_form()

require_once("panoseti.inc");

function main() {
    $run = get_str('run');
    $seconds = get_int('seconds');

    $cmd = sprintf('./write_images.py --run %s', $run);
    if ($seconds) {
        $cmd .= " --seconds $seconds";
    }
    system($cmd, $retval);
    if ($retval) {
        echo "$cmd returned $retval";
        return;
    }
    header("Location: analysis_type.php?type=visual&run=$run");
}

main();

?>
