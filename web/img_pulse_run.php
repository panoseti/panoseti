<?php

// do an image pulse run;
// form handler for img_pulse.inc:img_pulse_form()

require_once("panoseti.inc");

function main() {
    $username = get_login();
    $run = get_str('run');
    $vol = get_str('vol');
    $nlevels = get_int('nlevels');
    $win_size = get_int('win_size');
    $thresh = (double)get_str('thresh');
    $pixels = get_str('pixels');
    $seconds = get_int('seconds');
    $all_pixels = get_str('all_pixels', true)?1:0;
    $log_all = get_str('log_all', true)?1:0;

    if (!$all_pixels && strlen($pixels)==0) {
        error_page("no pixels specified");
    }
    $cmd = sprintf(
        './img_pulse.py --vol %s --run %s --nlevels %d --win_size %d --thresh %f',
        $vol, $run, $nlevels, $win_size, $thresh
    );
    if ($username) {
        $cmd .= " --username $username ";
    }
    if ($all_pixels) {
        $cmd .= ' --all_pixels';
    }
    if ($log_all) {
        $cmd .= ' --log_all';
    }
    if ($seconds) {
        $cmd .= " --seconds $seconds";
    }
    if (strlen($pixels)) {
        $pixels = str_replace(' ', '', $pixels);
        $cmd .= " --pixels $pixels";
    }
    page_head("Image pulse analysis run");
    echo "Command: $cmd
        <p>Output:
        <p>
        <pre>
    ";
    system($cmd, $retval);
    echo "</pre>";
    if ($retval) {
        echo "$cmd returned $retval";
        return;
    }
    echo "<p>
        <a href=analysis_type.php?type=img_pulse&vol=$vol&run=$run>
            Image pulse analysis of this run
        </a>
        </p>
    ";
    page_tail();
}

main();

?>
