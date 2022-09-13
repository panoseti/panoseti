<?php

// do an image pulse run;
// form handler for img_pulse.inc:img_pulse_form()

require_once("panoseti.inc");

function main() {
    $run = get_str('run');
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
        './img_pulse.py --run %s --nlevels %d --win_size %d --thresh %f',
        $run, $nlevels, $win_size, $thresh
    );
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
    system($cmd, $retval);
    if ($retval) {
        echo "$cmd returned $retval";
        return;
    }
    header("Location: analysis_type.php?type=img_pulse&run=$run");
}

main();

?>
