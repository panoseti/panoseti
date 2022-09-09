<?php

// do an image pulse run;
// form handler for img_pulse.inc:img_pulse_form()

require_once("panoseti.inc");

function main($run, $nlevels, $win_size, $thresh, $pixels, $all_pixels) {
    if (!$all_pixels && strlen($pixels)==0) {
        error_page("no pixels specified");
    }
    $cmd = sprintf(
        'img_pulse.py --run %s --nlevels %d --win_size %d --thresh %f',
        $run, $nlevels, $win_size, $thresh
    );
    if ($all_pixels) {
        $cmd .= ' --all_pixels';
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

$run = get_str('run');
$nlevels = get_int('nlevels');
$win_size = get_int('win_size');
$thresh = (double)get_str('thresh');
$pixels = get_str('pixels');
$all_pixels = get_str('all_pixels', true)?true:false;

main($run, $nlevels, $win_size, $thresh, $pixels, $all_pixels);

?>
