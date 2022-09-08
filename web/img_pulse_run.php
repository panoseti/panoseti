<?php

// do an image pulse run;
// form handler for img_pulse.inc:img_pulse_form()

require_once("panoseti.inc");

function main($run, $nlevels, $win_size, $thresh, $pixels, $all_pixels) {
    if (!$all_pixels && strlen($pixels)==0) {
        error_page("no pixels specified");
    }
    if ($all_pixels) {
        $cmd = sprintf(
            'pulse.py --run %s --nlevels %d --win_size %d --thresh %f',
            $run, $nlevels, $win_size, $thresh
        );
        //echo $cmd;
        system($cmd);
    }
    if (strlen($pixels)) {
        $pixels = str_replace(' ', '', $pixels);
        $cmd = sprintf(
            'pulse.py --run %s --nlevels %d --win_size %d --thresh %f --pixels %s',
            $run, $nlevels, $win_size, $thresh, $pixels
        );
        //echo $cmd;
        system($cmd);
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
