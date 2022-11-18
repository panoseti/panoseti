<?php

// show pulse height coincidence analysis output.
// if only "dir" (analysis dir):
//      for each module_pair
//          link to coincident event browser
//

require_once("panoseti.inc");
require_once("ph_coincidence.inc");
require_once("analysis.inc");

function show_analysis($vol, $run, $analysis_dir) {
    $dirpath = "$vol/analysis/$run/ph_coincidence/$analysis_dir";
    page_head("Pulse height pulse analysis");
    analysis_page_intro('ph_coincidence', $analysis_dir, $dirpath);
    $module_pair_fname_pattern = "{module_(\d+)\.module_(\d+)}"
    foreach (scandir($dirpath) as $mpdir) {
        if (preg_match($module_pair_fname_pattern, $mpdir, $matches)) {
            $modules = implode(',', array_slice($matches, 1);
            //TODO: Fix stuff after this line
            echo "<h3>Module $mnum</h3><ul>";
            $subdir = "$dirpath/$mdir";
            foreach (scandir($subdir) as $pdir) {
                if ($pdir == 'all_pixels' || substr($pdir, 0, 6) == 'pixel_') {
                    $url = sprintf(
                        "img_pulse_show.php?action=detail&vol=%s&run=%s&analysis=%s&module=%s&pixel=%s",
                        $vol, $run, $analysis_dir, $mdir,$pdir
                    );
                    $n = npulses_pixel($vol, $run, $analysis_dir, $mdir, $pdir);
                    echo "<li><a href=$url>$pdir</a>($n pulses)<br>";
                }
            }
        }
        echo "</ul>";
    }
    page_tail();
}

$run = get_str("run");
$vol = get_str("vol");
check_filename($run);
check_filename($vol);
$analysis_dir = get_str("analysis");
check_filename($analysis_dir);

$action = get_str('action', true);

if (!$action) {
    show_analysis($vol, $run, $analysis_dir);
} else if ($action == 'detail') {
    $module_dir = get_str('module', true);
    $pixel_dir = get_str('pixel', true);
    show_pixel_dir($vol, $run, $analysis_dir, $module_dir, $pixel_dir);
} else if ($action == 'list') {
    $module_dir = get_str('module', true);
    $pixel_dir = get_str('pixel', true);
    $level = get_int('level');
    show_list($vol, $run, $analysis_dir, $module_dir, $pixel_dir, $level);
} else if ($action == 'graph') {
    $module_dir = get_str('module', true);
    $pixel_dir = get_str('pixel', true);
    $level = get_int('level');
    $start = get_int('start');
    $n = get_int('n');
    show_graph(
        $vol, $run, $analysis_dir, $module_dir, $pixel_dir, $level, $start, $n
    );
} else {
    error_page("bad action $action");
}

?>
