<?php

// show pulse height coincidence analysis output.
// if only "dir" (analysis dir):
//      for each module_pair
//          link to coincident event browser
//

require_once("panoseti.inc");
require_once("ph_coincidence.inc");
require_once("analysis.inc");

function get_num_events($vol, $run, $analysis_dir, $module_pair_dir) {
    $dirpath = "$vol/analysis/$run/ph_coincidence/$analysis_dir/$module_pair_dir";
    $event_pattern = "/^event_\d+.*/";
    $n = 0;
    foreach(scandir($dirpath) as $event) {
        if (preg_match($event_pattern, $event)) {
            $n += 1;
        }
    }
    return $n;
}

function show_analysis($vol, $run, $analysis_dir) {
    $dirpath = "$vol/analysis/$run/ph_coincidence/$analysis_dir";
    page_head("Pulse height pulse analysis");
    analysis_page_intro('ph_coincidence', $analysis_dir, $dirpath);
    $module_pair_fname_pattern = "/module_(\d+)\.module_(\d+)/";
    foreach (scandir($dirpath) as $mpdir) {
        if (preg_match($module_pair_fname_pattern, $mpdir, $matches)) {
            $num_events = get_num_events($vol, $run, $analysis_dir, $mpdir);
            $module_pair = implode(',', array_slice($matches, 1));
            echo "<h3>Module pair: $module_pair</h3><ul>";
            echo "<ul>
                <li> <a href=ph_browser.php?vol=$vol&run=$run&analysis_dir=$analysis_dir&module_pair_dir=$mpdir&module_pair=$module_pair&num_events=$num_events&event=0>Event browser</a>
                ($num_events)
                </ul>
            ";
        }
        echo "</ul>";
    }
    page_tail();
}

$run = get_filename("run");
$vol = get_filename("vol");
$analysis_dir = get_filename("analysis");

$action = get_str('action', true);

if (!$action) {
    show_analysis($vol, $run, $analysis_dir);
} else {
    error_page("bad action $action");
}

?>
