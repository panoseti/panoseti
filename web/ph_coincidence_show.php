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
    $module_pair_fname_pattern = "/module_(\d+)\.module_(\d+)/";
    foreach (scandir($dirpath) as $mpdir) {
        if (preg_match($module_pair_fname_pattern, $mpdir, $matches)) {
            $module_pair_nums = implode(',', array_slice($matches, 1));
            echo "<h3>Module pair: $module_pair_nums</h3><ul>";
            $subdir = "$dirpath/$mpdir";
            echo "<ul>
                <li> <a href=ph_browser.php?vol=$vol&run=$run&analysis_dir=$analysis_dir&module_pair_dir=$mpdir&event=0>Event browser</a>
                </ul>
            ";
        }
        echo "</ul>";
    }
    page_tail();
}

$run = get_str("run");
$vol = get_str("vol");
$analysis_dir = get_str("analysis");
check_filename($run);
check_filename($vol);
check_filename($analysis_dir);

$action = get_str('action', true);

if (!$action) {
    show_analysis($vol, $run, $analysis_dir);
} else {
    error_page("bad action $action");
}

?>
