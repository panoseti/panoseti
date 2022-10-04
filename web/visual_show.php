<?php

// show visual analysis
// if only "dir" (analysis dir)
//      show params
//      for each module
//          link to movie
//          link to frame browser

require_once("panoseti.inc");
require_once("analysis.inc");

function show_analysis($run, $analysis_dir) {
    $dirpath = "analysis/$run/visual/$analysis_dir";
    page_head("Visual analysis");
    analysis_page_intro($analysis_dir, $dirpath);
    foreach (scandir($dirpath) as $mdir) {
        if (substr($mdir, 0, 7) != 'module_') continue;
        $mnum = substr($mdir, 7);
        echo "<h3>Module $mnum</h3>";
        echo "<ul>
            <li> <a href=image.php?run=$run&analysis_dir=$analysis_dir&module_dir=$mdir&frame=0>Frame browser</a>
            <li> <a href=analysis/$run/visual/$analysis_dir/$mdir/images.mp4>Movie</a>
            </ul>
        ";
    }
    page_tail();
}

$run = get_str("run");
check_filename($run);
$analysis_dir = get_str("analysis");
check_filename($analysis_dir);
show_analysis($run, $analysis_dir);

?>
