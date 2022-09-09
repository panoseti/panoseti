<?php

// show image pulse analysis output.
// output depends on GET args
// if only "dir":
//      for each module
//          for each pixel or all:
//              show link to details page
//
// if dir, module, pixel (-1 for all)
//      show list of links per file
//
// if show_list, dir, module, pixel, level, nsigma
//      show textual list of pulses above thresh

require_once("panoseti.inc");
require_once("img_pulse.inc");

function show_file($run, $analysis_dir, $module_dir, $pixel_dir, $type, $dur) {
    $title = img_pulse_title($type);
    $path = img_pulse_file_path($run, $analysis_dir, $module_dir, $pixel_dir, $type, $dur);
    $size = 0;
    if (file_exists($path)) {
        $size = filesize($path);
    }
    if ($size == 0) {
        echo "<li>$title: none\n";
        return;
    }
    $url = sprintf('graph.php?run=%s&analysis=%s&module=%s&pixel=%s&type=%s&dur=%d',
        $run, $analysis_dir, $module_dir, $pixel_dir, $type, $dur
    );
    echo "<li><a href=$url>$title</a> ($size bytes)";
}

function sigma_form($run, $analysis_dir, $module_dir, $pixel_dir, $dur) {
    echo "<li><form action=pulse_sigma.php>Pulses above
        <input type=text name=nsigma size=4> sigma
        <input type=hidden name=run value=$run>
        <input type=hidden name=analysis value=$analysis_dir>
        <input type=hidden name=module value=$module_dir>
        <input type=hidden name=pixel value=$pixel_dir>
        <input type=hidden name=dur value=$dur>
        <input type=submit class=\"btn-sm btn-primary\" name=list value=list>
        </form>
    ";
}

function show_pixel_dir($run, $analysis_dir, $module_dir, $pixel_dir) {
    $dirpath = "analysis/$run/img_pulse/$analysis_dir/$module_dir/$pixel_dir";
    page_head("Image pulse results");
    echo sprintf("Run: %s", $run);
    echo sprintf("Analysis: %s", $analysis_dir);
    echo sprintf("Module dir: %s", $module_dir);
    echo sprintf("Pixel dir: %s", $pixel_dir);
    //show_file($file, $pixel, "value_hist", 0);
    for ($i=0; $i<16; $i++) {
        $x = 1<<$i;
        echo "<h3>Pulse duration $x</h3>\n";
        echo "<ul>";
        show_file($run, $analysis_dir, $module_dir, $pixel_dir, "all", $i);
        show_file($run, $analysis_dir, $module_dir, $pixel_dir, "mean", $i);
        show_file($run, $analysis_dir, $module_dir, $pixel_dir, "stddev", $i);
        show_file($run, $analysis_dir, $module_dir, $pixel_dir, "thresh", $i);
        sigma_form($run, $analysis_dir, $module_dir, $pixel_dir, $i);
        echo "</ul>";
    }
    page_tail();
}

// show textual list of image pulses above a given threshold

function show_list($file, $pixel, $dur, $nsigma) {
    page_head("Pulses above $nsigma sigma");
    $fname = pulse_file_name($file, $pixel, 'all', $dur);
    $lines = file($fname);
    start_table("table-striped");
    table_header("time (sec)", "mean intensity", "sigma");
    foreach ($lines as $line) {
        $x = explode(',', $line);
        if (count($x) < 3) continue;
        $s = (double)$x[2];
        if ($s > $nsigma) {
            table_row($x[0], $x[1], $s);
        }
    }
    end_table();
    page_tail();
}

function show_analysis($run, $analysis_dir) {
    $dirpath = "analysis/$run/img_pulse/$analysis_dir";
    page_head("Image pulse analysis");
    foreach (scandir($dirpath) as $mdir) {
        if (substr($mdir, 0, 7) != 'module_') continue;
        echo "<h3>$mdir</h3><ul>";
        $subdir = "$dirpath/$mdir";
        foreach (scandir($subdir) as $pdir) {
            if ($pdir == 'all_pixels' || substr($pdir, 0, 6) == 'pixel_') {
                $url = sprintf(
                    "img_pulse_show.php?run=%s&analysis=%s&module=%s&pixel=%s",
                    $run, $analysis_dir, $mdir,$pdir
                );
                echo "<li><a href=$url>$pdir<br>";
            }
        }
        echo "</ul>";
    }
    page_tail();
}

$run = get_str("run");
check_filename($run);
$analysis_dir = get_str("analysis");
check_filename($analysis_dir);

$module_dir = get_str('module', true);
$pixel_dir = get_str('pixel', true);
if ($pixel_dir) {
    show_pixel_dir($run, $analysis_dir, $module_dir, $pixel_dir);
} else {
    show_analysis($run, $analysis_dir);
}
exit;

$file = get_str("file");
$pixel = get_int("pixel");
$dur = get_int("dur");
$nsigma = (double)get_str("nsigma");

//show_list($file, $pixel, $dur, $nsigma);


$file = get_str("file");
$pixel = get_int("pixel");
//main($file, $pixel);

?>
