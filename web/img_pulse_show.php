<?php

// show image pulse analysis output.
// output depends on GET args
// if only "dir" (analysis dir):
//      for each module
//          for each pixel and/or all:
//              link to details page
//
// detail: dir, module, pixel (-1 for all)
//      for each level L:
//          max nsigma
//          link to show top n
//          if all_L exists:
//              link to graph page
//
// top: dir, module, pixel, level
//      show textual list of pulses
//
// graph (dir, module, pixel, level)

require_once("panoseti.inc");
require_once("img_pulse.inc");

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

// get the max nsigma of pulses in file
// (file is sorted by nsigma, so it's the first one).
// Also return # of pulses in file
//
function max_nsigma($path) {
    $n = filesize($path);
    if (!$n) return [0, 0];
    $f = fopen($path, 'r');
    $s = fgets($f);
    fclose($f);
    $x = explode(' ', $s);
    return [(float)$x[OFFSET_NSIGMA], $n/strlen($s)];
}

// for each level, show:
//      - max sigma
//      - link to show top pulses
//      - if all_N exists, link to graph page
//
function show_pixel_dir($run, $analysis_dir, $module_dir, $pixel_dir) {
    $dirpath = "analysis/$run/img_pulse/$analysis_dir/$module_dir/$pixel_dir";
    page_head("Image pulse results");
    echo sprintf("Run: %s", $run);
    echo sprintf("<p>Analysis: %s", $analysis_dir);
    echo sprintf("<p>Module dir: %s", $module_dir);
    echo sprintf("<p>Pixel dir: %s", $pixel_dir);
    $spath = "analysis/$run/img_pulse/$analysis_dir/summary.json";
    $s = json_decode(file_get_contents($spath));
    $nlevels = $s->params->nlevels;
    for ($i=0; $i<$nlevels; $i++) {
        $x = 1<<$i;
        echo "<h3>Pulse duration $x</h3>\n";
        $path = "analysis/$run/img_pulse/$analysis_dir/$module_dir/$pixel_dir/thresh_$i.sorted";
        [$x, $n] = max_nsigma($path);
        echo "<ul>";
        echo "<li> Pulses with nsigma above threshhold
            <ul>
                <li> Number: $n
        ";
        if ($n) {
            echo "
                <li> Max nsigma: $x
                <li> <a href=img_pulse_show.php?action=list&run=$run&analysis=$analysis_dir&module=$module_dir&pixel=$pixel_dir&level=$i>View</a>
            ";
        }
        echo "
            </ul>
        ";
        $path = "analysis/$run/img_pulse/$analysis_dir/$module_dir/$pixel_dir/all_$i";
        if (file_exists($path)) {
            echo "<li> All pulses
                <ul>
                    <li> <a href=img_pulse_show.php?action=graph&run=$run&analysis=$analysis_dir&module=$module_dir&pixel=$pixel_dir&level=$i&start=0&n=1000>View graph</a>
                </ul>
            ";
        }
        echo "</ul>";
    }
    page_tail();
}

// show textual list of image pulses

function show_list($run, $analysis_dir, $module_dir, $pixel_dir, $level) {
    $fname = "analysis/$run/img_pulse/$analysis_dir/$module_dir/$pixel_dir/thresh_$level.sorted";
    $lines = file($fname);
    page_head("Pulses");
    start_table("table-striped");
    table_header("sample", "pulse", "mean", "stddev", "sigma", "pixel");
    foreach ($lines as $line) {
        $x = preg_split('/\s+/', trim($line));
        table_row(
            (int)$x[OFFSET_SAMPLE], (float)$x[OFFSET_VALUE], (float)$x[OFFSET_MEAN],
            (float)$x[OFFSET_STDDEV], (float)$x[OFFSET_NSIGMA], (int)$x[OFFSET_PIXEL]
        );
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
                    "img_pulse_show.php?action=detail&run=%s&analysis=%s&module=%s&pixel=%s",
                    $run, $analysis_dir, $mdir,$pdir
                );
                echo "<li><a href=$url>$pdir<br>";
            }
        }
        echo "</ul>";
    }
    page_tail();
}

function get_rec_size($f) {
    $x = fgets($f);
    return strlen($x);
}

function show_graph(
    $run, $analysis_dir, $module_dir, $pixel_dir, $level, $start, $n
) {
    $path = "analysis/$run/img_pulse/$analysis_dir/$module_dir/$pixel_dir/all_$level";
    $f = fopen($path, 'r');
    $rs = get_rec_size($f);
    fseek($f, $start*$rs);
    $f2 = fopen('ip_data.tmp', 'w');
    for ($i=0; $i<$n; $i++) {
        $s = fgets($f);
        fwrite($f2, $s);
    }
    fclose($f2);
    fclose($f);
    $s = sprintf('
        set terminal png size 1000, 1000
        plot "ip_data.tmp" using %d:%d title "value" with lines, "graph.tmp" using %d:%d title "mean" with lines
        ', OFFSET_SAMPLE+1, OFFSET_VALUE+1, OFFSET_SAMPLE+1, OFFSET_MEAN+1
    );
    $f = fopen('plot.gp', 'w');
    fwrite($f, $s);
    fclose($f);
    $cmd = 'gnuplot plot.gp > ip.png';
    system($cmd);
    page_head("Image pulse");
    echo "<img src=ip.png>";
    page_tail();
}

$run = get_str("run");
check_filename($run);
$analysis_dir = get_str("analysis");
check_filename($analysis_dir);

$action = get_str('action', true);

if (!$action) {
    show_analysis($run, $analysis_dir);
} else if ($action == 'detail') {
    $module_dir = get_str('module', true);
    $pixel_dir = get_str('pixel', true);
    show_pixel_dir($run, $analysis_dir, $module_dir, $pixel_dir);
} else if ($action == 'list') {
    $module_dir = get_str('module', true);
    $pixel_dir = get_str('pixel', true);
    $level = get_int('level');
    show_list($run, $analysis_dir, $module_dir, $pixel_dir, $level);
} else if ($action == 'graph') {
    $module_dir = get_str('module', true);
    $pixel_dir = get_str('pixel', true);
    $level = get_int('level');
    $start = get_int('start');
    $n = get_int('n');
    show_graph(
        $run, $analysis_dir, $module_dir, $pixel_dir, $level, $start, $n
    );
} else {
    error_page("bad action $action");
}

?>
