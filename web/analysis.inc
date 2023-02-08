<?php

require_once("img_pulse.inc");
require_once("visual.inc");
require_once("ph_coincidence.inc");

define('ANALYSIS_ROOT', 'analysis');

// multi-obs-run analyses
//
$global_analysis_type_name = [
    'repeated_pulse' => 'Repeated pulses',
];

// single-run analyses
//
$analysis_type_name = [
    'visual' => 'Visualizations',
    'stats' => 'Statistics',
    'img_pulse' => 'Image pulses',
    'ph_coincidence' => 'Pulse height coincidence',
];

function show_global_analysis_types() {
    global $global_analysis_type_name;
    foreach ($global_analysis_type_name as $type=>$name) {
        echo "<p><a href=global_analysis_type.php?type=$type>$name</a>";
    }
}

// return the number of analyses of the given type
//
function analysis_count($vol, $run, $type) {
    $dir = "$vol/analysis/$run/$type";
    if (!is_dir($dir)) return 0;
    return count(scandir($dir))-2;
}

function show_analysis_types($vol, $run) {
    global $analysis_type_name;
    foreach ($analysis_type_name as $type=>$name) {
        $n = analysis_count($vol, $run, $type);
        echo "<p><a href=analysis_type.php?type=$type&vol=$vol&run=$run>$name</a>";
        if ($n) {
            echo " ($n)";
        }
    }
}

// get list of analyses in a given dir,
// which is either for a run/type (single run)
// or type (multi-run)
//
function get_analyses_aux($dir) {
    if (!is_dir($dir)) return [];
    $analyses = [];
    foreach (scandir($dir) as $f) {
        if (substr($f, 0, 1) == '.') continue;
        $path = "$dir/$f/summary.json";
        if (!file_exists($path)) continue;

        $d = new stdClass;
        $d->dir = $f;
        $d->when = $f;
        $x = json_decode(file_get_contents($path));
        $d->username = $x->username;
        $d->params = $x->params;

        $path = "$dir/$f/comments.json";
        if (file_exists($path)) {
            $x = json_decode(file_get_contents($path));
            $d->comments = $x;
        } else {
            $d->comments = '';
        }
        $analyses[] = $d;
    }
    return $analyses;
}

function get_analyses($type, $vol, $run) {
    return get_analyses_aux(
        sprintf('%s/analysis/%s/%s', $vol, $run, $type)
    );
}

function get_global_analyses($type) {
    return get_analyses_aux(
        sprintf('analysis/%s', $type)
    );
}

// return list of analysis types that have been done for this run
//
function run_analyses_str($vol, $run) {
    global $analysis_type_name;
    $x = [];
    foreach ($analysis_type_name as $type=>$name) {
        if (is_dir(sprintf("%s/analysis/%s/%s", $vol, $run, $type))) {
            $x[] = $name;
        }
    }
    return implode(', ', $x);
}

// return a string summarizing a set of analysis params
//
function params_str($type, $params) {
    switch ($type) {
    case 'img_pulse':
        return img_pulse_params_str($params);
    case 'ph_coincidence':
        return ph_coincidence_params_str($params);
    case 'visual':
        return visual_params_str($params);
    default:
        die("bad type $type");
    }
}

function analysis_form($type, $vol, $run) {
    switch ($type) {
    case 'img_pulse':
        img_pulse_form($vol, $run);
        break;
    case 'ph_coincidence':
        ph_coincidence_form($vol, $run);
        break;
    case 'visual':
        visual_form($vol, $run);
        break;
    default:
        die("bad type $type");
    }
}

// type-independent info for analysis pages.
// Put this at top of page
//
function analysis_page_intro($type, $analysis_dir, $dirpath) {
    text_start();
    start_table('table-striped');
    row2("Analysis time", $analysis_dir);
    $summary_path = "$dirpath/summary.json";
    $s = json_decode(file_get_contents($summary_path));
    row2("Run by", $s->username);
    row2("Parameters", params_str($type, $s->params));
    row2("Analysis directory", "<a href=$dirpath>View</a>");
    end_table();
    text_end();
}

?>
