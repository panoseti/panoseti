<?php

require_once("img_pulse.inc");
require_once("visual.inc");

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
function analysis_count($run, $type) {
    $dir = "analysis/$run/$type";
    if (!is_dir($dir)) return 0;
    return count(scandir($dir))-2;
}

function show_analysis_types($run) {
    global $analysis_type_name;
    foreach ($analysis_type_name as $type=>$name) {
        $n = analysis_count($run, $type);
        echo "<p><a href=analysis_type.php?type=$type&run=$run>$name</a>";
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
        $d = new stdClass;
        $d->dir = $f;
        $d->when = $f;
        $d->username = '??';
        $d->comments = '';
        $d->params = '??';
        $path = "$dir/$f/summary.json";
        if (file_exists($path)) {
            $x = json_decode(file_get_contents($path));
            $d->username = $x->username;
            $d->params = $x->params;
        }
        $path = "$dir/$f/comments.json";
        if (file_exists($path)) {
            $x = json_decode(file_get_contents($path));
            $d->comments = $x;
        }
        $analyses[] = $d;
    }
    return $analyses;
}

function get_analyses($type, $run) {
    return get_analyses_aux(
        sprintf('%s/%s/%s', ANALYSIS_ROOT, $run, $type)
    );
}

function get_global_analyses($type) {
    return get_analyses_aux(
        sprintf('%s/%s', ANALYSIS_ROOT, $type)
    );
}

// return list of analysis types that have been done for this run
//
function run_analyses_str($run) {
    global $analysis_type_name;
    $x = [];
    foreach ($analysis_type_name as $type=>$name) {
        if (is_dir(sprintf("%s/%s/%s", ANALYSIS_ROOT, $run, $type))) {
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
    case 'visual':
        return visual_params_str($params);
    default:
        die("bad type $type");
    }
}

function analysis_form($type, $run) {
    switch ($type) {
    case 'img_pulse':
        img_pulse_form($run);
        break;
    case 'visual':
        visual_form($run);
        break;
    default:
        die("bad type $type");
    }
}

?>
