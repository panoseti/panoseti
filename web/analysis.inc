<?php

require_once("img_pulse.inc");

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
    'ph_coinc' => 'Pulse height coincidence',
];

function show_global_analysis_types() {
    global $global_analysis_type_name;
    foreach ($global_analysis_type_name as $type=>$name) {
        echo "<p><a href=global_analysis_type.php?type=$type>$name</a>";
    }
}

function show_analysis_types($run) {
    global $analysis_type_name;
    foreach ($analysis_type_name as $type=>$name) {
        echo "<p><a href=analysis_type.php?type=$type&run=$run>$name</a>";
    }
}

function get_analyses_aux($dir) {
    if (!is_dir($dir)) return [];
    $analyses = [];
    foreach (scandir($dir) as $d) {
        $d = new stdClass;
        $d->when = $d;
        $d->who = '??";
        $d->comments = '??";
        $d->params = '??";
        $f = "$dir/$d";
        if (file_exists($f)) {
            $x = json_decode(file_get_contents($f));
            $d->who = $x->who;
            $d->comments = $x->comments;
            $d->params = $x->params;
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

function analysis_form($type, $run) {
    switch ($type) {
    case 'img_pulse':
        img_pulse_form($run);
        break;
    default:
        die("bad type $type");
    }
}

?>
