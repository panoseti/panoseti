<?php

function pulse_file_name($file, $module, $pixel, $type, $dur) {
    $x = "pulse_out/$file/$module/$pixel/$type";
    switch($type) {
    case "value":
    case "value_hist":
        break;
    default:
        $x .= "_$dur";
    }
    return $x;
}

function pulse_title($type) {
    if ($type == 'value') return "Pixel values";
    if ($type == 'value_hist') return "Pixel value histogram";
    if ($type == 'all') return "Intensity";
    if ($type == 'mean') return "Local mean intensity";
    if ($type == 'stddev') return "Local stddev";
    if ($type == 'thresh') return "Pulses above threshold ";
    return "bad type";
}

?>
