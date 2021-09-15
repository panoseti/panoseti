<?php

function page_head($title) {
    echo sprintf('
<!DOCTYPE html>
<html lang="en">
<head>
<title>%s</title>
<meta charset="utf-8">
<link type="text/css" rel="stylesheet" href="https://setiathome.berkeley.edu/panoseti/bootstrap.min.css" media="all">
<link rel=stylesheet type="text/css" href="https://setiathome.berkeley.edu/panoseti/sah_custom_dark.css">
</head>
<body >
',
    "PanoSETI: $title"
    );
    echo '<div class="container-fluid">';
    echo "<h1>$title</h1>";
}

function page_tail() {
    echo "</div></body></html>";
}

function get_int($key) {
    if (array_key_exists($key, $_GET)) {
        return (int)$_GET[$key];
    }
    return null;
}

function get_str($key) {
    if (array_key_exists($key, $_GET)) {
        return $_GET[$key];
    }
    return null;
}

// make sure a path doesn't go above current dir.
// Do this for any filename passed in a URL
//
function check_filename($name) {
    if (strstr($name, "..")) die("");
    if ($name[0] == "/") die("");
}

// various stuff copied from boinc/html/inc/util.inc


function start_table_str($class="", $style="") {
    $s = $style?'style="'.$style.'"':'';
    return '<div class="table">
      <table '.$s.' width="100%" class="table table-condensed '.$class.'" >
    ';
}

function start_table($class="", $style="") {
    echo start_table_str($class, $style);
}

function end_table_str() {
    return '</table>
        </div>
    ';
}

function end_table() {
    echo end_table_str();
}

function table_header() {
    echo "<tr>\n";
    $c = 'class="bg-primary"';
    for ($i = 0; $i < func_num_args(); $i++) {
        if (is_array(func_get_arg($i))) {
            $col = func_get_arg($i);
            echo "<th $c ".$col[1].">".$col[0]."</th>\n";
        } else {
            echo "<th $c>".func_get_arg($i)."</th>\n";
        }
    }
    echo "</tr>\n";
}

// Table row with unlimited number of columns

function table_row() {
    echo "<tr>\n";
    for ($i = 0; $i < func_num_args(); $i++) {
        if (is_array(func_get_arg($i))) {
            $col = func_get_arg($i);
            echo "<td ".$col[1].">".$col[0]."</td>\n";
        } else {
            echo "<td>".func_get_arg($i)."</td>\n";
        }
    }
    echo "</tr>\n";
}
?>
