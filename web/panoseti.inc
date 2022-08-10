<?php

function is_pff($f) {
    if ($f == 'hk.pff') return false;
    return strstr($f, '.pff');
}

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

function post_str($key) {
    if (array_key_exists($key, $_POST)) {
        return $_POST[$key];
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

function row1($x, $ncols=2, $class="heading") {
    if ($class == "heading") {
        echo "<tr><th class=\"bg-primary\" colspan=\"$ncols\">$x</th></tr>\n";
    } else {
        echo "<tr><td class=\"$class\" colspan=\"$ncols\">$x</td></tr>\n";
    }
}

define('NAME_ATTRS', 'class="text-right " style="padding-right:12px"');
define('VALUE_ATTRS', 'style="padding-left:12px"');
define('VALUE_ATTRS_ERR', 'class="danger" style="padding-left:12px"');

function row2($x, $y, $show_error=false, $lwidth='40%') {
    if ($x==="") $x="<br>";
    if ($y==="") $y="<br>";
    $attrs = $show_error?VALUE_ATTRS_ERR:VALUE_ATTRS;
    echo "<tr>
        <td width=\"$lwidth\" ".NAME_ATTRS.">$x</td>
        <td $attrs >$y</td>
        </tr>
    ";
}

function row2_init($x, $y, $lwidth='40%') {
    echo '<tr>
        <td class="text-right " width="'.$lwidth.'"  style="padding-right: 20px;">'.$x.'</td>
        <td '.VALUE_ATTRS.'>'.$y.'
    ';
}

function row2_plain($x, $y) {
    echo "<tr><td>$x</td><td>$y</td></tr>\n";
}

function rowify($string) {
    echo "<tr><td>$string</td></tr>";
}

function row_array($x) {
    echo "<tr>\n";
    foreach ($x as $h) {
        echo "<td>$h</td>\n";
    }
    echo "</tr>\n";
}

define ('ALIGN_RIGHT', 'style="text-align:right;"');

function row_heading_array($x, $attrs=null, $class='bg-primary') {
    echo "<tr>";
    $i = 0;
    foreach ($x as $h) {
        $a = $attrs?$attrs[$i]:"";
        echo "<th $a class=\"$class\">$h</th>";
        $i++;
    }
    echo "</tr>\n";
}

function row_heading($x, $class='bg-primary') {
    echo sprintf('<tr><th class="%s" colspan=99>%s</th></tr>
        ', $class, $x
    );
}

function date_str($x) {
    if ($x == 0) return "---";
    return gmdate('j M Y', $x);
}

function time_str($x) {
    if ($x == 0) return "---";
    return gmdate('j M Y, G:i:s', $x) . " UTC";
}

function local_time_str($x) {
    if ($x == 0) return "---";
    return date('j M Y, H:i T', $x);
}

function parse_pff_name($name) {
    $x = [];
    foreach (explode('.', $name) as $part) {
        $p = explode('_', $part);
        if (count($p)<2) break;
        $x[$p[0]] = $p[1];
    }
    return $x;
}

?>
