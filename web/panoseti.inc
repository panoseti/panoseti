<?php

ini_set('display_errors', 1);


// ------------ login and password --------------

define('LOGIN_MANDATORY', 1);
define('LOGIN_OPTIONAL',  2);
define('LOGIN_NONE',      3);

$login_name = null;

function hash_passwd($p) {
    return md5($p."blahblah");
}

function check_passwd_hash($name, $passwd_hash) {
    $passwds = json_decode(file_get_contents('passwd.json'));
    if (!$passwds) {
        die ("can't parse passwd file");
    }
    if (!isset($passwds->$name)) {
        return null;
    }
    $u = $passwds->$name;
    if ($u->passwd_hash != $passwd_hash) {
        return null;
    }
    return $u;
}

function check_auth($name, $auth) {
    $passwds = json_decode(file_get_contents('passwd.json'));
    if (!isset($passwds->$name)) return null;
    $u = $passwds->$name;
    if ($u->auth != $auth) return null;
    return $u;
}

function get_login($optional=false) {
    $auth = isset($_COOKIE['auth']) ? $_COOKIE['auth'] : null;
    $name = isset($_COOKIE['name']) ? $_COOKIE['name'] : null;
    if (!$auth || !$name || !check_auth($name, $auth)) {
        if ($optional) return null;
        Header("Location: login.php");
        exit;
    }
    return $name;
}

function page_head($title, $login=LOGIN_MANDATORY, $is_front=false) {
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

    if ($login == LOGIN_NONE) {
        return;
    }
    global $login_name;
    $login_name = get_login(true);
    if ($login == LOGIN_MANDATORY && !$login_name) {
        echo "You must be logged in to view this page.
            <a href=login.php>Log in</a>.
        ";
        page_tail();
        exit;
    }
    if ($login_name) {
        $x = "Logged in as $login_name.  <a href=logout.php>Log out</a>.";
    } else {
        $x = "<a href=login.php>Log in</a>";
    }
    echo sprintf(
        '<p><table width=100%%><tr><td><font size=+3>%s</font></td><td align=right>%s</a></td></tr></table>',
        $is_front?"PanoSETI home page":'<a href=index.php>Return to home page</a>',
        $x
    );
    echo "<hr>";
    if (!$is_front) {
        echo "<h1>$title</h1>";
    }
}

function login_name() {
    global $login_name;
    $login_name = get_login();
    return $login_name;
}

function page_tail() {
    echo "</div></body></html>";
}

function error_page($msg) {
    page_head("Unable to handle request", LOGIN_NONE);
    echo $msg;
    page_tail();
    exit();
}

// ---------------- form stuff ------------

function get_int($name, $optional=false) {
    $x=null;
    if (isset($_GET[$name])) $x = $_GET[$name];
    if (!is_numeric($x)) {
        if ($optional) {
            if ($x) {
                error_page("bad value for $name");
            }
            return null;
        } else {
            error_page("missing $name");
        }
    }
    return (int)$x;
}

function get_str($key, $optional=false) {
    if (array_key_exists($key, $_GET)) {
        return $_GET[$key];
    }
    if ($optional) return null;
    error_page("missing arg: $key");
}

function post_str($key, $optional=false) {
    if (array_key_exists($key, $_POST)) {
        return $_POST[$key];
    }
    if ($optional) return null;
    error_page("missing arg: $key");
}

// Do this for any file or dir name passed in a URL
//
function check_filename($name) {
    if (preg_match('/[^a-z0-9\_\.\-\:]/i', $name)) {
        error_page("bad file/dir name: $name");
    }
}

// get a file or dir name from GET
//
function get_filename($key) {
    $x = get_str($key);
    check_filename($x);
    return $x;
}

function post_filename($key) {
    $x = post_str($key);
    check_filename($x);
    return $x;
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

// use the following around text with long lines,
// to limit the width and make it more readable.
//
function text_start($width=640) {
    echo sprintf("<div style=\"max-width: %dpx;\">\n", $width);
}
function text_end() {
    echo "</div>\n";
}

// -------------  date/time stuff --------------

// Unix time to UTC date string
//
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

// ISO format string to DateTime
//
function iso_to_dt($x) {
    return DateTime::createFromFormat('Y-m-d\TH:i:s\Z', $x,
        new DateTimeZone("UTC")
    );
}

// local format string to DateTime
//
function local_to_dt($x) {
    return DateTime::createFromFormat('Y-m-d\TH:i:s', $x);
}

// convert DateTime from UTC to local timezone
//
function dt_to_local($dt) {
    $dt->setTimezone(new DateTimeZone("America/Los_Angeles"));
}

// difference of DateTimes as string
//
function dt_diff_str($dt1, $dt2) {
     $diff = $dt1->diff($dt2);
     return $diff->format('%H:%I:%S');
}

// DateTime to string
//
function dt_date_str($dt) {
    return $dt->format('d M Y');
}

function dt_time_str($dt) {
    return $dt->format('H:i:s');
}

// -------------- PFF ----------------

function is_pff($f) {
    if ($f == 'hk.pff') return false;
    return strstr($f, '.pff');
}

function parse_pff_name($name) {
    $x = explode('.', $name);
    $p = [];
    for ($i=0; $i<count($x)-1; $i++) {
        $y = $x[$i];
        $z = explode('_', $y);
        if (count($z) != 2) continue;
        $p[$z[0]] = $z[1];
    }
    return $p;
}

// return list of data products in run
//
function run_data_products($vol, $name) {
    $s = [];
    $dc = json_decode(@file_get_contents("$vol/data/$name/data_config.json"));
    if (!$dc) {
        $s[] = '---';
        return $s;
    }
    if (!empty($dc->image)) {
        $x = $dc->image;
        $s[] = sprintf("Image, %d bit, %d usec",
            $x->quabo_sample_size, $x->integration_time_usec
        );
    }
    if (array_key_exists('pulse_height', $dc)) {
        $x = $dc->pulse_height;
        $s[] = sprintf("Pulse height, threshold %d", $x->pe_threshold);
    }
    return $s;
}

// return list of modules in run
//
function run_modules($vol, $name) {
    $s = [];
    $oc = json_decode(@file_get_contents("$vol/data/$name/obs_config.json"));
    if (!$oc) {
        $s[] = '---';
        return $s;
    }
    foreach ($oc->domes as $dome) {
        foreach ($dome->modules as $module) {
            $s[] = module_str($module, $dome);
        }
    }
    return $s; 
}

function module_str($module, $dome) {
    $x = ip_addr_to_module_num($module->ip_addr);
    if (!empty($dome->name)) {
        if (!empty($module->name)) {
            $x .= " ($dome->name, $module->name)";
        } else {
            $x .= " ($dome->name)";
        }
    }
    return $x;
}

function ip_addr_to_module_num($ip_addr) {
    $x = explode('.', $ip_addr);
    $n = (int)$x[3] + 256*(int)$x[2];
    return ($n>>2)&255;
}

// return length of header (JSON + 1 for the *)
//
function header_size($f) {
    $n = 0;
    while (1) {
        $s = fgets($f);
        if (!$s) break;
        $n += strlen($s);
        if ($s == "\n") break;
    }
    return $n+1;
}

?>
