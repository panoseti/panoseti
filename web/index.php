<?php

ini_set('display_errors', 1);

// top-level page: show list of data files,
// with link to per-file pages

require_once("panoseti.inc");

function compare($x, $y) {
    return $x[0] < $y[0];
}

function tags($run) {
    $t = @file_get_contents("data/$run/tags.json");
    if ($t) {
        $tags = json_decode($t);
    } else {
        $tags = [];
    }
    if (!count($tags)) return "---";
    $x = '';
    foreach ($tags as $tag) {
        if ($x) {
            $x .= '<br>';
        }
        $x .= $tag->who.': '.$tag->tag;
    }
    return $x;
}

function main() {
    page_head("PanoSETI");
    echo "
        <h2>Graphical parameter logs</h2>
        <p>
        <a href=http://visigoth.ucolick.org:3000>View</a>
    ";

    echo "
        <h2>Observing runs</h2>
        <p>
    ";
    $runs = [];
    foreach (scandir("data") as $f) {
        if (!strstr($f, '.pffd')) continue;
        $n = parse_pff_name($f);
        $runs[] = [$n['start'], $f];
    }
    usort($runs, 'compare');
    $prev_day =  '';
    start_table('table_striped');
    table_header('Start', 'Run type', 'Tags', 'Click to view');
    foreach ($runs as $run) {
        $name = $run[1];
        $n = parse_pff_name($name);
        $start = $run[0];
        $s = explode('T', $start);
        $day = $s[0];
        $time = $s[1];
        if ($day != $prev_day) {
            row1($day, 99, 'info');
            $prev_day = $day;
        }
        table_row(
            $time, $n['runtype'],
            tags($name),
            "<a href=run.php?name=$name>View</a>"
        );
    }
    end_table();
    page_tail();
}

main();

?>
