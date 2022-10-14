<?php

// top-level page: show list of data files,
// with link to per-file pages

require_once("panoseti.inc");
require_once("analysis.inc");

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

function get_durations($run, $start_dt) {
    $rec_dur = '---';
    $collect_dur = '---';
    $cleanup_dur = '---';
    $rec_end = @file_get_contents("data/$run/recording_ended");
    if ($rec_end) {
        $rec_end_dt = local_to_dt($rec_end);
        $rec_dur = dt_diff_str($start_dt, $rec_end_dt);
        $collect_end = @file_get_contents("data/$run/collect_complete");
        if ($collect_end) {
            $collect_end_dt = local_to_dt($collect_end);
            $collect_dur = dt_diff_str($rec_end_dt, $collect_end_dt);
            $cleanup_end = @file_get_contents("data/$run/run_complete");
            if ($cleanup_end) {
                $cleanup_end_dt = local_to_dt($cleanup_end);
                $cleanup_dur = dt_diff_str($collect_end_dt, $cleanup_end_dt);
            }
        }
    }
    return [$rec_dur, $collect_dur, $cleanup_dur];
}

function main() {
    page_head("PanoSETI");
    echo "
        <h2>Hardware parameter logs (Grafana)</h2>
        <p>
        <a href=http://visigoth.ucolick.org:3000>View</a>
    ";

    echo "<h2>Multi-run analysis</h2>";
    show_global_analysis_types();

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
    table_header(
        'Start<br><small>Click to view</a>',
        'Recording time',
        'Copy time',
        'Cleanup time',
        'Run type',
        'Tags',
        'Analyses'
    );
    foreach ($runs as $run) {
        $name = $run[1];
        $n = parse_pff_name($name);
        $start = $run[0];
        $start_dt = iso_to_dt($start);
        dt_to_local($start_dt);
        $day = dt_date_str($start_dt);
        $time = dt_time_str($start_dt);
        if ($day != $prev_day) {
            row1($day, 99, 'info');
            $prev_day = $day;
        }
        [$rec_dur, $collect_dur, $cleanup_dur] = get_durations($name, $start_dt);
        table_row(
            "<a href=run.php?name=$name>$time</a>",
            $rec_dur,
            $collect_dur,
            $cleanup_dur,
            $n['runtype'],
            tags($name),
            run_analyses_str($name)
        );
    }
    end_table();
    page_tail();
}

main();

?>
