<?php

// top-level page: show list of data files,
// with link to per-file pages

require_once("panoseti.inc");
require_once("analysis.inc");

function compare($x, $y) {
    return $x[0] < $y[0];
}

function tags($vol, $run) {
    $t = @file_get_contents("$vol/data/$run/tags.json");
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

function get_durations($vol, $run, $start_dt) {
    $rec_dur = '---';
    $collect_dur = '---';
    $cleanup_dur = '---';
    $rec_end = @file_get_contents("$vol/data/$run/recording_ended");
    if ($rec_end) {
        $rec_end_dt = local_to_dt($rec_end);
        $rec_dur = dt_diff_str($start_dt, $rec_end_dt);
        $collect_end = @file_get_contents("$vol/data/$run/collect_complete");
        if ($collect_end) {
            $collect_end_dt = local_to_dt($collect_end);
            $collect_dur = dt_diff_str($rec_end_dt, $collect_end_dt);
            $cleanup_end = @file_get_contents("$vol/data/$run/run_complete");
            if ($cleanup_end) {
                $cleanup_end_dt = local_to_dt($cleanup_end);
                $cleanup_dur = dt_diff_str($collect_end_dt, $cleanup_end_dt);
            }
        }
    }
    return [$rec_dur, $collect_dur, $cleanup_dur];
}

function get_pointing_info($vol, $run) {
    // Returns the elevation of the Barnard dome module (Module 3) for prototype obs planning.
    // NOTE: must generalize this function for full system.
    $file_path = "$vol/data/$run/obs_config.json";
    if (!file_exists($file_path)) return "---";
    $obs_config = json_decode(file_get_contents($file_path));
    // Check if we observed with the Astrograph and Barnard domes
    if (sizeof($obs_config->domes) != 2) return "---";
    if (!strcmp($obs_config->domes[0]->name, "barnard")) {
        $elevation = $obs_config->domes[0]->modules[0]->elevation;
    } else if (!strcmp($obs_config->domes[1]->name, "barnard")) {
        $elevation = $obs_config->domes[1]->modules[0]->elevation;
    } else {
        return "---";
    }
    return $elevation . "&deg";
}

function main() {
    page_head("PanoSETI", LOGIN_MANDATORY, true);
    echo "
        <h2>Hardware parameter logs (Grafana)</h2>
        <p>
        <a href=http://visigoth.ucolick.org:3000>View</a>
        <br
        <small>(User: admin; password: visigoth password)</small>
    ";

    echo "<h2>Multi-run analysis</h2>";
    show_global_analysis_types();

    echo "
        <h2>Observing runs</h2>
        <p>
    ";
    $vols = json_decode(file_get_contents('head_node_volumes.json'));
    $runs = [];
    foreach ($vols as $vol) {
        foreach (scandir("$vol/data") as $f) {
            if (!strstr($f, '.pffd')) continue;
            $n = parse_pff_name($f);
            $birdie_seq = -1;
            if (array_key_exists('birdie', $n)) {
                $birdie_seq = $n['birdie'];
            }
            $runs[] = [$n['start'], $f, $vol, $birdie_seq];
        }
    }
    usort($runs, 'compare');
    $prev_day =  '';
    start_table('table-striped');
    table_header(
        'Start<br><small>Click to view</a>',
        'Recording time',
        'Copy time',
        'Cleanup time',
        'Modules',
        'Elevation (Module 3)',
        'Data',
        'Run type',
        'Volume',
        'Tags',
        'Analyses'
    );
    foreach ($runs as $run) {
        $name = $run[1];
        $vol = $run[2];
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
        [$rec_dur, $collect_dur, $cleanup_dur] = get_durations($vol, $name, $start_dt);

        $birdie_seq = $run[3];
        $b = '';
        if ($birdie_seq >= 0) {
            $b = "<br>(birdie seq $birdie_seq)";
        }
        table_row(
            "<a href=run.php?vol=$vol&name=$name>$time</a>$b",
            $rec_dur,
            $collect_dur,
            $cleanup_dur,
            implode('<br>', run_modules($vol, $name)),
            get_pointing_info($vol, $name),
            implode('<br>', run_data_products($vol, $name)),
            $n['runtype'],
            $vol,
            tags($vol, $name),
            run_analyses_str($vol, $name)
        );
    }
    end_table();
    page_tail();
}

main();

?>
