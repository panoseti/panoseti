<?php

// show a coincident pulse height event image,
// with buttons for moving forward or back in time

require_once("panoseti.inc");
require_once("analysis.inc");

function get_num_events($vol, $run, $analysis_dir, $module_pair_dir) {
    $dirpath = "$vol/analysis/$run/ph_coincidence/$analysis_dir/$module_pair_dir";
    $event_pattern = "/^event_\d+.*/";
    $n = 0;
    foreach(scandir($dirpath) as $event) {
        if (preg_match($event_pattern, $event)) {
            $n += 1;
    }
    return $n;
}

function arrows_str($vol, $run, $analysis_dir, $module_pair_dir, $event, $num_events) {
    $url = "ph_browser.php?vol=$vol&run=$run&analysis_dir=$analysis_dir&module_pair_dir=$module_pair_dir&event=";
    return sprintf(
        <a class="btn btn-sm btn-primary" href=%s%d><< event</a>
        <a class="btn btn-sm btn-primary" href=%s%d> event >></a>
        $url, ($event - 1) % $num_events,
        $url, ($event + 1) % $num_events,
    );
}

function show_event($event_path, $arrows) {
    echo "<table>";
    echo "<tr><img src=$event_path></tr>\n";
    echo "<tr><td colspan=32 align=center><br>$arrows</td></tr>\n";
    echo "</table>";
}


function main($vol, $run, $analysis_dir, $module_pair_dir, $event) {
page_head("Pulse-Height Coincidence");
    echo "<p>Run: <a href=run.php?vol=$vol&name=$run>$run</a>\n";
    echo "<p>Modules: $module_pair_dir\n";
    $num_events = get_num_events($vol, $run, $analysis_dir, $module_pair_dir);
    echo "<p>Event: $event / $num_events\n";
    $event_path = "$vol/analysis/$run/ph_coincidence/$analysis_dir/$module_pair_dir/event_$event.png";
    $as = arrows_str($vol, $run, $analysis_dir, $module_pair_dir, $event, $num_events);
    show_event($event_path, $as);
    page_tail();
}

$run = get_str("run");
$vol = get_str("vol");
check_filename($run);
check_filename($vol);
$analysis_dir = get_str("analysis_dir");
check_filename($analysis_dir);
$module_pair_dir = get_str("module_pair_dir");
check_filename($module_event_dir);
$event = get_int("event");

main($vol, $run, $analysis_dir, $module_pair_dir, $event);

?>
