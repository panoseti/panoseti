<?php

// show a coincident pulse height event image,
// with buttons for moving forward or back in time

require_once("panoseti.inc");
require_once("analysis.inc");


function truemod($num, $mod) {
  return ($mod + ($num % $mod)) % $mod;
}

function arrows_str($vol, $run, $analysis_dir, $module_pair_dir, $module_pair, $num_events, $event) {
    $url = "ph_browser.php?vol=$vol&run=$run&analysis_dir=$analysis_dir&module_pair_dir=$module_pair_dir&module_pair=$module_pair&num_events=$num_events&event=";
    if ($num_events > 0) {
        return sprintf(
            '<a class="btn btn-sm btn-primary" href=%s%d><< 100 </a>
            <a class="btn btn-sm btn-primary" href=%s%d><< 10 </a>
            <a class="btn btn-sm btn-primary" href=%s%d><< 5 </a>
            <a class="btn btn-sm btn-primary" href=%s%d><< 1 </a>
            <a class="btn btn-sm btn-primary" href=%s%d> 1 >></a>
            <a class="btn btn-sm btn-primary" href=%s%d> 5 >></a>
            <a class="btn btn-sm btn-primary" href=%s%d> 10 >></a>
            <a class="btn btn-sm btn-primary" href=%s%d> 100 >></a>',
            $url, truemod($event - 100, $num_events),
            $url, truemod($event - 10, $num_events),
            $url, truemod($event - 5, $num_events),
            $url, truemod($event - 1, $num_events),
            $url, truemod($event + 1, $num_events),
            $url, truemod($event + 5, $num_events),
            $url, truemod($event + 10, $num_events),
            $url, truemod($event + 100, $num_events)
        );
    } else {
        return "No events";
    }
}

function show_event($event_path, $arrows) {
    echo "<table>";
    echo "<tr><br><img src=$event_path width=700 height=500></tr>\n";
    echo "<tr><td colspan=32 align=center><br>$arrows</td></tr>\n";
    echo "</table>";
}


function main($vol, $run, $analysis_dir, $module_pair_dir, $module_pair, $num_events, $event) {
    page_head("Pulse-height coincidence");
    echo "<p>Run: <a href=run.php?vol=$vol&name=$run>$run</a>\n";
    echo "<p>Module pair: $module_pair</p>";
    if ($num_events > 0) {
        echo sprintf(
            '<p>Event: %d / %d',
            $event + 1, $num_events
        );
        $event_path = "$vol/analysis/$run/ph_coincidence/$analysis_dir/$module_pair_dir/event_$event.png";
        $as = arrows_str($vol, $run, $analysis_dir, $module_pair_dir, $module_pair, $num_events, $event);
        show_event($event_path, $as);
    } else {
        echo "No events";
    }
    page_tail();
}


$run = get_str("run");
$vol = get_str("vol");
check_filename($run);
check_filename($vol);
$analysis_dir = get_str("analysis_dir");
check_filename($analysis_dir);
$module_pair_dir = get_str("module_pair_dir");
check_filename($module_pair_dir);
$module_pair = get_str("module_pair");
$num_events = get_int("num_events");
$event = get_int("event");

main($vol, $run, $analysis_dir, $module_pair_dir, $module_pair, $num_events, $event);

?>
