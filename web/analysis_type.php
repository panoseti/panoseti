<?php

require_once("panoseti.inc");
require_once("analysis.inc");

// show analyses of the given type for an obs run,
// and a form for doing another analysis

function show_analyses($type, $vol, $run) {
    $analyses = get_analyses($type, $vol, $run);
    if (!$analyses) {
        echo "No analyses";
        return;
    }
    start_table("table-striped");
    table_header(
        "Date<br><small>Click for details</small>",
        "Who",
        "Comments",
        "Params"
    );
    foreach ($analyses as $a) {
        table_row(
            sprintf(
                "<a href=%s_show.php?vol=%s&run=%s&analysis=%s>%s</a>",
                $type, $vol, $run, $a->dir, $a->when
            ),
            $a->username,
            $a->comments,
            params_str($type, $a->params)
        );
    }
    end_table();
}

function main($type, $vol, $run) {
    global $analysis_type_name;
    page_head(sprintf('%s for %s', $analysis_type_name[$type], $run));
    echo "<h2>Analyses</h2>\n";
    show_analyses($type, $vol, $run);
    echo "<h2>Do analysis</h2>\n";
    analysis_form($type, $vol, $run);
    page_tail();
}

$type = get_str('type');
$run = get_filename('run');
$vol = get_filename('vol');

main($type, $vol, $run);

?>
