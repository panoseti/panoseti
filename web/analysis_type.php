<?php

require_once("panoseti.inc");
require_once("analysis.inc");

// show analyses of the given type for an obs run,
// and a form for doing another analysis

function show_analyses($type, $run) {
    $analyses = get_analyses($type, $run);
    if (!$analyses) {
        echo "No analyses";
        return;
    }
    start_table("table-striped");
    table_header("Date", "Who", "Comments", "Params");
    foreach ($analyses as $a) {
        table_row(
            $a->when,
            $a->who,
            $a->comments,
            $a->params
        );
    }
    end_table();
}

function main($type, $run) {
    global $analysis_type_name;
    page_head(sprintf('%s for %s', $analysis_type_name[$type], $run));
    echo "<h2>Analyses</h2>\n";
    show_analyses($type, $run);
    echo "<h2>Do analysis</h2>\n";
    analysis_form($type, $run);
    page_tail();
}

$type = get_str('type');
$run = get_str('run');

main($type, $run);

?>
