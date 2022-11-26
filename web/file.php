<?php

require_once("panoseti.inc");

function main($vol, $run, $fname) {
    page_head("PFF file $fname");

    echo "<font size=+1>";

    start_table();
    row2("Observing run", $run);
    row2("File", $fname);
    row2("Frame browser", "<a href=image.php?vol=$vol&run=$run&file=$fname&frame=0>View</a>");
    row2("Download",
        "<a href=$vol/data/$run/$fname>PFF</a> &middot; FITS"
    );
    end_table();
    page_tail();
}

$vol = get_str('vol');
$run = get_str('run');
$fname = get_str('fname');
check_filename($fname);
check_filename($run);
check_filename($vol);
main($vol, $run, $fname);

?>
