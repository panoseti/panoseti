<?php

// remove derived files for a run

require_once("panoseti.inc");

$run = get_filename('name');
$path = ANALYSIS_ROOT."/$run";
if (!file_exists($path)) {
    die("no such dir");
}
$cmd = "rm -rf $path";
page_head("Removed analysis files for $run");
system($cmd);
echo "<a href=run.php?name=$run>Return to run page</a>";
page_tail();

?>
