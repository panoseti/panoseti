<?php

// remove derived files for a run

require_once("panoseti.inc");

$run = get_str('name');
check_filename($run);
$path = "derived/$run";
if (!file_exists($path)) {
    die("no such dir");
}
$cmd = "rm -rf $path";
page_head("Removed derived files for $run");
system($cmd);
echo "<a href=run.php?name=$run>Return to run page</a>";
page_tail();

?>
