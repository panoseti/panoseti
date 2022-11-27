<?php

DEPRECATED

require_once("panoseti.inc");

$run = get_str('run');
//$run = "obs_Lick.start_2022-08-01T03:10:30Z.runtype_eng.pffd";
$cmd = "./process_run.py $run";
page_head("Processing run");

echo "<a href=run.php?name=$run>Return to observing run page</a>.<br>
<pre>
";
system($cmd);
echo "</pre>
<a href=run.php?name=$run>Return to observing run page</a>.<br>";
page_tail();

?>
