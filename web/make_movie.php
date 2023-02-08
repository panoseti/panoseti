<?php

require_once("panoseti.inc");

// make movie with given params, then redirect to it

function main($run, $fname, $min, $max, $nframes) {
    $cmd = sprintf('php ../analysis/pipe_images.php %s/%s/%s/images.bin %d %d %d | ffmpeg -y -f rawvideo -pix_fmt argb -s 128x128 -r 25 -i - -pix_fmt yuv420p -c:v libx264 -movflags +faststart -vf scale=512:512 %s/%s/%s/images_%d_%d_%d.mp4',
        ANALYSIS_ROOT, $run, $fname, $min, $max, $nframes,
        ANALYSIS_ROOT, $run, $fname, $min, $max, $nframes
    );
    system($cmd);
    Header("Location: file.php?run=$run&fname=$fname");

}

$run = post_filename('run');
$fname = post_filename('fname');
$min = (int)post_str('min');
$max = (int)post_str('max');
$nframes = (int)post_str('nframes');

if ($max==0) $max=65536;
if ($nframes == 0) $nframes=1000;

main($run, $fname, $min, $max, $nframes);
?>
