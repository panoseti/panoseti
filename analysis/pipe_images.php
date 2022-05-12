<?php

// pipe_images.php dir nframes
// input: dir/images.bin
// output: a stream of images in argb format,
//      with each pixel expanded to a 4x4 block
//      This is piped into ffmpeg

ini_set('display_errors', 1);

// output RGB frames (for ffmpeg)
// $frame is 1024 16-bit numbers
// output 1024 4-byte argb, 4x4 blocks
//
function output_frame($frame) {
    $s = 4;
    $x = [];
    for ($i=0; $i<32; $i++) {
        for ($ii=0; $ii<$s; $ii++) {
            for ($j=0; $j<32; $j++) {
                $v = $frame[$i*32+$j] >> 8;
                for ($jj=0; $jj<$s; $jj++) {
                    $x[] = 255;
                    $x[] = $v;
                    $x[] = $v;
                    $x[] = $v;
                }
            }
        }
    }
    $n = 1024*4*$s*$s;
    echo pack("C$n", ...$x);
}

function output_frames($file, $nframes) {
    $f = fopen($file, "rb");
    for ($i=0; $i<$nframes; $i++) {
        if (feof($f)) break;
        $x = fread($f, 1024*2);
        if (strlen($x)==0 ) break;
        $y = array_merge(unpack("S1024", $x));
            // unpack makes 1-offset array - ?????
        if (!$y) {
            die("unpack");
        }
        output_frame($y);
    }
}

$fname = $argv[1];
$nframes = (int)$argv[2];
output_frames("$fname", $nframes);

?>
