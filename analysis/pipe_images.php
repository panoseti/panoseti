<?php

// pipe_images.php dir min max nframes
// input: dir/images.bin
// output: a stream of images in argb format,
//      with each pixel expanded to a 4x4 block.
//      Scale data so min->0, max->255
//      This is piped into ffmpeg

ini_set('display_errors', 1);

// output RGB frames (for ffmpeg)
// $frame is 1024 16-bit numbers
// output 1024 4-byte argb, 4x4 blocks
//

function output_frame($frame, $min, $max, $bytes_per_pixel) {
    $s = 4;
    $x = [];
    if ($max == 0) {
        // no scaling
        for ($i=0; $i<32; $i++) {
            for ($ii=0; $ii<$s; $ii++) {
                for ($j=0; $j<32; $j++) {
                    if ($bytes_per_pixel == 2) {
                        $v = $frame[$i*32+$j] >> 8;
                    } else {
                        $v = $frame[$i*32+$j];
                    }
                    for ($jj=0; $jj<$s; $jj++) {
                        $x[] = 255;     // alpha
                        $x[] = $v;
                        $x[] = $v;
                        $x[] = $v;
                    }
                }
            }
        }
    } else {
        $d = $max-$min;
        for ($i=0; $i<32; $i++) {
            for ($ii=0; $ii<$s; $ii++) {
                for ($j=0; $j<32; $j++) {
                    $v = (int)(($frame[$i*32+$j]-$min)*256./$d);
                    if ($v<0) $v = 0;
                    if ($v>255) $v = 255;
                    for ($jj=0; $jj<$s; $jj++) {
                        $x[] = 255;     // alpha
                        $x[] = $v;
                        $x[] = $v;
                        $x[] = $v;
                    }
                }
            }
        }
    }

    $n = 1024*4*$s*$s;
    echo pack("C$n", ...$x);
}

function output_frames($file, $min, $max, $nframes, $bytes_per_pixel) {
    $f = fopen($file, "rb");
    for ($i=0; $i<$nframes; $i++) {
        if (feof($f)) break;
        $x = fread($f, 1024*$bytes_per_pixel);
        if (strlen($x)==0 ) break;
        if ($bytes_per_pixel == 2) {
            $y = array_merge(unpack("S1024", $x));
        } else {
            $y = array_merge(unpack("C1024", $x));
        }
        // unpack returns a 1-offset array - huh?????
        // array_merge() makes it 0-offset
            
        if (!$y) {
            die("unpack");
        }
        output_frame($y, $min, $max, $bytes_per_pixel);
    }
}

$fname = $argv[1];
$min = (int)$argv[2];
$max = (int)$argv[3];
$nframes = (int)$argv[4];
$bytes_per_pixel = (int)$argv[5];
output_frames($fname, $min, $max, $nframes, $bytes_per_pixel);

?>
