function movie_form($run, $fname) {
    return sprintf("
        <form action=make_movie.php method=post>
        <input type=hidden name=run value=%s>
        <input type=hidden name=fname value=%s>
            min value: <input name=min size=6> max value: <input name=max size=6>
            #frames: <input name=nframes size=6>
            <input type=submit value=OK>
        <form>
        ",
        $run, $fname
    );
}

function movie_links($run, $fname) {
    $x = [];
    foreach (scandir(ANALYSIS_ROOT."/$run/$fname") as $f) {
        if (strstr($f, '.mp4')) {
            $x[] = "<a href=".ANALYSIS_ROOT."/$run/$fname/$f>$f</a>";
        }
    }
    return implode('<br>', $x);
}


    row2("Frame browser",
        "<a href=image.php?run=$run&fname=$fname&frame=0>View</a>"
    );
    row2("Movies", movie_links($run, $fname));
    row2("Make new movie", movie_form($run, $fname));
