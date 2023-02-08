<?php

// main page for an observing run.
// show links to the component files,
// and show comments and tags

require_once("panoseti.inc");
require_once("analysis.inc");

function add_tag($vol, $run) {
    $t = @file_get_contents("$vol/data/$run/tags.json");
    if ($t) {
        $tags = json_decode($t);
    } else {
        $tags = [];
    }
    $tag = new stdClass;
    $tag->who = login_name();
    $tag->when = time();
    $tag->tag = post_str('tag');
    if (!$tag->who) {
        die("must be logged in");
    }
    if (!$tag->tag) {
        die("must give a tag");
    }
    $tags[] = $tag;
    file_put_contents("$vol/data/$run/tags.json", json_encode($tags));
    header("Location: run.php?vol=$vol&name=$run");
}

function tags_form($vol, $run) {
    if (!login_name()) {
        echo "<a href=login.php>Log in</a> to add a tag.";
        return;
    }
    echo "
        <p>
        Add tag:
        <form method=post action=run.php>
        <input type=hidden name=name value=$run>
        <input type=hidden name=vol value=$vol>
        <input name=tag>
        <p><p>
        <input type=submit name=add_tag value=OK>
        </form>
    ";
}

function show_tags($vol, $run) {
    $t = @file_get_contents("$vol/data/$run/tags.json");
    if ($t) {
        $tags = json_decode($t);
    } else {
        $tags = [];
    }
    if ($tags) {
        start_table();
        table_header("Who", "When", "Tag");
        foreach ($tags as $tag) {
            table_row($tag->who, date_str($tag->when), $tag->tag);
        }
        end_table();
    } else {
        echo "no tags";
    }
}

function add_comment($vol, $run) {
    $t = @file_get_contents("$vol/data/$run/comments.json");
    if ($t) {
        $comments = json_decode($t);
    } else {
        $comments = [];
    }
    $c = new stdClass;
    $c->who = login_name();
    $c->when = time();
    $c->comment = post_str('comment');
    if (!$c->who) {
        die("must be logged in");
    }
    if (!$c->comment) {
        die("must give a comment");
    }
    $comments[] = $c;
    file_put_contents("$vol/data/$run/comments.json", json_encode($comments));
    header("Location: run.php?vol=$vol&name=$run");
}

function comments_form($vol, $run) {
    if (!login_name()) {
        echo "<a href=login.php>Log in</a> to add a comment.";
        return;
    }
    echo "
        <p>
        Add comment:
        <p>
        <form method=post action=run.php>
        <input type=hidden name=name value=$run>
        <input type=hidden name=vol value=$vol>
        <textarea name=comment rows4 cols=40></textarea>
        <p>
        <input type=submit name=add_comment value=OK>
        </form>
    ";
}

function show_comments($vol, $run) {
    $t = @file_get_contents("$vol/data/$run/comments.json");
    if ($t) {
        $comments = json_decode($t);
    } else {
        $comments = [];
    }
    if ($comments) {
        start_table();
        table_header("Who", "When", "Comment");
        foreach ($comments as $comment) {
            table_row($comment->who, date_str($comment->when), $comment->comment);
        }
        end_table();
    } else {
        echo "no comments";
    }
}

function main($vol, $run) {
    page_head("Observing run: $vol $run");

    $dir = "$vol/data/$run";

    echo "<h2>Data files</h2>";
    start_table('table-striped');
    table_header(
        "Start time<br><small>click for details</small>",
        "Type",
        "Module",
        "Size (MB)"
    );
    foreach (scandir($dir) as $f) {
        if ($f[0] == ".") continue;
        if (!is_pff($f)) continue;
        $n = filesize("$vol/data/$run/$f");
        if (!$n) continue;
        $n = number_format($n/1e6, 2);
        $p = parse_pff_name($f);
        if (!$p) continue;
        $start = iso_to_dt($p['start']);
        dt_to_local($start);
        table_row(
            sprintf(
                '<a href=file.php?vol=%s&run=%s&fname=%s>%s</a>',
                $vol, $run, $f, dt_time_str($start)
            ),
            $p['dp'], $p['module'], $n
        );
    }
    end_table();
    echo "<p><a href=$vol/data/$run/>See all files</a> (config files, HK data)<p>";

    echo "<h2>Analyses</h2>\n";
    show_analysis_types($vol, $run);

    echo "<h2>Comments</h2>";
    show_comments($vol, $run);
    echo "<p>";
    comments_form($vol, $run);

    echo "<h2>Tags</h2>";
    show_tags($vol, $run);
    tags_form($vol, $run);
    page_tail();
}

$run = get_str('name', true);
if ($run) {
    $vol = get_str('vol');
} else {
    $run = post_str('name');
    $vol = post_str('vol');
}
check_filename($vol);
check_filename($run);

if (post_str('add_comment', true)) {
    add_comment($vol, $run);
} else if (post_str('add_tag', true)) {
    add_tag($vol, $run);
} else {
    main($vol, $run);
}

?>
