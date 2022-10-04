<?php

// main page for an observing run.
// show links to the component files,
// and show comments and tags

require_once("panoseti.inc");
require_once("analysis.inc");

function add_tag($run) {
    $t = @file_get_contents("data/$run/tags.json");
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
    file_put_contents("data/$run/tags.json", json_encode($tags));
    header("Location: run.php?name=$run");
}

function tags_form($run) {
    if (!login_name()) {
        echo "<a href=login.php>Log in</a> to add a tag.";
        return;
    }
    echo "
        <p>
        Add tag:
        <form method=post action=run.php>
        <input type=hidden name=name value=$run>
        <input name=tag>
        <p><p>
        <input type=submit name=add_tag value=OK>
        </form>
    ";
}

function show_tags($run) {
    $t = @file_get_contents("data/$run/tags.json");
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

function add_comment($run) {
    $t = @file_get_contents("data/$run/comments.json");
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
    file_put_contents("data/$run/comments.json", json_encode($comments));
    header("Location: run.php?name=$run");
}

function comments_form($run) {
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
        <textarea name=comment rows4 cols=40></textarea>
        <p>
        <input type=submit name=add_comment value=OK>
        </form>
    ";
}

function show_comments($run) {
    $t = @file_get_contents("data/$run/comments.json");
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

function main($run) {
    page_head("Observing run: $run");

    $dir = "data/$run";

    echo "<h2>Data files</h2>";
    start_table('table-striped');
    table_header(
        "Start time<br><small>click for details</small>",
        "Type",
        "Dome",
        "Module",
        "Size (MB)"
    );
    foreach (scandir($dir) as $f) {
        if ($f[0] == ".") continue;
        if (!is_pff($f)) continue;
        $n = filesize("data/$run/$f");
        if (!$n) continue;
        $n = number_format($n/1e6, 2);
        $p = parse_pff_name($f);
        if (!$p) continue;
        $start = iso_to_dt($p['start']);
        dt_to_local($start);
        table_row(
            sprintf(
                '<a href=file.php?run=%s&fname=%s>%s</a>',
                $run, $f, dt_time_str($start)
            ),
            $p['dp'], $p['dome'], $p['module'], $n
        );
    }
    end_table();
    echo "<h2>Other files</h2>";
    foreach (scandir($dir) as $f) {
        if ($f[0] == ".") continue;
        if (is_pff($f)) continue;
        if (in_array($f, ['comments.json', 'tags.json'])) continue;
        echo "<p>
            <a href=data/$run/$f>$f</a>
        ";
    }

    echo "<h2>Analyses</h2>\n";
    show_analysis_types($run);

    echo "<h2>Comments</h2>";
    show_comments($run);
    echo "<p>";
    comments_form($run);

    echo "<h2>Tags</h2>";
    show_tags($run);
    tags_form($run);
    page_tail();
}

$run = get_str('name', true);
if (!$run) $run = post_str('name');
check_filename($run);

if (post_str('add_comment', true)) {
    add_comment($run);
} else if (post_str('add_tag', true)) {
    add_tag($run);
} else {
    main($run);
}

?>
