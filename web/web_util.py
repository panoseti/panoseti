def page_head(title):
    return '''
        <html lang="en">
        <head>
        <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate" />
        <meta http-equiv="Pragma" content="no-cache" />
        <meta http-equiv="Expires" content="0" />
        <title>%s</title>
        <meta charset="utf-8">
        <link type="text/css" rel="stylesheet" href="https://setiathome.berkeley.edu/panoseti/bootstrap.min.css" media="all">
        <link rel=stylesheet type="text/css" href="https://setiathome.berkeley.edu/panoseti/sah_custom_dark.css">
        </head>
        <body >
        <div class="container-fluid">
        <h1>%s</h1>
    '''%(title, title)

def page_tail():
    return '''
        </div>
        </body>
        </html>
    '''

def table_start(tclass=''):
    return '''
        <div class="table">
        <table width=100%% class="table table-condensed %s">
    '''%tclass

def table_end():
    return '''
        </table>
        </div>
    '''

def table_header(cols):
    z = '<tr>'
    for y in cols:
        z += '<th>%s</th>'%y
    z += '</tr>\n'
    return z

def table_row(cols):
    z = '<tr>'
    for y in cols:
        z += '<td>%s</td>'%y
    z += '</tr>\n'
    return z

def table_subheader(x):
    return '<tr><td class="info" colspan=99>%s</td></tr>\n'%x
