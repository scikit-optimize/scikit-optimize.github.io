(function() {
	// taken from https://raw.githubusercontent.com/scikit-learn/scikit-learn.github.io/master/index.html
	// scikit-learn contributors
    var latestStable = '0.8';
    var goodPaths = ['stable', 'dev', latestStable];
    var showWarning = (msg) => {
        $('.body[role=main]').prepend(
            '<p style="' +
            [
                'padding: 1em',
                'font-size: 1.5em',
                'border: 1px solid red',
                'background: pink',
            ].join('; ') +
            '">' + msg + '</p>')
    };
    if (location.hostname == 'scikit-optimize.github.io/') {
        if (location.pathname == '/dev/versions.html') {
            // this page is specially generated and linked from all versions
            return;
        }
        var versionPath = location.pathname.split('/')[1];
        if (!goodPaths.includes(versionPath)) {
            showWarning('This is documentation for an old release of ' +
                        'Scikit-optimize (version ' + versionPath + '). Try the ' +
                        '<a href="https://scikit-optimize.github.io">latest stable ' +
                        'release</a> (version ' + latestStable + ') or ' +
                        '<a href="https://scikit-optimize.github.io/dev">development</a> ' +
                        '(unstable) versions.')
        } else if (versionPath == 'dev') {
            showWarning('This is documentation for the unstable ' +
                        'development version of Scikit-optimize. ' +
                        'The latest stable ' +
                        'release is <a href="https://scikit-optimize.github.io">version ' +
                        latestStable + '</a>.')
        }
    }
})()
