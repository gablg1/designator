var phantom = require('phantom');

if (process.argv.length != 3) {
    console.log('Wrong number of args');
    process.exit();
}
phantom.create(function (ph) {
    ph.createPage(function (page) {
        page.open(process.argv[2], function (status) {
        if (status !== 'success') {
            console.log('Unable to access network');
        } else {
            page.injectJs('jquery.min.js');
            page.evaluate(function () {
                var jQuery = window.jQuery || window.$;
                var fontArray = {};
                jQuery('body *').each(function() {
                    var fontFamily = jQuery(this).css("font-family");
                    if (fontFamily) {
                    if (fontFamily in fontArray)
                        fontArray[fontFamily]++;
                    else
                        fontArray[fontFamily] = 1;
                    }
                });
                return fontArray;
            }, function (fontArray) {
                var keysSorted = Object.keys(fontArray);
                keysSorted.sort(function(a,b){return fontArray[a]-fontArray[b]});
                keysSorted.reverse();

                for (key in keysSorted) {
                    font = keysSorted[key];
                    console.log(font);
                    console.log(fontArray[font]);
                }
                ph.exit();
            });
        }
    });
  });
});
