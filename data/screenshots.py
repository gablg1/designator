import os
import sys
from urlparse import urlparse

to_scrape = open('to_scrape.dat', 'r')
CUT = True
if CUT:
    to_path = 'cut_screenshots'
else:
	to_path = 'screenshots'
for url in to_scrape:
    # URL can be malformed, so we use urlparse to make sure it becomes
    # nicely formatted
    url = url.strip()
    if url[:7] != 'http://':
        url = 'http://%s' % url
    o = urlparse(url)
    website = o.geturl()
    if CUT:
    	cut = '--cliprect 0x0x1366x768'
    else:
    	cut = ''
    cmd = "capturejs --uri '%s' %s --viewport 1366x768 --output %s/%s.png" % (website, cut, to_path, o.netloc)
    print cmd
    os.system(cmd)
