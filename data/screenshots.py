import os
import sys
from urlparse import urlparse
import csv

websites_file = 'top-10k.csv'
CUT = True
if CUT:
    to_path = 'cut_screenshots'
else:
	to_path = 'screenshots'

with open(websites_file, 'r') as csvfile:
    lines = csv.reader(csvfile, delimiter=',')
    for line in lines:
    	assert(len(lines) == 2)
    	ranking, url = line[0], line[1]

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
        print "Website number %d" % ranking
        print cmd
        os.system(cmd)
