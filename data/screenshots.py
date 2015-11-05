import os
import sys
from urlparse import urlparse
import csv


amount = 'top-15k'
websites_file = amount + '.csv'
CUT = True
if CUT:
    to_path = 'cut_screenshots'
else:
	to_path = 'screenshots'
to_path = '%s/%s' % (amount, to_path)

with open(websites_file, 'r') as csvfile:
    lines = csv.reader(csvfile, delimiter=',')
    for line in lines:
        assert(len(line) == 2)
    	ranking, url = int(line[0]), line[1]
        print "Website number %d" % ranking

        # URL can be malformed, so we use urlparse to make sure it becomes
        # nicely formatted
        url = url.strip()
        if url[:7] != 'http://':
            url = 'http://%s' % url
        o = urlparse(url)

    	outfile = '%s/%d.%s.png' % (to_path, ranking, o.netloc)
        # skip files already there
        if os.path.exists(outfile):
        	print 'Already there!'
        	continue

        website = o.geturl()
        if CUT:
            cut = '--cliprect 0x0x1366x768'
        else:
            cut = ''
        cmd = "capturejs --uri '%s' %s -T 5000 --viewport 1366x768 --output %s/%d.%s.png" % (website, cut, to_path, ranking, o.netloc)
        print cmd
        os.system(cmd)
