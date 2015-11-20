import os
import sys
from urlparse import urlparse
import csv
import data

CUT = True

amount = 'top-100'
to_path = data.getDataDir(amount=amount, cut=CUT, big=True)
csv_path = '%s/%s' % (amount, amount + '.csv')

with open(csv_path, 'r') as csvfile:
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
