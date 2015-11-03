import os
import sys
from urlparse import urlparse

to_scrape = open('to_scrape.dat', 'r')
for url in to_scrape:
    # URL can be malformed, so we use urlparse to make sure it becomes
    # nicely formatted
    o = urlparse(url)
    website = o.geturl()
    cmd = "capturejs --uri '%s' --viewport 1366x768 --output screenshots/%s.png" % (website, o.netloc)
    print cmd
    os.system(cmd)
