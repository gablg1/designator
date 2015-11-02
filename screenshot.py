import os
import sys

name = sys.argv[1]
os.system("capturejs --uri http://%s --selector '.editor-container' --viewport 1366x768 --output %s.png" % (name, name))
