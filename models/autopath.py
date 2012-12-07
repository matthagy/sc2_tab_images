'''Install lib directory in python path
'''

import sys
import os.path as pth

scriptdir = pth.dirname(pth.abspath(__file__))
rootdir = pth.dirname(scriptdir)
libdir = pth.join(rootdir, 'lib')
assert pth.isdir(libdir)

sys.path.append(libdir)
