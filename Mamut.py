#!/usr/bin/env python3

import xmltodict
import sys
from optparse import OptionParser
from scipy import spatial
import networkx as nx
import numpy as np
import progressbar
from MamutUtils import MamutUtils



parser = OptionParser()
parser.add_option("-i", "--input", dest="inpath", type="string",
                  help="[REQUIRED] mamut XML file path")
parser.add_option("-o", "--output", dest="outpath", type="string",
                  help="[REQUIRED] mamut XML output file path")
parser.add_option("-C", "--cleanUnlaid", dest="unlaid", action="store_true", default=False,
                  help="Remove all spots that are not in a track")
parser.add_option("-M", "--MergeTracks", dest="merge", action="store_true", default=False,
                  help="Merge all Ilastik broken mitosis")
parser.add_option("-R", "--CleanRadius", dest="radius", type="int", default=0,
                  help="(-R radius_X) Remove all spots bigger than radius X")

(options, args) = parser.parse_args()

if not options.inpath:
    print("You must specify an input file")
    sys.exit(1)
if not options.outpath:
    print("you must specify and output file")
    sys.exit(1)

M = MamutUtils(options.inpath)

if options.unlaid:
    M.CleanUnlaid()
if options.merge:
    M.MergeColocalizingSpots()
if options.radius:
    M.CleanBigRadius(options.radius)
M.regenerateXML()
M.writeXML(options.outpath)
