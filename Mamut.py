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
parser.add_option("-a", "--append", dest="appendpath", type="string",
                  help="mamut XML file to append")
parser.add_option("-p", "--XMLpath", dest="changepath", type="string",
                  help="change BDV XML file path")
parser.add_option("-n", "--nframe", dest="nframe", type="string",
                  help="change number of frames")

(options, args) = parser.parse_args()

if not options.inpath:
    print("You must specify an input file")
    sys.exit(1)
if not options.outpath:
    print("you must specify and output file")
    sys.exit(1)

M = MamutUtils(options.inpath)


if options.merge:
    M.MergeColocalizingSpots()
if options.radius:
    M.CleanBigRadius(options.radius)
if options.appendpath:
    M.AppendFiles(options.appendpath)
if options.changepath:
    M.ChangeXMLPath(options.changepath)
if options.nframe:
    M.ChangeXMLnframe(options.nframe)
if options.unlaid:
    M.CleanUnlaid()
M.regenerateXML()
M.writeXML(options.outpath)
