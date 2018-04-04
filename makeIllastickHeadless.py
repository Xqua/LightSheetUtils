#!/usr/bin/env python3

import os
import sys
from optparse import OptionParser


parser = OptionParser()
parser.add_option("-i", "--input", dest="inputpath", type="string",
                  help="[REQUIRED] hdf5 file path")
parser.add_option("-p", "--project", dest="projectpath", type="string",
                  help="[REQUIRED] Illastick Project File")
parser.add_option("-e", "--export_source", dest="export", type="string",
                  help="[REQUIRED] Exported Object")
parser.add_option("-l", "--level", dest="level", type="int",
                  help="[REQUIRED] resolution level to extract (0: full size, 1: 2x downsample, 2: 4x downsample, 3: 8x downsample)")
parser.add_option("-c", "--channel", dest="channel", type="string",
                  help="[REQUIRED] bin file path")
parser.add_option("-0", "--timepoint-start", dest="t0", type="int", default=None,
                  help="first time point to extract")
parser.add_option("-n", "--timepoint-end", dest="tn", type="int", default=None,
                  help="last time point to extract")

(options, args) = parser.parse_args()
# Lets make the file path string for inputs
paths = []
for i in range(options.t0, options.tn+1):
    s = os.path.join(options.inputpath, "t{0:05d}".format(i), options.channel, str(options.level), "cells")
    paths.append(s)

print(paths)



cmd = './run_ilastik.sh --headless --project={} --export_source="{}" --output_format=hdf5 --output_filename_format={{dataset_dir}}/{{nickname}}_{{result_type}}.h5 {}'.format(options.projectpath, options.export, ' '.join(paths))
print(cmd)
