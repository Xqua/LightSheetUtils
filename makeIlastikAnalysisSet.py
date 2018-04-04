#!/usr/bin/env python3

import h5py
import os
import sys
import numpy as np
from optparse import OptionParser
from skimage.io._plugins import freeimage_plugin as freeimg


parser = OptionParser()
parser.add_option("-i", "--input", dest="filename", type="string",
                  help="[REQUIRED] hdf5 file path")
parser.add_option("-o", "--output", dest="filenameout", type="string",
                  help="[REQUIRED] bin file path")
parser.add_option("-l", "--level", dest="level", type="int",
                  help="[REQUIRED] resolution level to extract (0: full size, 1: 2x downsample, 2: 4x downsample, 3: 8x downsample)")
parser.add_option("-t", "--timepoint-slices", dest="tc", type="int", default=0,
                  help="[OPTIONAL] Timepoints reslice factor (2 will take every 2 tp, 3 every 3 etc) to extract")
parser.add_option("-C", "--multichannel",
                  action="store_true", dest="multichannel", default=False,
                  help="Is the dataset a multichannel dataset?")
parser.add_option("-c", "--crop", dest="crop", type="int", default=None,
                  help="Number of pixels to crop from the cube")
parser.add_option("-0", "--timepoint-start", dest="t0", type="int", default=None,
                  help="first time point to extract")
parser.add_option("-n", "--timepoint-end", dest="tn", type="int", default=None,
                  help="last time point to extract")


(options, args) = parser.parse_args()

print(options)

if not options.filename or not options.filenameout or options.level == None:
    sys.exit(1)


hdf5 = h5py.File(options.filename,'r')

groups = [i for i in hdf5.keys()]
channels = [i for i in groups if 's' in i]
if not options.multichannel:
    channels = [channels[0]]

outfiles = ["{}_{}.h5".format(options.filenameout, i) for i in channels]

timepoints = [int(i.replace('t','')) for i in groups if 't' in i]
if options.t0:
    timepoints = [i for i in timepoints if i >= int(options.t0)]
if options.tn:
    timepoints = [i for i in timepoints if i <= int(options.tn)]
timepoints = sorted(timepoints)
if options.tc:
    timepoints = timepoints[0::options.tc]


print("Going to extract, parameters: ")
print("HDF5 file: {}".format(options.filename))
print("Timepoints range: {}".format( '-'.join([str(i) for i in timepoints]) ))
print("{} channels: {}".format(len(channels), ', '.join(channels)))
if options.multichannel:
    print("Outfiles: ")
else:
    print("Outfile: ")
[print(i) for i in outfiles]


def rescale(im, Min=None, Max=None):
    if Min == None:
        Min = im.min()
    if Max == None:
        Max = im.max()
    im -= Min
    im = np.clip(im, 0, None)
    if options.dtype == 8:
        im = im * 255.0/Max
        im = im.astype(np.uint8)
    elif options.dtype == 16:
        im = im * 65535.0/Max
        im = im.astype(np.uint16)
    return im

L = options.level

norm = {}

print("Extracting ...")

for c in range(len(channels))[1:]:
    print("Extracting channel: {}".format(channels[c]))
    channel = channels[c]
    im = hdf5.get('t{0:05d}/{1}/{2}/cells'.format(0, channel, L)).value.astype(np.uint16)
    shape = list(im.shape)
    shape = [len(timepoints)] + shape
    shape = tuple(shape)
    h5file = h5py.File(outfiles[c], 'w')
    grp = h5file.create_group('/dataset')
    res = grp.create_dataset("images", shape=shape, chunks=True, dtype=np.uint16, compression="gzip")
    for i in range(len(timepoints)):
        timepoint = timepoints[i]
        print("Extracting timepoint: {}".format(timepoint))
        im = hdf5.get('t{0:05d}/{1}/{2}/cells'.format(timepoint, channel, L)).value.astype(np.uint16)
        if options.crop:
            c = int(options.crop)
            im = im[c:-c, c:-c, c:-c]
        res[i] = im
