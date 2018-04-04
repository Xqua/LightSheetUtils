#!/usr/bin/env python3

import h5py
import os
import sys
import numpy as np
from optparse import OptionParser


parser = OptionParser()
parser.add_option("-i", "--input", dest="filename", type="string",
                  help="[REQUIRED] hdf5 file path")
parser.add_option("-o", "--output", dest="filenameout", type="string",
                  help="[REQUIRED] bin file path")
parser.add_option("-l", "--level", dest="level", type="int",
                  help="[REQUIRED] resolution level to extract (0: full size, 1: 2x downsample, 2: 4x downsample, 3: 8x downsample)")
parser.add_option("-0", "--timepoint-start", dest="t0", type="int", default=None,
                  help="first time point to extract")
parser.add_option("-n", "--timepoint-end", dest="tn", type="int", default=None,
                  help="last time point to extract")
parser.add_option("-C", "--multichannel",
                  action="store_true", dest="multichannel", default=False,
                  help="Is the dataset a multichannel dataset?")
parser.add_option("-N", "--normalize",
                  action="store_true", dest="normalize", default=False,
                  help="Global intensity normalization?")
parser.add_option("-c", "--crop", dest="crop", type="int", default=None,
                  help="Number of pixels to crop from the cube")
parser.add_option("-m", "--min", dest="min", type="string", default=None,
                  help="Manually set the min. channel1,channel2")
parser.add_option("-M", "--max", dest="max", type="string", default=None,
                  help="Manually set the max. channel1,channel2")


(options, args) = parser.parse_args()

print(options)

if not options.filename or not options.filenameout or options.level == None:
    sys.exit(1)

print('bip')

hdf5 = h5py.File(options.filename,'r')

groups = [i for i in hdf5.keys()]
channels = [i for i in groups if 's' in i]
if not options.multichannel:
    channels = [channels[0]]

outfiles = ["{}_{}.bin".format(options.filenameout, i) for i in channels]

timepoints = [int(i.replace('t','')) for i in groups if 't' in i]
if options.t0:
    timepoints = [i for i in timepoints if i >= int(options.t0)]
if options.tn:
    timepoints = [i for i in timepoints if i <= int(options.tn)]

timepoints = sorted(timepoints)

print("Going to extract, parameters: ")
print("HDF5 file: {}".format(options.filename))
print("Timepoints range: {0} - {1}".format( min(timepoints) , max(timepoints) ))
print("{} channels: {}".format(len(channels), ', '.join(channels)))
if options.multichannel:
    print("Outfiles: ")
else:
    print("Outfile: {}")
[print(i) for i in outfiles]


def rescale(im, Min=None, Max=None):
    if Min == None:
        Min = im.min()
    if Max == None:
        Max = im.max()
    im -= Min
    im = np.clip(im, 0, None)
    im = im * 255.0/Max
    im = im.astype(np.uint8)
    return im

L = options.level

norm = {}

if options.min:
    options.min = [int(i) for i in options.min.strip().split(',')]
if options.max:
    options.max = [int(i) for i in options.max.strip().split(',')]


if options.normalize:
    print("Finding global maximum for normalization")
    for c in range(len(channels)):
        channel = channels[c]
        if c not in norm:
            norm[channel] = {'Min':0 , 'Max':0 }
        for timepoint in timepoints:
            if not options.max or not options.min:
                im = hdf5.get('t{0:05d}/{1}/{2}/cells'.format(timepoint, channel, 1)).value

                if options.max:
                    norm[channel]['Max'] = options.max[c]
                else:
                    ma = np.max(im)
                    if ma > norm[channel]['Max']:
                        norm[channel]['Max'] = ma

                if options.min:
                    norm[channel]['Min'] = options.min[c]
                else:
                    mi = np.min(im)
                    if mi > norm[channel]['Min']:
                        norm[channel]['Min'] = mi
            else:
                norm[channel]['Max'] = options.max[c]
                norm[channel]['Min'] = options.min[c]
    print("Normalization values: {}".format(norm))

print("Extracting ...")


for c in range(len(channels)):
    print("Extracting channel: {}".format(channels[c]))
    channel = channels[c]
    binfile = open(outfiles[c],'wb')
    for timepoint in timepoints:
        print("Extracting timepoint: {}".format(timepoint))
        im = hdf5.get('t{0:05d}/{1}/{2}/cells'.format(timepoint, channel, L)).value.astype(np.int16)
        if options.crop:
            c = int(options.crop)
            im = im[c:-c, c:-c, c:-c]
        im = rescale(im, Min=norm[channel]['Min'], Max=norm[channel]['Max'])
        im.T.flatten().tofile(binfile)
print("Cube Dimensions: | x {} | y {} | z {} |".format(im.shape[0], im.shape[1], im.shape[2]))
