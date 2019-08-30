#!/usr/bin/env python3

import xmltodict
import sys
from optparse import OptionParser
import os
import h5py
import tqdm

parser = OptionParser()
parser.add_option("-1", "--file1", dest="input1", type="string",
                  help="[REQUIRED] path to first HDF5 file")
parser.add_option("-2", "--file2", dest="input2", type="string",
                  help="[REQUIRED] path to second HDF5 file")
parser.add_option("-o", "--output", dest="outpath", type="string",
                  help="[REQUIRED] HDF5 output file path")

(options, args) = parser.parse_args()

print("Preparing for copy ...")
# First we need to get the files and organize them by ViewSetup

H5_1 = h5py.File(options.input1, 'r')
H5_2 = h5py.File(options.input2, 'r')
H5_out = h5py.File(options.outpath, 'w')

keys_1 = [k for k in H5_1.keys()]
keys_2 = [k for k in H5_2.keys()]

channels_1 = [k for k in keys_1 if k[0] == 's']
channels_2 = [k for k in keys_2 if k[0] == 's']
tps_1 = [k for k in keys_1 if k[0] == 't']
tps_2 = [k for k in keys_2 if k[0] == 't']
dims_1 = H5_1.get('/{}/{}/0/cells'.format(tps_1[0], channels_1[0])).shape
dims_2 = H5_2.get('/{}/{}/0/cells'.format(tps_2[0], channels_2[0])).shape

assert(dims_1 == dims_2)
assert(channels_1 == channels_2)

from_1 = [k for k in keys_1 if k not in keys_2]
from_2 = [k for k in keys_2 if k not in from_1]

print("Copying for file into output ...")
for k in tqdm.tqdm(from_1):
    H5_1.copy(k, H5_out)

print("Copying second file into output ...")
for k in tqdm.tqdm(from_2):
    H5_2.copy(k, H5_out)
