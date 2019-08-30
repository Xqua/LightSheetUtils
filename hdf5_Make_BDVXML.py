#!/usr/bin/env python3

import xmltodict
import sys
from optparse import OptionParser
import os
from collections import OrderedDict
import h5py


parser = OptionParser()
parser.add_option("-i", "--input", dest="input", type="string",
                  help="[REQUIRED] path to HDF5 file")
parser.add_option("-r", "--res", dest="res", type="string",
                  help="[OPTIONAL] pixel resolution of axes in um comma separated: x_res,y_res,z_res")
parser.add_option("-o", "--output", dest="outpath", type="string",
                  help="[REQUIRED] BDV XML output file path")

(options, args) = parser.parse_args()

if options.res:
    res = [float(r) for r in options.res.split(',')]
else:
    res = [0.406, 0.406, 0.406]

print("Getting the file list ...")
# First we need to get the files and organize them by ViewSetup

H5 = h5py.File(options.input, 'r')

keys = [k for k in H5.keys()]

channels = [k for k in keys if k[0] == 's']
tps = [k for k in keys if k[0] == 't']
dims = H5.get('/{}/{}/0/cells'.format(tps[0], channels[0])).shape


print("Generating the xml file ...")
# Define the XML
viewsetups = []
attrs_channels = []
i = 0
size = " ".join([str(i) for i in dims])
voxelSize = " ".join([str(i) for i in res])

for channel in channels:
    c = channel.replace('s','')
    viewsetups.append(
            OrderedDict([('id', c),
                         ('size', size),
                         ('voxelSize', OrderedDict([('unit', 'µm'),
                                                    ('size', voxelSize)
                                                    ])),
                         ('attributes', OrderedDict([('illumination', '1'),
                                                     ('channel', '{}'.format(i)),
                                                     ('tile', '0'),
                                                     ('angle', '2')]))
                         ])
                     )

    attrs_channels.append(
        OrderedDict([('id', '{}'.format(i)),
                     ('name', '{}'.format(i))
                     ])
        )

registrations = []
for tp in tps:
    t = str(int(tp.replace('t','')))
    for channel in channels:
        c = channel.replace('s','')
        registrations.append(
        OrderedDict([('@timepoint', t),
                     ('@setup', c),
                     ('ViewTransform', OrderedDict([('@type', 'affine'),
                                                    ('Name', 'fusion bounding box'),
                                                    ('affine', '1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 0.0')])
                      )
                     ])
        )


XML = OrderedDict([('SpimData',
              OrderedDict([('@version', '0.2'),
                           ('BasePath',
                            OrderedDict([('@type', 'relative'),
                                         ('#text', '.')])),
                           ('SequenceDescription',
                            OrderedDict([('ImageLoader',
                                          OrderedDict([('@format', 'bdv.hdf5'),
                                                       ('hdf5',
                                                        OrderedDict([('@type',
                                                                      'relative'),
                                                                     ('#text', options.input.split('/')[-1] )]))])),
                                         ('ViewSetups',
                                          OrderedDict([('ViewSetup', viewsetups),
                                                       ('Attributes',
                                                        [OrderedDict([('@name',
                                                                       'illumination'),
                                                                      ('Illumination',
                                                                       OrderedDict([('id',
                                                                                     '1'),
                                                                                    ('name',
                                                                                     'Fused_1')]))]),
                                                         OrderedDict([('@name',
                                                                       'channel'),
                                                                      ('Channel', attrs_channels)]),
                                                         OrderedDict([('@name',
                                                                       'tile'),
                                                                      ('Tile',
                                                                       OrderedDict([('id',
                                                                                     '0'),
                                                                                    ('name',
                                                                                     '0')]))]),
                                                         OrderedDict([('@name',
                                                                       'angle'),
                                                                      ('Angle',
                                                                       OrderedDict([('id',
                                                                                     '2'),
                                                                                    ('name',
                                                                                     'Fused_2')]))])])])),
                                         ('Timepoints',
                                          OrderedDict([('@type', 'range'),
                                                       ('first', '{}'.format(int(tps[0].replace('t',''))) ),
                                                       ('last', '{}'.format(int(tps[-1].replace('t',''))) )]))])),
                           ('ViewRegistrations',
                            OrderedDict([('ViewRegistration', registrations)]))
                  ])
              )])

print("Saving the XML file ...")

out = xmltodict.unparse(XML, pretty=True)
f = open(options.outpath, 'w')
f.write(out)
f.close()

print("Done !")
