#!/usr/bin/env python3

import xmltodict
import sys
from optparse import OptionParser

parser = OptionParser()
parser.add_option("-i", "--input", dest="inpath", type="string",
                  help="[REQUIRED] mamut XML file path")
parser.add_option("-o", "--output", dest="outpath", type="string",
                  help="[REQUIRED] mamut XML output file path")

(options, args) = parser.parse_args()

if not options.inpath:
    print("You must specify an input file")
    sys.exit(1)
if not options.outpath:
    print("you must specify and output file")
    sys.exit(1)

xml = xmltodict.parse(open(options.inpath).read())

spots = xml['TrackMate']['Model']['AllSpots']['SpotsInFrame']
tracks = xml['TrackMate']['Model']['AllTracks']['Track']

spotsInTrack = []

for track in tracks:
    for edge in track['Edge']:
        try:
            if edge['@SPOT_SOURCE_ID'] not in spotsInTrack:
                spotsInTrack.append(edge['@SPOT_SOURCE_ID'])
            if edge['@SPOT_TARGET_ID'] not in spotsInTrack:
                spotsInTrack.append(edge['@SPOT_TARGET_ID'])
        except:
            # for k in edge:
                # print(k, edge[k])
            pass
newframes = []

c = 0
for frame in spots:
    newframe = {'@frame': c, 'Spot':[]}
    c += 1
    for spot in frame['Spot']:
        ID = spot['@ID']
        if ID in spotsInTrack:
            newframe['Spot'].append(spot)
    newframes.append(newframe)

xml['TrackMate']['Model']['AllSpots']['SpotsInFrame'] = newframes
result = xmltodict.unparse(xml)

f = open(options.outpath, 'w')
f.write(result)
f.close()
