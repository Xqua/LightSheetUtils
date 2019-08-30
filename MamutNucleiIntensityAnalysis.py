import xmltodict
import h5py
import networkx as nx
import os
from scipy.optimize import curve_fit
from scipy.stats import multivariate_normal
import numpy as np
import collections
import pandas as pd
import progressbar

class IntensityMeasure:
    def __init__(self, inpath):
        self.inpath = inpath
        self.LoadMamut()
        self.LoadH5()

    def LoadMamut(self):
        self.xml = xmltodict.parse(open(self.inpath).read())
        spots = self.xml['TrackMate']['Model']['AllSpots']['SpotsInFrame']
        try:
            tracks = self.xml['TrackMate']['Model']['AllTracks']['Track']
        except:
            tracks = []
        self.BDVXMLpath = self.xml['TrackMate']['Settings']["ImageData"]["@folder"]
        self.BDVXMLfile = self.xml['TrackMate']['Settings']["ImageData"]["@filename"]
        self.BDVXML = xmltodict.parse(open(os.path.join(self.BDVXMLpath, self.BDVXMLfile)).read())
        self.startframe = 0
        G = nx.DiGraph()
        c = 0
        if type(spots) == collections.OrderedDict:
             for spot in spots['Spot']:
                 ID = int(spot['@ID'])
                 X, Y, Z = float(spot['@POSITION_X']), float(spot['@POSITION_Y']), float(spot['@POSITION_Z'])
                 radius = spot['@RADIUS']
                 if '@MANUAL_COLOR' in spot:
                     color = spot['@MANUAL_COLOR']
                 else:
                     color = ''
                 G.add_node(ID, x= X, y= Y , z= Z, frame=int(spot['@FRAME']), radius= float(radius), color=color)
                 # print(G[ID])
        else:
            for frame in spots:
                for spot in frame['Spot']:
                    ID = int(spot['@ID'])
                    X, Y, Z = float(spot['@POSITION_X']), float(spot['@POSITION_Y']), float(spot['@POSITION_Z'])
                    radius = spot['@RADIUS']
                    if '@MANUAL_COLOR' in spot:
                        color = spot['@MANUAL_COLOR']
                    else:
                        color = ''
                    G.add_node(ID, x= X, y= Y , z= Z, frame= c, radius= float(radius), color=color)
                    # print(G[ID])
                c += 1
        self.stopframe = c

        for track in tracks:
            for edge in track['Edge']:
                try:
                    source = int(edge['@SPOT_SOURCE_ID'])
                    target = int(edge['@SPOT_TARGET_ID'])
                    if '@MANUAL_COLOR' in edge:
                        color = edge['@MANUAL_COLOR']
                    else:
                        color = ''
                    v = spatial.distance.euclidean([G.node[source]['x'], G.node[source]['y'], G.node[source]['z']],
                                                   [G.node[target]['x'], G.node[target]['y'], G.node[target]['z']])
                    G.add_edge(source, target, distance=v, color=color)
                except:
                    pass
        self.G = G
        print("Loaded")

    def LoadH5(self):
        path = self.BDVXML["SpimData"]["SequenceDescription"]["ImageLoader"]["hdf5"]["#text"]
        if self.BDVXML["SpimData"]["SequenceDescription"]["ImageLoader"]["hdf5"]["@type"] == 'relative':
            path = os.path.join(self.BDVXMLpath, path)
        self.H5 = h5py.File(path)
        T = self.BDVXML['SpimData']['ViewRegistrations']['ViewRegistration'][0]['ViewTransform']['affine'].split(' ')
        self.T = T
        self.xoff = int(float(T[3]))
        self.yoff = int(float(T[7]))
        self.zoff = int(float(T[-1]))
        self.xscale = int(float(T[0]))
        self.yscale = int(float(T[5]))
        self.zscale = int(float(T[10]))

    def ExtractCube(self, channel, nodeID, radius=None, offset=[0,0,0]):
        if not self.H5.get(channel):
            print("Channel does not exist")
            return False
        N = self.G.node[nodeID]
        T = "t{0:05d}".format(N['frame'])
        path = "{}/{}/0/cells".format(T, channel)
        dt = self.H5.get(path)
        X, Y, Z = int(N['x']), int(N['y']), int(N['z'])
        X = int((X - self.xoff)/self.xscale)
        Y = int((Y - self.yoff)/self.yscale)
        Z = int((Z - self.zoff)/self.zscale)

        # print(X,Y,Z, dt.shape)
        if not radius:
            r = int((N["radius"] / 2) + 5)
        else:
            r = int(radius)
        cube = dt[Z-r:Z+r, Y-r:Y+r, X-r:X+r]
        return cube

    def SetDimensions(self):
        dim = self.BDVXML["SpimData"]["SequenceDescription"]["ViewSetups"]["ViewSetup"][0]["size"]
        dim = dim.split(' ')
        dim = [int(i) for i in [dim[2], dim[1], dim[0]]]
        self.dim = dim

    def sphere(self, shape, radius, position):
        # assume shape and position are both a 3-tuple of int or float
        # the units are pixels / voxels (px for short)
        # radius is a int or float in px
        semisizes = (radius,) * 3

        # genereate the grid for the support points
        # centered at the position indicated by position
        grid = [slice(-x0, dim - x0) for x0, dim in zip(position, shape)]
        position = np.ogrid[grid]
        # calculate the distance of all points from `position` center
        # scaled by the radius
        arr = np.zeros(shape, dtype=float)
        for x_i, semisize in zip(position, semisizes):
            arr += (np.abs(x_i / semisize) ** 2)
        # the inner part of the sphere will have distance below 1
        return arr <= 1.0

    def ExtractIntensityFeaturesOneCube(self, channel, nodeID, offset=[0,0,0]):
        N = self.G.node[nodeID]
        # print("Getting Data")
        cube = self.ExtractCube(channel, nodeID, offset=offset)
        # print("Making Sphere")
        sphere = self.sphere(cube.shape, int(N["radius"])/2, np.array(cube.shape)/2)

        insphere = np.extract(sphere, cube)
        outsphere = np.extract(np.invert(sphere), cube)

        mean = np.mean(insphere)
        Max = np.max(insphere)
        Min = np.min(insphere)
        std = np.std(insphere)
        background_mean = np.mean(outsphere)
        background_Max = np.max(outsphere)
        background_Min = np.min(outsphere)
        background_std = np.std(outsphere)
        return mean, Min, Max, std, int(N["radius"]), background_mean, background_Max, background_Min, background_std

    def ExtractIntensityFeatures(self, channels, savepath=None, offset=[0,0,0]):
        "channels is a list of channels to extract"
        res = []
        widgets = [progressbar.Percentage(), progressbar.ETA(), progressbar.Bar()]
        bar = progressbar.ProgressBar(widgets=widgets, maxval=len(self.G.nodes())*len(channels)).start()
        i = 0
        for nodeID in self.G.nodes():
            for channel in channels:
                # print("Doing Node", nodeID, "and channel", channel)
                # try:
                N = self.G.node[nodeID]
                X, Y, Z = int(N['x']), int(N['y']), int(N['z'])
                mean, Min, Max, std, radius, background_mean, background_Max, background_Min, background_std = self.ExtractIntensityFeaturesOneCube(channel, nodeID, offset=offset)
                res.append([nodeID, X, Y, Z, channel, radius, mean, std, Min, Max, background_mean, background_std, background_Max, background_Min])
                # except:
                #     print("Error on node:", nodeID)
                bar.update(i)
                i += 1
        bar.finish()
        df = pd.DataFrame(res, columns=['NodeID', 'X', 'Y', 'Z', 'Channel', 'Radius', 'Mean', 'Std', 'Min', 'Max',  "background_Mean", "background_Max", "background_Min", "background_Std"])
        if savepath:
            df.to_csv(savepath, index=False)
        return df
