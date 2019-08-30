import xmltodict
import sys
from scipy import spatial
from scipy import stats
import networkx as nx
import numpy as np
import progressbar


class MamutUtils:
    def __init__(self, inpath):
        print("Loading XML...")
        self.xml = xmltodict.parse(open(inpath).read())
        spots = self.xml['TrackMate']['Model']['AllSpots']['SpotsInFrame']
        tracks = self.xml['TrackMate']['Model']['AllTracks']['Track']
        self.startframe = 0
        G = nx.DiGraph()
        c = 0
        for frame in spots:
            for spot in frame['Spot']:
                ID = int(spot['@ID'])
                X, Y, Z = float(spot['@POSITION_X']), float(spot['@POSITION_Y']), float(spot['@POSITION_Z'])
                radius = spot['@RADIUS']
                if '@MANUAL_COLOR' in spot:
                    color = spot['@MANUAL_COLOR']
                else:
                    color = ''
                G.add_node(ID, x=X, y= Y, z= Z, frame= c, radius= radius, color=color)
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

    def makeSpot(self, spotID):
        res = {'@ID': spotID,
               '@name': 'ID{}'.format(spotID),
               '@VISIBILITY': '1',
               '@QUALITY': '0.0',
               '@RADIUS': self.G.nodes[spotID]['radius'],
               '@POSITION_X': self.G.nodes[spotID]['x'],
               '@POSITION_Y': self.G.nodes[spotID]['y'],
               '@POSITION_Z': self.G.nodes[spotID]['z'],
               '@POSITION_T': self.G.nodes[spotID]['frame'],
               '@FRAME': self.G.nodes[spotID]['frame']}
        if self.G.nodes[spotID]['color']:
            res['@MANUAL_COLOR'] = self.G.nodes[spotID]['color']
        return res

    def makeFrame(self, frame):
        spots = []
        nodes = [n for n in self.G if self.G.nodes[n]['frame'] == frame]
        for node in nodes:
            spots.append(self.makeSpot(node))
        res = {
            '@frame': frame,
            'Spot': spots
        }
        return res

    def makeEdge(self, source, target):
        res = {'@LINK_COST' : '0.0',
               '@SPOT_SOURCE_ID' : source,
               '@SPOT_TARGET_ID' : target,
               '@VELOCITY' : self.G.edges[source, target]['relative_distance'],
               '@DISPLACEMENT' : self.G.edges[source, target]['distance'],
               '@RELVELOCITY' : self.G.edges[source, target]['relative_distance']}
        if self.G.edges[source, target]['color']:
            res['@MANUAL_COLOR'] = self.G.edges[source, target]['color']
        return res

    def makeTrack(self, root, trackID):
        start = 1000000
        stop = 0
        Edges = []
        displacement, gaps, gapmax, nbsplit, nbmerge = 0, 0, 0, 0, 0

        trackNodes = list(nx.nodes(nx.dfs_tree(self.G, root)))
        trackEdges = list(nx.edges(nx.dfs_tree(self.G, root)))
        velocities = []
        for edge in trackEdges:
            if 'distance' not in self.G.edges[edge[0], edge[1]]:
                self.G.edges[edge[0], edge[1]]['distance'] = self.get_distance(edge[0], edge[1])
            velocities.append(self.G.edges[edge[0], edge[1]]['distance'])

        Zs = stats.zscore(velocities)

        for i in range(len(trackEdges)):
            edge = trackEdges[i]
            self.G.edges[edge[0], edge[1]]['relative_distance'] = Zs[i]

        for node in trackNodes:
            frame = self.G.nodes[node]['frame']
            start = min(start, frame)
            stop = max(stop, frame)

        splitcounted = []
        mergecounted = []

        for edge in trackEdges:
            # if 'distance' not in self.G.edge[edge[0]][edge[1]]:
            #     self.G.edge[edge[0]][edge[1]]['distance'] = self.get_distance(edge[0], edge[1])
            displacement += self.G.edges[edge[0], edge[1]]['distance']
            if edge[0] not in splitcounted:
                splitcounted.append(edge[0])
                if self.G.out_degree(edge[0]) > 1: nbsplit += 1
            if edge[1] not in mergecounted:
                mergecounted.append(edge[0])
                if self.G.in_degree(edge[1]) > 1: nbmerge += 1
            if abs(self.G.nodes[edge[0]]['frame'] - self.G.nodes[edge[1]]['frame']) > 1:
                gaps += 1
                gapmax = max(gapmax, abs(self.G.nodes[edge[0]]['frame'] - self.G.nodes[edge[1]]['frame']))
            Edges.append(self.makeEdge(edge[0], edge[1]))

        res = {
            '@name': "Track_{}".format(trackID),
            '@TRACK_ID': trackID,
            '@TRACK_INDEX': trackID,
            '@TRACK_DURATION': stop-start,
            '@TRACK_START': start,
            '@TRACK_STOP': stop,
            '@TRACK_DISPLACEMENT': displacement,
            '@NUMBER_SPOTS': len(trackNodes),
            '@NUMBER_GAPS': gaps,
            '@LONGEST_GAP': gapmax,
            '@NUMBER_SPLITS': nbsplit,
            '@NUMBER_MERGES': nbmerge,
            '@NUMBER_COMPLEX': 0,
            '@DIVISION_TIME_MEAN':0,
            '@DIVISION_TIME_STD':0,
            'Edge':Edges}
        return res

    def MakeEdgeFeatures(self):
        features = self.xml['TrackMate']['Model']['FeatureDeclarations']['EdgeFeatures']['Feature']
        features.append(self.MakeEdgeFeature('RELVELOCITY', "Velocity_Zscore", "VelZ", 'NONE', 'false'))
        return features

    def MakeEdgeFeature(self, featureID, featureNAME, featureSHORTNAME, dimension, isint):
        feature = xmltodict.OrderedDict()
        feature['@feature'] = featureID
        feature['@name'] = featureNAME
        feature['@shortname'] = featureSHORTNAME
        feature['@dimension'] = dimension
        feature['@isint'] =  isint

        return feature

    def makeTracks(self):
        tracks = []
        components = []
        for i in nx.components.weakly_connected_components(self.G):
            components.append(i)
        i = 0
        for component in components:
            # if len(component) > 1:
            root = self.get_root(list(component)[0])
            tracks.append(self.makeTrack(root, i))
            i += 1
        return tracks

    def makeSpots(self):
        spots = []
        for frame in range(self.stopframe):
            spots.append(self.makeFrame(frame))
        return spots

    def RenameNodes(self):
        nodes = self.G.nodes()
        nodes = sorted(nodes)
        new = range(len(nodes))
        mapping = {}
        for i in range(len(nodes)):
            mapping[nodes[i]] = new[i]
        nx.relabel_nodes(self.G, mapping, copy=False)

    def regenerateXML(self):
        self.RenameNodes()
        spots = self.makeSpots()
        print("Regenerated {} frames".format(len(spots)))
        print("With {} spots".format(len(self.G)))
        tracks = self.makeTracks()
        print("Regenerated {} tracks".format(len(tracks)))
        print("With {} edges".format(len(self.G.edges())))
        tracksIDs = [{"@TRACK_ID":track['@TRACK_ID']} for track in tracks]
        EdgeFeatures = self.MakeEdgeFeatures()

        self.xml['TrackMate']['Model']['FilteredTracks']['TrackID'] = tracksIDs
        self.xml['TrackMate']['Model']['AllSpots']['SpotsInFrame'] = spots
        self.xml['TrackMate']['Model']['AllTracks']['Track'] = tracks
        self.xml['TrackMate']['Model']['FeatureDeclarations']['EdgeFeatures']['Feature'] = EdgeFeatures

    def MergeLoadXML(self, inpath, shift=0):
        print("Loading XML...")
        xml = xmltodict.parse(open(inpath).read())
        spots = xml['TrackMate']['Model']['AllSpots']['SpotsInFrame']
        tracks = xml['TrackMate']['Model']['AllTracks']['Track']
        startframe = 0
        G = nx.DiGraph()
        c = shift
        for frame in spots:
            for spot in frame['Spot']:
                ID = int(spot['@ID']) + 100000
                X, Y, Z = float(spot['@POSITION_X']), float(spot['@POSITION_Y']), float(spot['@POSITION_Z'])
                radius = spot['@RADIUS']
                if '@MANUAL_COLOR' in spot:
                    color = spot['@MANUAL_COLOR']
                else:
                    color = ''
                G.add_node(ID,x= X, y= Y, z= Z, frame= c, radius= radius, color=color)
                # print(G[ID])
            c += 1
        stopframe = c

        for track in tracks:
            for edge in track['Edge']:
                try:
                    source = int(edge['@SPOT_SOURCE_ID']) + 100000
                    target = int(edge['@SPOT_TARGET_ID']) + 100000
                    if '@MANUAL_COLOR' in edge:
                        color = edge['@MANUAL_COLOR']
                    else:
                        color = ''
                    v = spatial.distance.euclidean([G.node[source]['x'], G.node[source]['y'], G.node[source]['z']],
                                                   [G.node[target]['x'], G.node[target]['y'], G.node[target]['z']])
                    G.add_edge(source, target, distance=v, color=color)
                except:
                    pass
        print("Loaded")
        return G, stopframe

    def AppendFiles(self, inpath):
        G, stopframe = self.MergeLoadXML(inpath, self.stopframe)
        print("Merging trees...")
        self.stopframe = stopframe
        self.G = nx.compose(self.G,G)

    def ChangeXMLPath(self, path):
        self.xml['TrackMate']['Settings']['ImageData']['@filename'] = path

    def ChangeXMLnframe(self, nframe):
        self.xml['TrackMate']['Settings']['ImageData']['@nframes'] = nframe

    def writeXML(self, outpath):
        print("Writting to: " + outpath)
        f = open(outpath, 'w')
        f.write(xmltodict.unparse(self.xml, pretty=True))
        f.close()
        print("Saved !")

    def get_root(self, ID):
        p = list(self.G.predecessors(ID))
        lastp = p
        if p:
            while True:
                p = list(self.G.predecessors(lastp[0]))
                if p:
                    lastp = p
                else:
                    break
            return lastp[0]
        else:
            return ID

    def get_distance(self, source, target):
        v = spatial.distance.euclidean([self.G.nodes[source]['x'], self.G.nodes[source]['y'], self.G.nodes[source]['z']],
                                       [self.G.nodes[target]['x'], self.G.nodes[target]['y'], self.G.nodes[target]['z']])
        return v

    def CleanUnlaid(self):
        print("Cleaning Unlaid spots")
        widgets = [progressbar.Percentage(), progressbar.Bar(), progressbar.ETA()]
        bar = progressbar.ProgressBar(widgets=widgets, max_value=len(self.G)).start()
        p = 0
        toremove = []
        for node in self.G:
            if self.G.degree(node) == 0:
                toremove.append(node)
            bar.update(p)
            p += 1
        for node in toremove:
            self.G.remove_node(node)
        bar.finish()

    def CleanBigRadius(self, radius):
        print("Cleaning BigRadius spots")
        widgets = [progressbar.Percentage(), progressbar.Bar(), progressbar.ETA()]
        bar = progressbar.ProgressBar(widgets=widgets, max_value=len(self.G)).start()
        p = 0
        toremove = []
        for node in self.G:
            if float(self.G.nodes[node]['radius']) > radius:
                toremove.append(node)
            bar.update(p)
            p += 1
        for node in toremove:
            self.G.remove_node(node)
        bar.finish()

    def MergeColocalizingSpots(self):
        print("Merging colocalized spots ...")
        widgets = [progressbar.Percentage(), progressbar.Bar(), progressbar.ETA()]
        steps = 0
        for frame in range(self.stopframe):
            spots = [n for n in self.G if self.G.nodes[n]['frame'] == frame]
            steps += ((len(spots) * len(spots)) / 2)
        bar = progressbar.ProgressBar(widgets=widgets, max_value=steps).start()
        self.spotsToMerge = {}
        nodetoremove = []
        edgetoremove = []
        p = 0
        c = 0
        for frame in range(self.stopframe):
            spots = [n for n in self.G if self.G.nodes[n]['frame'] == frame]
            for i in range(len(spots)):
                for j in range(len(spots)):
                    if i > j:
                        v = self.get_distance(spots[i], spots[j])
                        if v < 0.000001:
                            c += 1
                            pred_i = self.G.predecessors(spots[i])
                            pred_j = self.G.predecessors(spots[j])
                            if pred_i:
                                if pred_j:
                                    d1 = self.get_distance(pred_i[0], spots[j])
                                    d2 = self.get_distance(pred_j[0], spots[i])
                                    if d1 > d2:
                                        self.G.add_edge(pred_j[0], spots[i])
                                        nodetoremove.append(spots[j])
                                    else:
                                        self.G.add_edge(pred_i[0], spots[j])
                                        nodetoremove.append(spots[i])
                                else:
                                    self.G.add_edge(pred_i[0], spots[j])
                                    nodetoremove.append(spots[i])
                            elif pred_j:
                                self.G.add_edge(pred_j[0], spots[i])
                                nodetoremove.append(spots[j])
                        bar.update(p)
                        p += 1
        print(p)
        print(steps)
        bar.finish()
        nodetoremove = np.unique(nodetoremove)
        for n in nodetoremove:
            self.G.remove_node(n)
