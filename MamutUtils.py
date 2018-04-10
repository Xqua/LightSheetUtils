import xmltodict
import sys
from scipy import spatial
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
                ID = spot['@ID']
                X, Y, Z = float(spot['@POSITION_X']), float(spot['@POSITION_Y']), float(spot['@POSITION_Z'])
                radius = spot['@RADIUS']
                G.add_node(ID, attr_dict={"x": X, "y": Y, "z": Z, "frame": c, "radius": radius})
                # print(G[ID])
            c += 1
        self.stopframe = c

        for track in tracks:
            for edge in track['Edge']:
                try:
                    source = edge['@SPOT_SOURCE_ID']
                    target = edge['@SPOT_TARGET_ID']
                    v = spatial.distance.euclidean([G.node[source]['x'], G.node[source]['y'], G.node[source]['z']],
                                                   [G.node[target]['x'], G.node[target]['y'], G.node[target]['z']])
                    G.add_edge(source, target, distance=v)
                except:
                    pass
        self.G = G
        print("Loaded")

    def makeSpot(self, spotID):
        res = {'@ID': spotID,
               '@name': 'ID{}'.format(spotID),
               '@VISIBILITY': '1',
               '@QUALITY': '0.0',
               '@RADIUS': self.G.node[spotID]['radius'],
               '@POSITION_X': self.G.node[spotID]['x'],
               '@POSITION_Y': self.G.node[spotID]['y'],
               '@POSITION_Z': self.G.node[spotID]['z'],
               '@POSITION_T': self.G.node[spotID]['frame'],
               '@FRAME': self.G.node[spotID]['frame']}
        return res

    def makeFrame(self, frame):
        spots = []
        nodes = [n for n in self.G if self.G.node[n]['frame'] == frame]
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
               '@VELOCITY' : self.G.edge[source][target]['distance'],
               '@DISPLACEMENT' : self.G.edge[source][target]['distance']}
        return res

    def makeTrack(self, root, trackID):
        start = 1000000
        stop = 0
        Edges = []
        displacement, gaps, gapmax, nbsplit, nbmerge = 0, 0, 0, 0, 0

        trackNodes = nx.nodes(nx.dfs_tree(self.G, root))
        trackEdges = nx.edges(nx.dfs_tree(self.G, root))

        for node in trackNodes:
            frame = self.G.node[node]['frame']
            start = min(start, frame)
            stop = max(stop, frame)

        splitcounted = []
        mergecounted = []

        for edge in trackEdges:
            if 'distance' not in self.G.edge[edge[0]][edge[1]]:
                self.G.edge[edge[0]][edge[1]]['distance'] = self.get_distance(edge[0], edge[1])
            displacement += self.G.edge[edge[0]][edge[1]]['distance']
            if edge[0] not in splitcounted:
                splitcounted.append(edge[0])
                if self.G.out_degree(edge[0]) > 1: nbsplit += 1
            if edge[1] not in mergecounted:
                mergecounted.append(edge[0])
                if self.G.in_degree(edge[1]) > 1: nbmerge += 1
            if abs(self.G.node[edge[0]]['frame'] - self.G.node[edge[1]]['frame']) > 1:
                gaps += 1
                gapmax = max(gapmax, abs(self.G.node[edge[0]]['frame'] - self.G.node[edge[1]]['frame']))
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

    def regenerateXML(self):
        spots = self.makeSpots()
        print("Regenerated {} frames".format(len(spots)))
        print("With {} spots".format(len(self.G)))
        tracks = self.makeTracks()
        print("Regenerated {} tracks".format(len(tracks)))
        print("With {} edges".format(len(self.G.edges())))
        tracksIDs = [{"@TRACK_ID":track['@TRACK_ID']} for track in tracks]

        self.xml['TrackMate']['Model']['FilteredTracks']['TrackID'] = tracksIDs
        self.xml['TrackMate']['Model']['AllSpots']['SpotsInFrame'] = spots
        self.xml['TrackMate']['Model']['AllTracks']['Track'] = tracks

    def writeXML(self, outpath):
        print("Writting to: " + outpath)
        f = open(outpath, 'w')
        f.write(xmltodict.unparse(self.xml))
        f.close()
        print("Saved !")

    def get_root(self, ID):
        p = self.G.predecessors(ID)
        lastp = p
        if p:
            while True:
                p = self.G.predecessors(lastp[0])
                if p:
                    lastp = p
                else:
                    break
            return lastp[0]
        else:
            return ID

    def get_distance(self, source, target):
        v = spatial.distance.euclidean([self.G.node[source]['x'], self.G.node[source]['y'], self.G.node[source]['z']],
                                       [self.G.node[target]['x'], self.G.node[target]['y'], self.G.node[target]['z']])
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
            if self.G.node[node]['radius'] > radius:
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
            spots = [n for n in self.G if self.G.node[n]['frame'] == frame]
            steps += ((len(spots) * len(spots)) / 2)
        bar = progressbar.ProgressBar(widgets=widgets, max_value=steps).start()
        self.spotsToMerge = {}
        nodetoremove = []
        edgetoremove = []
        p = 0
        c = 0
        for frame in range(self.stopframe):
            spots = [n for n in self.G if self.G.node[n]['frame'] == frame]
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
