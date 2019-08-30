import xmltodict
import h5py
import networkx as nx
import os
from scipy.optimize import curve_fit
from scipy.stats import multivariate_normal
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from optparse import OptionParser

import progressbar

# import keras
import keras

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

# import miscellaneous modules
import time

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

# use this environment flag to change which GPU to use
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())




class PixelExtractor:
    def __init__(self, inpath):
        self.inpath = inpath
        self.LoadMamut()
        self.LoadH5()

    def LoadMamut(self):
        self.xml = xmltodict.parse(open(self.inpath).read())
        spots = self.xml['TrackMate']['Model']['AllSpots']['SpotsInFrame']
        tracks = self.xml['TrackMate']['Model']['AllTracks']['Track']
        self.BDVXMLpath = self.xml['TrackMate']['Settings']["ImageData"]["@folder"]
        self.BDVXMLfile = self.xml['TrackMate']['Settings']["ImageData"]["@filename"]
        self.BDVXML = xmltodict.parse(open(os.path.join(self.BDVXMLpath, self.BDVXMLfile)).read())
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

    def ExtractCube(self, channel, nodeID, radius=None):
        if not self.H5.get(channel):
            print("Channel does not exist")
            return False
        N = self.G.node[nodeID]
        T = "t{0:05d}".format(N['frame'])
        path = "{}/{}/0/cells".format(T, channel)
        dt = self.H5.get(path)
        X, Y, Z = int(N['x']), int(N['y']), int(N['z'])
        if not radius:
            r = int(N["radius"])
        else:
            r = int(radius)
        cube = dt[Z-r:Z+r, Y-r:Y+r, X-r:X+r]
        return cube

    def SetDimensions(self):
        dim = self.BDVXML["SpimData"]["SequenceDescription"]["ViewSetups"]["ViewSetup"][0]["size"]
        dim = dim.split(' ')
        dim = [int(i) for i in [dim[2], dim[1], dim[0]]]
        self.dim = dim

    def getCube(self, frame, channel='s17'):
        DataSet = []
        print("Doing Frame:", frame)
        T = "t{0:05d}".format(frame)
        path = "{}/{}/0/cells".format(T, channel)
        print("Loading the cube..")
        im = self.H5.get(path)
        return im

class RetinaNet:
    def __init__(self, modelpath, pixelextractor, backbone_name='resnet50'):
        self.model = models.load_model(modelpath, backbone_name=backbone_name)
        self.P = pixelextractor

    def preprocess(self, image):
        image = preprocess_image(image)
        image, scale = resize_image(image)
        return image, scale

    def process(self, image):
        # start = time.time()
        boxes, scores, labels = self.model.predict_on_batch(np.expand_dims(image, axis=0))
        # print("processing time: ", time.time() - start)
        return boxes, scores, labels

    def predictcube(self, tp):
        bar = progressbar.ProgressBar()
        resZ = {}
        resY = {}
        resX = {}

        cube = self.P.getCube(tp).value
        zdim = cube.shape
        for z in bar(range(1,cube.shape[2]-2)):
            im = cube[:,:,z-1:z+2]
            boxes, scores, labels = self.process(im)
            resZ[z] = {'boxes':boxes[0], 'scores':scores[0], 'labels':labels[0]}
        bar = progressbar.ProgressBar()
        # place Y axis at the End
        cube = np.rot90(cube).T
        ydim = cube.shape
        for y in bar(range(1,cube.shape[2]-2)):
            im = cube[:,:,y-1:y+2]
            boxes, scores, labels = self.process(im)
            resY[y] = {'boxes':boxes[0], 'scores':scores[0], 'labels':labels[0]}
        bar = progressbar.ProgressBar()
        # Place X axis at the end
        cube = np.rot90(cube.T).T
        xdim = cube.shape
        for x in bar(range(1,cube.shape[2]-2)):
            im = cube[:,:,x-1:x+2]
            boxes, scores, labels = self.process(im)
            resX[x] = {'boxes':boxes[0], 'scores':scores[0], 'labels':labels[0]}
            res = {
               'xdim':xdim,
               'ydim':ydim,
               'zdim':zdim,
               'X':resX,
               'Y':resY,
               'Z':resZ
               }
        return res

    def IoU_3D(self, anchors, thresh = 0.5, iou_thresh=0.1):
        anchors['X'] = self.IoU_max_Area_filter(anchors['X'], thresh=thresh, iou_thresh=iou_thresh)
        anchors['Y'] = self.IoU_max_Area_filter(anchors['Y'], thresh=thresh, iou_thresh=iou_thresh)
        anchors['Z'] = self.IoU_max_Area_filter(anchors['Z'], thresh=thresh, iou_thresh=iou_thresh)
        return anchors

    def mergePredictions(self, anchors, tp):
        cube = self.P.getCube(tp)
        Xpred = np.zeros(anchors['xdim'])
        Ypred = np.zeros(anchors['ydim'])
        Zpred = np.zeros(anchors['zdim'])
        for x in range(1, len(anchors['X'])-2):
            for bbox in anchors['X'][x]['boxes']:
                centers = self.makecenter(bbox)
                for xy in centers:
                    Xpred[xy[0]][xy[1]][x] = 1
        for y in range(1, len(anchors['Y'])-2):
            for bbox in anchors['Y'][y]['boxes']:
                centers = self.makecenter(bbox)
                for xy in centers:
                    Ypred[xy[0]][xy[1]][y] = 1
        for z in range(1, len(anchors['Z'])-2):
            for bbox in anchors['Z'][z]['boxes']:
                centers = self.makecenter(bbox)
                for xy in centers:
                    Zpred[xy[0]][xy[1]][z] = 1
        merged = Zpred.copy()
        merged *= Xpred.T
        merged *= np.rot90(Ypred.T)
        return Xpred, Ypred, Zpred, merged

    def makecenter(self, bbox):

        xc = int(round(bbox[1] + ((bbox[3] - bbox[1]) / 2)))
        yc = int(round(bbox[0] + ((bbox[2] - bbox[0]) / 2)))
        res = []
        for x in range(xc-2, xc+2):
            for y in range(yc-2, yc+2):
                res.append([x,y])
        return res

    def IoU_max_Area_filter(self, anchors, thresh = 0.5, iou_thresh=0.1):
        res = {}
        bar = progressbar.ProgressBar()
        for z in bar(anchors):
            G = nx.Graph()
            boxes, scores, labels = anchors[z]['boxes'], anchors[z]['scores'], anchors[z]['labels']
            filtered_boxes, filtered_scores, filtered_labels = [], [], []
            for i in range(len(boxes)):
                for j in range(i+1, len(boxes)):
                    if scores[i] > thresh:
                        b1 = boxes[i]
                        b2 = boxes[j]
                        iou = self.bb_intersection_over_union(b1, b2)
                        if iou > iou_thresh:
                            G.add_edge(i,j)
            CC = nx.connected_components(G)

            for C in CC:
                C = list(C)
                m = C[0]
                for x in C[1:]:
                    if scores[x] > scores[m]:
                        m = x
                filtered_boxes.append(boxes[m])
                filtered_scores.append(scores[m])
                filtered_labels.append(labels[m])
            res[z] = {'boxes':filtered_boxes, 'scores':filtered_scores, 'labels':filtered_labels}
        return res

    def drawcube(self, tp, anchors, thresh=0.3):
        stack = []
        cube = self.P.getCube(tp)
        bar = progressbar.ProgressBar()
        for z in bar(range(1,cube.shape[2]-2)):
            # print("Annotating Z:", z)
            boxes, scores, labels = anchors[z]['boxes'], anchors[z]['scores'], anchors[z]['labels']

            draw = cube[:,:,z-1:z+2]
            draw[:,:,0] = draw[:,:,1]
            draw[:,:,2] = draw[:,:,1]
            draw = draw/draw.max()
            draw = draw*255
            draw = draw.astype(np.uint8)
            draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
            # boxes /= scale
            for box, score, label in zip(boxes, scores, labels):
                # scores are sorted so we can break
                if score < thresh:
                    break
                color = label_color(8)
                b = box.astype(int)
                draw_box(draw, b, color=color)
            stack.append(draw)
        return np.array(stack)

    def bb_area(self, bbox):
        l = bbox[2] - bbox[0]
        w = bbox[3] - bbox[1]
        return l*w

    def bb_intersection_over_union(self, boxA, boxB):
    	# determine the (x, y)-coordinates of the intersection rectangle
    	xA = max(boxA[0], boxB[0])
    	yA = max(boxA[1], boxB[1])
    	xB = min(boxA[2], boxB[2])
    	yB = min(boxA[3], boxB[3])

    	# compute the area of intersection rectangle
    	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    	# compute the area of both the prediction and ground-truth
    	# rectangles
    	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    	# compute the intersection over union by taking the intersection
    	# area and dividing it by the sum of prediction + ground-truth
    	# areas - the interesection area
    	iou = interArea / float(boxAArea + boxBArea - interArea)

    	# return the intersection over union value
    	return iou

    def remove_keymap_conflicts(self, new_keys_set):
        for prop in plt.rcParams:
            if prop.startswith('keymap.'):
                keys = plt.rcParams[prop]
                remove_list = set(keys) & new_keys_set
                for key in remove_list:
                    keys.remove(key)

    def multi_slice_viewer(self, volume):
        self.remove_keymap_conflicts({'j', 'k'})
        fig, ax = plt.subplots(figsize=(10,10))
        ax.volume = volume
        ax.index = volume.shape[0] // 2
        ax.imshow(volume[ax.index])
        fig.canvas.mpl_connect('key_press_event', self.process_key)

    def process_key(self, event):
        fig = event.canvas.figure
        ax = fig.axes[0]
        if event.key == 'j':
            self.previous_slice(ax)
        elif event.key == 'k':
            self.next_slice(ax)
        fig.canvas.draw()

    def previous_slice(self, ax):
        volume = ax.volume
        ax.index = (ax.index - 1) % volume.shape[0]  # wrap around using %
        ax.images[0].set_array(volume[ax.index])

    def next_slice(self, ax):
        volume = ax.volume
        ax.index = (ax.index + 1) % volume.shape[0]
        ax.images[0].set_array(volume[ax.index])


if __name__=="__main__":

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


    P = PixelExtractor('../../Microscopy/JaneliaLightSheet/Bro1/Mamut/Bro1-fused2-mamut.xml')
    # P.MakeDeepLearningSet("/home/lblondel/Documents/Harvard/ExtavourLab/projects/Project_Parhyale/Microscopy/JaneliaLightSheet/Bro1/keras_retina2")
    R = RetinaNet('/home/lblondel/Documents/Harvard/ExtavourLab/projects/Project_Parhyale/Microscopy/JaneliaLightSheet/Bro1/keras_retina/models/resnet50_csv_29_infer.h5')
