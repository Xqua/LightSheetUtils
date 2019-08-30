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

    def Normal3DPDF(self, v, sigma):
        N = 3
        n = np.linalg.norm(v)
        G = 1/((np.sqrt(2*np.pi) * sigma) ** N)
        G *= np.exp( - ( n**2 / ( 2 * ( sigma**2 ) ) ) )
        return G

    def Normal3DKernel(self, size, sigma):
        M = np.zeros((size*2 +1, size*2 +1, size*2 + 1))
        ci = 0
        for i in range(-size,size+1):
            cj = 0
            for j in range(-size,size+1):
                ck = 0
                for k in range(-size,size+1):
                    v = [i,j,k]
                    G = Normal3DPDF(v, sigma)
                    M[ci][cj][ck] = G
                    ck += 1
                cj +=1
            ci += 1
        return M

    def Gaussian3DKernel(self, size, mu, sigma, Range):
        """Takes a mean vector mu of shape (3), and a covariance matrix (3x3).
        It returns the kernel of shape fixed by the size parameter (size is the slice size).
        The Range parameter is a 3x2 matrix that describes the min and max value to sample from."""
        M = []
        X = np.linspace(Range[0][0], Range[0][1], size)
        Y = np.linspace(Range[1][0], Range[1][1], size)
        Z = np.linspace(Range[2][0], Range[2][1], size)
        for x in X:
            N = []
            for y in Y:
                O = []
                for z in Z:
                    v = np.array([x,y,z])
                    G = Gaussian3DPDF(v, mu, sigma)
                    O.append(G)
                N.append(O)
            M.append(N)
        M = np.array(M)
        return M

    def Gaussian3DPDFFunc(self, X, mu0, mu1, mu2, s0, s1, s2, A):
        mu = np.array([mu0, mu1, mu2])
        Sigma = np.array([[s0, 0, 0], [0, s1, 0], [0, 0, s2]])
        res = multivariate_normal.pdf(X, mean=mu, cov=Sigma)
        res *= A
        res += 100
        return res

    def FitGaussian(self, cube):
        # prepare the data for curvefit
        X = []
        Y = []
        for i in range(cube.shape[0]):
            for j in range(cube.shape[1]):
                for k in range(cube.shape[2]):
                    X.append([i,j,k])
                    Y.append(cube[i][j][k])
        bounds = [[3,3,3,3,3,3,50], [cube.shape[0] - 3, cube.shape[1] - 3, cube.shape[2] - 3, 30, 30, 30, 100000]]
        p0 = [cube.shape[0]/2, cube.shape[1]/2, cube.shape[2]/2, 10, 10, 10, 100]
        popt, pcov = curve_fit(self.Gaussian3DPDFFunc, X, Y, p0, bounds=bounds)
        mu = [popt[0], popt[1], popt[2]]
        sigma = [[popt[3], 0, 0], [0, popt[4], 0], [0, 0, popt[5]]]
        A = popt[6]
        res = multivariate_normal.pdf(X, mean=mu, cov=sigma)
        return mu, sigma, A, res

    def Segment(self, cube, r):
        R = r.reshape(cube.shape)
        R *= cube
        b, n = np.histogram(R, bins=1000)
        thresh = n[1]
        T = R * (R > thresh)
        T = 1 * (T>0)
        return T

    def Test(self, cube):
        mu, sigma, A, res = FitGaussian(cube)
        print(mu, sigma)
        T = Segment(cube, res)
        multi_slice_viewer(cube)
        multi_slice_viewer(T)
        plt.show()

    def bbox(self, img, min_spread=2):
        a = np.where(img != 0)
        if (a[0].sum() != 0) and (a[1].sum() != 0):
            bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
            if (bbox[1] - bbox[0] <= min_spread) or (bbox[3] - bbox[2]  <= min_spread):
                bbox = None
        else:
            bbox = None
        return bbox

    def ShowOneSliceFrame(self, frame, Z, annotations, channel='s17'):
        if Z not in annotations:
            print("No annotation for this slice !")
            annotations[Z] = []
        T = "t{0:05d}".format(frame)
        path = "{}/{}/0/cells".format(T, channel)
        # print("Loading the cube..")
        C = self.H5.get(path)
        im = np.zeros((C.shape[1], C.shape[2], 3))
        tmp = C[Z][:,:]
        im[:,:,0] = tmp
        im[:,:,1] = tmp
        im[:,:,2] = tmp
        im = ((im-im.min())/im.max())*255
        im = im.astype(np.uint8)

        fig,ax = plt.subplots(1)
        # ax.imshow(im)
        # leg = np.zeros((im.shape[0],im.shape[1],3), np.uint8)
        for bbox in annotations[Z]:
            cv2.rectangle(im,(bbox[0],bbox[2]),(bbox[1],bbox[3]),(0,255,0),1)
        ax.imshow(im)
        plt.show()

    def PredictOneFrame(self, frame):
        dim = self.BDVXML["SpimData"]["SequenceDescription"]["ViewSetups"]["ViewSetup"][0]["size"]
        dim = dim.split(' ')
        dim = [int(i) for i in [dim[2], dim[1], dim[0]]]
        M = np.zeros(dim)
        Annotations = {}
        nodes = [i for i in self.G.nodes() if self.G.node[i]['frame'] == frame]
        for node in nodes:
            N = self.G.node[node]
            X, Y, Z = int(N['x']), int(N['y']), int(N['z'])
            cube = self.ExtractCube('s17', node)
            try:
                mu, sigma, A, res = self.FitGaussian(cube)
                sigma = np.diag(sigma)
                T = self.Segment(cube, res)
                for i in range(len(T)): # from mu_z - 2sigma to mu_z + 2sigma
                    bbox = self.bbox(T[i])
                    if bbox:
                        z = Z + (i - int(T.shape[0]/2))
                        xmin, xmax = X + (bbox[0] - int(T.shape[2]/2)), X + (bbox[1] - int(T.shape[2]/2))
                        ymin, ymax = Y + (bbox[2] - int(T.shape[2]/2)), Y + (bbox[3] - int(T.shape[2]/2))
                        if not z in Annotations:
                            Annotations[z] = []
                        Annotations[z].append([xmin, xmax, ymin, ymax])
                # for i in range(T.shape[0]):
                #     for j in range(T.shape[1]):
                #         for k in range(T.shape[2]):
                #             M[Z + (i - int(T.shape[0]/2))][Y+(j - int(T.shape[1]/2))][X+(k - int(T.shape[2]/2))] = T[i][j][k]
            except:
                print("No gaussian in cube !")
        return Annotations

    def MakePredictionMap(self):
        dim = self.BDVXML["SpimData"]["SequenceDescription"]["ViewSetups"]["ViewSetup"][0]["size"]
        dim = dim.split(' ')
        dim = [int(i) for i in [self.stopframe, dim[2], dim[1], dim[0] ] ]
        h5file = h5py.File("PredictionMap.h5",'w')
        grp = h5file.create_group('/dataset')
        res = grp.create_dataset("images", shape=dim, chunks=True, dtype=np.bool, compression="gzip")
        for f in range(self.stopframe):
            m = self.PredictOneFrame(f)
            res[f] = m.astype(np.bool)
            break

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

    def MakeDeepLearningSet(self, outfile, channel='s17', size=100000, split=[0.8,0.2], img_size=[549, 483], maxframe=None):
        """The size parameter regulates how many samples are generated in total.
        The split parameter controls the proportion that goes into train, test, and validation set.
        channel is indicating the channel to extract the images from the BDV file.
        outfile is the output H5 file.
        img_size is the output image size
        maxframe can limit the frame extraction to a max frame number, by default it goes accross the whole dataset."""
        trainN = int(size * split[0])
        testN = int(size * split[1])
        framedone = [] # This serve to verify that we aren't sampling twice the same image
        self.SetDimensions()

        if not maxframe:
            maxframe = self.stopframe

        # Define our h5 architecture:
        # 3 first level train test validate
        # each contains the image info X and the label output Y
        print("Creating the output dataset folders...")
        if not os.path.isdir(outfile):
            os.mkdir(outfile)
            os.mkdir(os.path.join(outfile, 'train'))
            os.mkdir(os.path.join(outfile, 'train', 'images'))
            os.mkdir(os.path.join(outfile, 'train', 'labels'))
            os.mkdir(os.path.join(outfile, 'test'))
            os.mkdir(os.path.join(outfile, 'test', 'images'))
            os.mkdir(os.path.join(outfile, 'test', 'labels'))


        minperframe = int(size / maxframe)
        print("Number of sample per frame: ", minperframe)

        train_i, test_i, validate_i = 0, 0, 0
        f = open(os.path.join(outfile, 'train', 'train.csv'), 'w')
        f.close()
        f = open(os.path.join(outfile, 'test', 'test.csv'), 'w')
        f.close()

        print("Starting the extraction")
        for frame in range(maxframe):

            DataSet = []
            print("Doing Frame:", frame)
            T = "t{0:05d}".format(frame)
            path = "{}/{}/0/cells".format(T, channel)
            print("Loading the cube..")
            im = self.H5.get(path)
            print("Creating the label cube...")
            A = self.PredictOneFrame(frame)
            print("Applying geometric transforms")

            if im.shape[1] > img_size[0] or im.shape[2] > img_size[1]:
                print("ERROR: Image size (img_size: {}, {}) is smaller than real image: {}, {} ".format(img_size[0], img_size[1], im.shape[1], im.shape[2]))
                print("Either you need to apply geometric transforms using the transform list, or increase your img_size.")
            # Extract dataset

            Zs = []
            for j in range(minperframe):
                if len(Zs) >= len(A):
                    break
                z = np.random.randint(1, im.shape[0]-1)
                # print(A.keys())
                while (z in Zs) or (z not in A.keys()):
                    z = np.random.randint(1, im.shape[0]-1)
                print("Grabbing Z:", z)
                Zs.append(z)
                tmp_img = np.zeros((img_size[0], img_size[1], 3))
                tmp_img[:,:,0] = im[z-1]
                tmp_img[:,:,1] = im[z]
                tmp_img[:,:,2] = im[z+1]

                DataSet.append([tmp_img.astype(np.uint16), A[z]])

            DataSet = np.array(DataSet)
            np.random.shuffle(DataSet)

            trainN = int(len(DataSet) * split[0])
            testN = int(len(DataSet) * split[1])

            trainCSV = []
            for i in range(trainN):
                path = os.path.join(outfile, 'train', 'images')
                path2 = os.path.join(outfile, 'train', 'labels')
                filepath = os.path.join(path, "train_{0:04d}_{1:06d}.png".format(frame, i))
                filepath2 = os.path.join(path2, "train_annot_{0:04d}_{1:06d}.png".format(frame, i))
                im = DataSet[i][0]
                im2 = im.copy()
                im2 = ((im2-im2.min())/im2.max())*255
                im2 = im2.astype(np.uint8)
                annots = DataSet[i][1]
                for annot in annots:
                    cv2.rectangle(im2,(annot[0],annot[2]),(annot[1],annot[3]),(0,255,0),1)
                    trainCSV.append([filepath] + annot)
                misc.imsave(filepath, im)
                misc.imsave(filepath2, im2)


            testCSV = []
            for i in range(trainN, trainN + testN):
                path = os.path.join(outfile, 'test', 'images')
                path2 = os.path.join(outfile, 'test', 'labels')
                filepath = os.path.join(path, "test_{0:04d}_{1:06d}.png".format(frame, i))
                filepath2 = os.path.join(path2, "test_annot_{0:04d}_{1:06d}.png".format(frame, i))
                im = DataSet[i][0]
                im2 = im.copy()
                im2 = ((im2-im2.min())/im2.max())*255
                im2 = im2.astype(np.uint8)
                annots = DataSet[i][1]
                for annot in annots:
                    cv2.rectangle(im2,(annot[0],annot[2]),(annot[1],annot[3]),(0,255,0),1)
                    testCSV.append([filepath] + annot)
                misc.imsave(filepath, im)
                misc.imsave(filepath2, im2)

            f = open(os.path.join(outfile, 'train', 'train.csv'), 'a')
            for l in trainCSV:
                f.write(','.join([str(i) for i in l]) + '\n')
            f.close()

            f = open(os.path.join(outfile, 'test', 'test.csv'), 'a')
            for l in testCSV:
                f.write(','.join([str(i) for i in l]) + '\n')
            f.close()

            train_i += trainN
            test_i += testN

            del(A)
            del(DataSet)

        print("DONE !")

    def GetRandomImage3Coord(self):
        frame = np.random.randint(0, self.stopframe)
        Z = np.random.randint(0, dim[0])


if __name__=="__main__":
    P = PixelExtractor('../../Microscopy/JaneliaLightSheet/Bro1/Mamut/Bro1-fused2-mamut.xml')
    P.MakeDeepLearningSet("/home/lblondel/Documents/Harvard/ExtavourLab/projects/Project_Parhyale/Microscopy/JaneliaLightSheet/Bro1/keras_retina2")
