import xmltodict
import h5py
import networkx as nx
import os
from scipy.optimize import curve_fit
from scipy.stats import multivariate_normal
import numpy as np

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
                G.add_node(ID, attr_dict={"x": X, "y": Y, "z": Z, "frame": c, "radius": float(radius), "color":color})
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
        b, n = np.histogram(R)
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

    def PredictOneFrame(self, frame):
        dim = self.BDVXML["SpimData"]["SequenceDescription"]["ViewSetups"]["ViewSetup"][0]["size"]
        dim = dim.split(' ')
        dim = [int(i) for i in [dim[2], dim[1], dim[0]]]
        M = np.zeros(dim)
        nodes = [i for i in self.G.nodes() if self.G.node[i]['frame'] == frame]
        for node in nodes:
            N = self.G.node[node]
            X, Y, Z = int(N['x']), int(N['y']), int(N['z'])
            cube = self.ExtractCube('s17', node)
            try:
                mu, sigma, A, res = self.FitGaussian(cube)
                T = self.Segment(cube, res)
                for i in range(T.shape[0]):
                    for j in range(T.shape[1]):
                        for k in range(T.shape[2]):
                            M[Z + (i - int(T.shape[0]/2))][Y+(j - int(T.shape[1]/2))][X+(k - int(T.shape[2]/2))] = T[i][j][k]
            except:
                print("No gaussian in cube !")
        return M

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
        if not P.H5.get(channel):
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

    def MakeDeepLearningSet(self, outfile, channel='s17', size=100000, split=[0.7,0.2,0.1], img_size=[500,600], geometric=None, maxframe=None):
        """The size parameter regulates how many samples are generated in total.
        The split parameter controls the proportion that goes into train, test, and validation set.
        channel is indicating the channel to extract the images from the BDV file.
        outfile is the output H5 file.
        img_size is the output image size
        geometric is a list of a serie of geometric transform, ie transpose or rotate (np.rot90), it can be an arbitrary serie of transform.
        maxframe can limit the frame extraction to a max frame number, by default it goes accross the whole dataset."""
        trainN = int(size * split[0])
        testN = int(size * split[1])
        validateN = int(size * split[2])
        framedone = [] # This serve to verify that we aren't sampling twice the same image
        self.SetDimensions()

        if not maxframe:
            maxframe = self.stopframe

        # Define our h5 architecture:
        # 3 first level train test validate
        # each contains the image info X and the label output Y
        print("Creating the output dataset...")
        h5file = h5py.File(outfile, 'w')
        train = h5file.create_group('/train')
        train_images = train.create_dataset("images", shape=[trainN, img_size[1], img_size[0], 3], chunks=True, dtype=np.int16, compression="gzip")
        train_labels = train.create_dataset("labels", shape=[trainN, img_size[1], img_size[0], 3], chunks=True, dtype=np.int8, compression="gzip")
        test = h5file.create_group('/test')
        test_images = test.create_dataset("images", shape=[testN, img_size[1], img_size[0], 3], chunks=True, dtype=np.int16, compression="gzip")
        test_labels = test.create_dataset("labels", shape=[testN, img_size[1], img_size[0], 3], chunks=True, dtype=np.int8, compression="gzip")
        validate = h5file.create_group('/validate')
        validate_images = validate.create_dataset("images", shape=[validateN, img_size[1], img_size[0], 3], chunks=True, dtype=np.int16, compression="gzip")
        validate_labels = validate.create_dataset("labels", shape=[validateN, img_size[1], img_size[0], 3], chunks=True, dtype=np.int8, compression="gzip")
        # h5file.close()

        minperframe = int(size / maxframe)
        print("Number of sample per frame: ", minperframe)

        train_i, test_i, validate_i = 0, 0, 0

        print("Starting the extraction")
        for frame in range(maxframe):
            # h5file = h5py.File(outfile, 'a')
            # train_images = h5file.get("/train/images")
            # train_labels = h5file.get("/train/labels")
            # test_images = h5file.get("/test/images")
            # test_labels = h5file.get("/test/labels")
            # validate_images = h5file.get("/validate/images")
            # validate_labels = h5file.get("/validate/labels")
            DataSet = []
            print("Doing Frame:", frame)
            T = "t{0:05d}".format(frame)
            path = "{}/{}/0/cells".format(T, channel)
            print("Loading the cube..")
            im = self.H5.get(path)
            print("Creating the label cube...")
            M = self.PredictOneFrame(frame)
            print("Applying geometric transforms")
            if geometric:
                for action in geometric:
                    if action == 'transpose':
                        im = im.T
                        M = M.T
                    elif action == 'rotate':
                        im = np.rot90(im)
                        M = np.rot90(M)
            if im.shape[1] > img_size[0] or im.shape[2] > img_size[1]:
                print("ERROR: Image size (img_size: {}, {}) is smaller than real image: {}, {} ".format(img_size[0], img_size[1], im.shape[1], im.shape[2]))
                print("Either you need to apply geometric transforms using the transform list, or increase your img_size.")
            # Extract dataset
            # samplenumber = np.random.randint(minperframe, im.shape[0])

            Zs = []
            for j in range(minperframe):
                z = np.random.randint(1, im.shape[0]-1)
                while z in Zs:
                    z = np.random.randint(1, im.shape[0]-1)
                print("Grabbing Z:", z)
                tmp_img = np.zeros((3, img_size[0], img_size[1]))
                tmp_img[0:3, 0:im.shape[1], 0:im.shape[2]] = im[z-1:z+2]

                tmp_lab = np.zeros((3, img_size[0], img_size[1]))
                tmp_lab[0:3, 0:M.shape[1], 0:M.shape[2]] = M[z-1:z+2]

                p = np.random.randint(4)
                if p == 0:
                    tmp_img = np.fliplr(tmp_img)
                    tmp_lab = np.fliplr(tmp_lab)
                elif p == 1:
                    tmp_img = np.flipud(tmp_img)
                    tmp_lab = np.flipud(tmp_lab)
                elif p == 2:
                    tmp_img = np.fliplr(tmp_img)
                    tmp_lab = np.fliplr(tmp_lab)
                    tmp_img = np.flipud(tmp_img)
                    tmp_lab = np.flipud(tmp_lab)

                DataSet.append([tmp_img.T.astype(np.uint16), tmp_lab.T.astype(np.uint8)])

            DataSet = np.array(DataSet)
            np.random.shuffle(DataSet)

            trainN = int(len(DataSet) * split[0])
            testN = int(len(DataSet) * split[1])
            validateN = int(len(DataSet) * split[2])

            train_images[train_i:train_i+trainN] = DataSet[0:trainN, 0, :, :, :]
            train_labels[train_i:train_i+trainN] = DataSet[0:trainN, 1, :, :, :]
            test_images[test_i:test_i+testN] = DataSet[trainN:trainN+testN, 0, :, :, :]
            test_labels[test_i:test_i+testN] = DataSet[trainN:trainN+testN, 1, :, :, :]
            validate_images[validate_i:validate_i+validateN] = DataSet[trainN+testN:trainN+testN+validateN, 0, :, :, :]
            validate_labels[validate_i:validate_i+validateN] = DataSet[trainN+testN:trainN+testN+validateN, 1, :, :, :]

            train_i += trainN
            test_i += testN
            validate_i += validateN

            del(M)
            del(DataSet)
        h5file.close()


        # print("Splitting the dataset into training testing and validation")
        #
        # train_images[:] = DataSet[0:trainN, 0, :, :, :]
        # train_labels[:] = DataSet[0:trainN, 1, :, :, :]
        # test_images[:] = DataSet[trainN:trainN+testN, 0, :, :, :]
        # test_labels[:] = DataSet[trainN:trainN+testN, 1, :, :, :]
        # validate_images[:] = DataSet[trainN+testN:trainN+testN+validateN, 0, :, :, :]
        # validate_labels[:] = DataSet[trainN+testN:trainN+testN+validateN, 1, :, :, :]

        # Let's start to generate our sets
        # we iterate of the number of training images we need
        #
        #
        # train_i, test_i, val_i = 0,0,0
        #
        # for i in range(len(DataSet)):
        #     if i < trainN:
        #         train_images[train_i] = DataSet[i][0]
        #         train_labels[train_i] = DataSet[i][1]
        #         train_i += 1
        #     elif i < trainN+testN:
        #         test_images[test_i] = DataSet[i][0]
        #         test_labels[test_i] = DataSet[i][1]
        #         test_i += 1
        #     elif i < trainN+testN+validateN:
        #         validate_images[val_i] = DataSet[i][0]
        #         validate_labels[val_i] = DataSet[i][1]
        #         val_i += 1
        #     else:
        #         break

        print("DONE !")


    def GetRandomImage3Coord(self):
        frame = np.random.randint(0, self.stopframe)
        Z = np.random.randint(0, dim[0])
