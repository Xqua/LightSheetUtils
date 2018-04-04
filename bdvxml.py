class BDVXML:
    def __init__(self):
        self.xml = etree.Element('SpimData', version="0.2")
        self.doc = etree.ElementTree(self.xml)

        self.BasePath = etree.SubElement(self.xml, 'BasePath', type="relative")
        self.BasePath.text = "."

        self.SequenceDescription = etree.SubElement(self.xml, 'SequenceDescription')
        self.ImageLoader = etree.SubElement(self.SequenceDescription, 'ImageLoader', format="bdv.hdf5")


        self.ViewSetups = etree.SubElement(self.SequenceDescription, 'ViewSetups')
        self.ViewRegistrations = etree.SubElement(self.xml, 'ViewRegistrations')

        etree.SubElement(self.xml, "ViewInterestPoints")
        etree.SubElement(self.xml, "BoundingBoxes")
        etree.SubElement(self.xml, "PointSpreadFunctions")
        etree.SubElement(self.xml, "StitchingResults")

    def write(self, path):
        out = etree.tostring(root, pretty_print=True, xml_declaration=True, encoding='UTF-8')
        f = open(path, 'w')
        f.write(out)
        f.close()

    def addFile(self, path):
        image = etree.SubElement(self.ImageLoader, 'hdf5', type="relative")
        image.text = path

    def addviewsetup(self, Id, name, size, vosize_unit, vosize, illumination, channel, tile, angle):
        V = etree.SubElement(self.ViewSetups, 'ViewSetup')

        Id =  etree.SubElement(V, 'id')
        Id.text = Id
        name =  etree.SubElement(V, 'name')
        name.text = name
        size =  etree.SubElement(V, 'size')
        size.text = ' '.join(size)

        voxelSize =  etree.SubElement(V, 'voxelSize')
        unit =  etree.SubElement(voxelSize, 'unit')
        unit.text = vosize_unit
        size =  etree.SubElement(voxelSize, 'size')
        size.text = vosize

        attributes =  etree.SubElement(V, 'attributes')
        Ilum =  etree.SubElement(attributes, 'illumination')
        Ilum.text = illumination
        Chan =  etree.SubElement(attributes, 'channel')
        Chan.text = Chan
        Tile =  etree.SubElement(attributes, 'tile')
        Tile.text = tile
        Ang =  etree.SubElement(attributes, 'angle')
        Ang.text = angle

    def setViewSize(self, Id, size):
        trigger = False
        for child in self.ViewSetups:
            for el in child:
                if el.tag == 'id':
                    if el.text == Id:
                        trigger = True
                if el.tag == 'size' and trigger:
                    el.text = ' '.join(size)
                    trigger = False
                    return True
        return False

    def addAttributes(self, illuminations, channels, tiles, angles):
        illum = etree.SubElement(self.ViewSetups, 'Attributes', name="illumination")
        chan = etree.SubElement(self.ViewSetups, 'Attributes', name="channel")
        til = etree.SubElement(self.ViewSetups, 'Attributes', name="tile")
        ang = etree.SubElement(self.ViewSetups, 'Attributes', name="angle")

        for illumination in illuminations:
            I = etree.SubElement(illum, 'Illumination')
            Id = etree.SubElement(I, 'id')
            Id.text = illumination
            Name = etree.SubElement(I, 'name')
            Name.text = illumination

        for channel in channels:
            I = etree.SubElement(illum, 'Channel')
            Id = etree.SubElement(I, 'id')
            Id.text = channel
            Name = etree.SubElement(I, 'name')
            Name.text = channel

        for tile in tiles:
            I = etree.SubElement(illum, 'Tile')
            Id = etree.SubElement(I, 'id')
            Id.text = tile
            Name = etree.SubElement(I, 'name')
            Name.text = tile

        for angle in angles:
            I = etree.SubElement(illum, 'Angle')
            Id = etree.SubElement(I, 'id')
            Id.text = angle
            Name = etree.SubElement(I, 'name')
            Name.text = angle

    def addTimepoints(self, timepoints):
        TP = etree.SubElement(self.SequenceDescription, 'Timepoints', type="pattern")
        I = etree.SubElement(TP, 'integerpattern')
        I.text = ', '.join(timepoints)

    def addRegistration(self, tp, view):
        V = etree.SubElement(self.ViewRegistrations, 'ViewRegistration', timepoint=tp, setup=view)
        VT = etree.SubElement(V, 'ViewTransform', type="affine")
        name = etree.SubElement(VT, 'Name')
        name.text = "calibration"
        affine = etree.SubElement(VT, 'affine')
        affine.text = '1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 0.0'
