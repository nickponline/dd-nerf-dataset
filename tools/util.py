import numpy as np
from collections import namedtuple
import pymap3d
from xml.etree import ElementTree as ET
import os
import random
import pymap3d
import cv2
import open3d as o3d
import numpy as np
import click
import json
from pathlib import Path


def rotmat(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))

def closest_point_2_lines(oa, da, ob, db): # returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
    da = da / np.linalg.norm(da)
    db = db / np.linalg.norm(db)
    c = np.cross(da, db)
    denom = np.linalg.norm(c)**2
    t = ob - oa
    ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
    tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
    if ta > 0: ta = 0
    if tb > 0: tb = 0
    return (oa+ta*da+ob+tb*db) * 0.5, denom

def center_attention(frames):
    print("computing center of attention...")
    totw = 0.0
    totp = np.array([0.0, 0.0, 0.0])
    for f in frames:
        mf = f["transform_matrix"][0:3,:]
        for g in frames:
            mg = g["transform_matrix"][0:3,:]
            p, w = closest_point_2_lines(np.asarray(mf[:,3]).flatten(), np.asarray(mf[:,2]).flatten(), np.asarray(mg[:,3]).flatten(), np.asarray(mg[:,2]).flatten())
            if w > 0.01:
                totp += p*w
                totw += w
    totp /= totw
    print("center of attention: ", totp) # the cameras are looking at totp
    return totp


def encode(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return json.JSONEncoder.default(self, obj)

def undistort(image, camera):
    coefficients = np.array([
        camera.sensor.camera.k1,
        camera.sensor.camera.k2,
        camera.sensor.camera.p1,
        camera.sensor.camera.p2,
        camera.sensor.camera.k3,
        camera.sensor.camera.k4,
        0,
        0,
    ])
    undistorted = cv2.undistort(image, camera.sensor.camera.K, coefficients)
    return undistorted



# some simple class like objects
LLA = namedtuple('LLA', ['long', 'lat', 'alt'])
XY = namedtuple('XY', ['x', 'y'])
XYZ = namedtuple('XYZ', ['x', 'y', 'z'])
RPY = namedtuple('RPY', ['r', 'p', 'y'])
Ref = namedtuple('Reference', ('lla', 'rpy', 'enabled'))
Marker = namedtuple('Marker', ('pixel', 'camera'))
Covariance = namedtuple('Covariance', ('labels', 'M'))


class Transform(object):
    """ Object for storing ECEF to ENU transform"""
    def __init__(self, origin, R, T, s):
        """
        Args:
            origin (list): triple Lon, Lat, Alt for origin of ENU
            R (list): 9 element rotation matrix
            T (list): triple translation vector
            s (float): scale
        """
        self.origin = LLA(*origin)
        self.R = np.matrix(R).reshape((3, 3))
        self.Rinv = self.R.T
        self.T = np.array(T).reshape(3, 1)
        self.S = np.eye(3) * s
        self.Sinv = np.eye(3) * (1.0 / s)

    def ecef_to_enu(self, ecef):
        """ convert ecef to enu """
        return (self.Rinv * self.Sinv * (np.array(ecef).reshape(3, 1) - self.T)).reshape(3).tolist()[0]

    def enu_to_ecef(self, enu):
        """ convert enu to ecef """
        return (self.S * self.R * np.array(enu).reshape(3, 1) + self.T).reshape(3).tolist()[0]

    def lla_to_enu(self, lla):
        """
        Convert wgs84 to enu
        Args:
            lla (list): longitude, latitude, altitude
        """
        ecef = pymap3d.geodetic2ecef(lla[1], lla[0], lla[2])
        return self.ecef_to_enu(ecef)

    def enu_to_lla(self, enu):
        """
        Convert wgs84 to enu
        Args:
            enu (list): east, north, up
        """
        ecef = self.enu_to_ecef(enu)
        y, x, z = pymap3d.ecef2geodetic(ecef[0], ecef[1], ecef[2])
        x = x.tolist()
        y = y.tolist()
        z = z.tolist()
        return [x, y, z]

class GCP(object):
    """ Object for storing GCP info """
    def __init__(self, _id, label, ref, est, cameras):
        """
        Args:
            _id (int): GCP index
            label (str): GCP label
            ref (tuple): triple of reference longitude, latitude, altitude
            est (tuple): triple of estimated longitude, latitude, altitude
            cameras (list): list of Marker objects containing pixels and images
        """
        self.id = _id
        self.label = label
        self.reference = LLA(*ref)
        self.estimated = None if est is None else LLA(*est)
        self.cameras = cameras

    def is_checkpoint(self):
        """ returns True if GCP is a checkpoint """
        return 'checkpoint' in self.label.lower()


class CamerasXML(object):
    """ Parses and stores the data in the cameras XML """
    def __init__(self):
        self.sensors = {}
        self.cameras = {}
        self.transform = None
        self.gcps = {}

    @classmethod
    def read(cls, xml_file):
        """
        Parse the cameras.xml given a file name
        Args:
            xml_file (str): xml file path
        """
        xml = cls()
        doc = ET.parse(xml_file)
        root = doc.getroot()

        xml.parse_sensors(root)
        xml.parse_cameras(root)
        xml.parse_transform(root)
        xml.parse_gcps(root)
        return xml

    @classmethod
    def from_string(cls, xmlstr):
        """
        Parse the cameras.xml from a string
        Args:
            xml (str): xml string
        """
        xml = cls()
        root = ET.fromstring(xmlstr)

        xml.parse_sensors(root)
        xml.parse_cameras(root)
        xml.parse_transform(root)
        xml.parse_gcps(root)
        return xml

    def parse_sensors(self, root):
        """
        Parse the sensor section of the XML
        Generates a dictionary mapping sensor_id to sensor parameters
        Args:
            root (ElementTree): xml root element
        """
        self.sensors = {}

        # mapping of int value to camera type
        models = {0: PinholeCamera,
                  # 1: UR1Camera,
                  2: BrownCamera,
                  # 3: BrownFullCamera,
                  # 4: FisheyeFull,
                  # 5: FOVCamera,
                  # 6: SphericalCamera
                  }

        class Sensor(object):
            """ Simple object to store sensor info """
            def __init__(self, _id, _type, label):
                self.id = _id
                self.type = _type
                self.label = label
                self.resolution = None
                self.pixel_size = None
                self.focal_length = None
                self.camera = None
                self.covar = None

            @property
            def stddev(self):
                """ Returns the stddev of the covariance matrix diag """
                if self.covar is None:
                    return None
                return np.sqrt(np.diag(self.covar.M))

            @property
            def correlation(self):
                """ Returns the correlation matrix of the camera intrinsics of covar is valid """
                if self.covar is None:
                    return None

                d = self.stddev
                mask = d != 0
                d[mask] = 1./d[mask]
                d = np.repeat(d, self.covar.M.shape[0]).reshape(self.covar.M.shape)
                dd = d.T * d
                return dd * self.covar.M

            @classmethod
            def from_elementtree(cls, node):
                """
                Parse the sensor ojbect form an ElementTree
                Args:
                    node (ElementTree): sensor element tree
                """
                sen = cls(node.get('id'), node.get('type'), node.get('label'))

                # resolution
                res = node.find('./resolution')
                sen.resolution = XY(int(res.get('width')), int(res.get('height')))

                properties = {}
                for child in node.iter('property'):
                    properties[child.get('name')] = child.get('value')

                # pixel size
                if 'pixel_width' in properties and 'pixel_height' in properties:
                    sen.pixel_size = XY(float(properties.get('pixel_width')),
                                        float(properties.get('pixel_height')))

                # focal length
                if 'focal_length' in properties:
                    sen.focal_length = float(properties.get('focal_length'))

                distortion = {}
                size = None
                for child in node.find('./calibration'):
                    if child.tag != 'resolution':
                        distortion[child.tag] = float(child.text)
                    elif child.tag == 'resolution':
                        width = int(child.get('width', 0))
                        height = int(child.get('height', 0))
                        size = XY(width, height)

                # build k matrix
                skew = 0
                if 'skew' in distortion:
                    skew = distortion.pop('skew')

                fx = distortion.pop('fx') if 'fx' in distortion else 1.0
                fy = distortion.pop('fy') if 'fy' in distortion else 1.0
                cx = distortion.pop('cx') if 'cx' in distortion else 0.0
                cy = distortion.pop('cy') if 'cy' in distortion else 0.0

                K = np.matrix([[fx, skew, cx],
                               [0, fy, cy],
                               [0, 0, 1]])

                model = int(node.get('model'))
                if model == 6:  # SphericalCamera
                    sen.camera = models[model](K, size)
                else:
                    sen.camera = models[model](distortion, K, size)

                covar = node.find('./covariance')
                if covar is not None:
                    elem = covar.find('./labels')
                    labels = elem.text.split() if elem is not None else None
                    M = []
                    for r in covar.findall('./row'):
                        M.append([float(v) for v in r.text.split()])
                    M = np.array(M)
                    sen.covar = Covariance(labels, M)

                return sen

        # loop over all sensors in the list
        for sensor in root.findall('./chunk/sensors/sensor'):
            sen = Sensor.from_elementtree(sensor)
            self.sensors[sen.id] = sen

    def parse_cameras(self, root):
        """
        Parse the camera section of the XML file
        returns a mapping from camera_id (index) to camera data
        """
        class Camera(object):
            """ Object for storing the data about an image taken with a camera """
            def __init__(self, _id, label, _dir):
                self.id = _id
                self.label = label
                self.directory = _dir
                self.structured = None
                self.sensor = None
                self.orientation = None
                self.reference = None
                self.project = None
                self.depth = None
                self.covar = None

            @property
            def ag(self):
                return self.project is not None and not self.structured

            @classmethod
            def from_elementtree(cls, node, sensors):
                """
                Parse camera data from ElementTree
                Args:
                    node (ElementTree): xml node containing camera data
                    sensors (dict): mapping of sensor ids to Sensor objects
                """
                label = node.get('label')
                cam = cls(int(c.get('id')), os.path.basename(label), os.path.dirname(label))

                # structured
                cam.structured = c.get('enabled') == 'true'

                # sensor
                sensor_id = c.get('sensor_id')
                if sensor_id is not None:
                    cam.sensor = sensors[sensor_id]

                # orientation
                orientation = c.find('./orientation')
                cam.orientation = None
                if orientation is not None:
                    cam.orientation = int(orientation.text)

                # raw image meta data
                ref = c.find('./reference')
                if ref is not None:
                    x = float(ref.get('x', '0'))
                    y = float(ref.get('y', '0'))
                    z = ref.get('z')
                    # camera could not have altitude set
                    if z is not None:
                        z = float(z)
                    ref_lla = LLA(x, y, z)
                    if 'roll' in ref.attrib:
                        ref_rpy = RPY(float(ref.get('roll')),
                                      float(ref.get('pitch')),
                                      float(ref.get('yaw')))
                    else:
                        ref_rpy = None
                    ref_enabled = ref.get('enabled') == '1'
                    cam.reference = Ref(ref_lla, ref_rpy, ref_enabled)

                # transform
                trans = c.find('./transform')
                if trans is not None:
                    pose = [float(v) for v in trans.text.split(' ')]
                    pose = np.reshape(np.matrix(pose), (4, 4))
                    cam.project = Projector(cam.sensor.camera, pose)

                depth = c.find('./depth')
                if depth is not None:
                    cam.depth = float(depth.text)

                covar = node.find('./covariance')
                if covar is not None:
                    elem = covar.find('./labels')
                    labels = elem.text.split() if elem is not None else None
                    M = []
                    for r in covar.findall('./row'):
                        M.append([float(v) for v in r.text.split()])
                    M = np.array(M)
                    cam.covar = Covariance(labels, M)

                return cam

        self.cameras = {}
        for c in root.findall('./chunk/cameras/camera'):
            cam = Camera.from_elementtree(c, self.sensors)
            self.cameras[cam.id] = cam

    def parse_transform(self, root):
        """ Parses the transform part of the XML file """
        self.transform = None

        transform = root.find('./chunk/transform')

        # scene is not geo-referenced
        if transform is None:
            return

        # Rotation matrix
        rotation = transform.find('./rotation')
        R = [float(x) for x in rotation.text.split(' ')]

        # translation matrix
        translation = transform.find('./translation')
        T = [float(x) for x in translation.text.split(' ')]

        # scale value
        s = float(transform.find('./scale').text)

        # origin (if it has one)
        try:
            origin = LLA(*[float(x) for x in transform.find('./origin').text.split(',')])
        except ValueError:
            origin = LLA(0, 0, 0)

        self.transform = Transform(origin, R, T, s)


    def parse_gcps(self, root):
        """
        Parses the gcp section of the XML file
        generates a mapping from morker id (index) to marker data
        """

        def from_elementtree(node, frame_markers, cameras):
            """
            Parse gcp data from ElmeentTree
            Args:
                node (ElementTree): node with GCP data
                frame_markers (ElementTree): node containing frame/markers
                cameras (dict): mapping from camera id to Camera object
            """
            _id = int(node.get('id'))
            label = node.get('label')

            # reference WGS84 coordinate
            reference = node.find('./reference')
            ref = LLA(float(reference.get('x')),
                      float(reference.get('y')),
                      float(reference.get('z')))

            # estimated WGS84 coordinate if it exists
            estimated = node.find('./estimated')
            est = None
            if estimated is not None:
                est = LLA(float(estimated.get('x')),
                          float(estimated.get('y')),
                          float(estimated.get('z')))

            markers = []
            for location in frame_markers.findall('.//*[@marker_id="{}"]/location'.format(_id)):
                # camera ID
                camera_id = int(location.get('camera_id'))

                # pixel location
                pixel = XY(float(location.get('x')), float(location.get('y')))

                # refrence to camera in camera section of XML
                camera = cameras.get(camera_id)
                markers.append(Marker(pixel, camera))

            markers.sort(key=lambda x: x.camera.id)

            return GCP(_id, label, ref, est, markers)

        self.gcps = {}

        markers = {}
        for g in root.findall('./chunk/markers/marker'):
            gcp = from_elementtree(g, root.find('./chunk/frames/frame/markers'), self.cameras)
            markers[gcp.id] = gcp

        self.gcps = markers


def read_points_as_numpy(filename):
    with laspy.file.File(filename, mode='r') as f:
        data = np.vstack([
            f.x,
            f.y,
            f.z,
            f.red   / 256,
            f.green / 256,
            f.blue  / 256,
        ])

        return data.T

def read_numpy(filename):
    """ Load and return a numpy.array from file """
    points = np.load(filename)
    try:
        points = points[ points[:,2].argsort() ]
    except:
        print("Could not sort")
    return points

def read_pointcloud(camerasfile, filename, dont_convert=False):
    """ Convert .las in wgs84 to .npy in enu """

    points = read_points_as_numpy(filename)
    cameras = CamerasXML().read(camerasfile)

    # Use the transform from the camera.xml to bring points into camera coordinate system
    ecef = pymap3d.geodetic2ecef(points[:, 1], points[:, 0], points[:, 2])
    ecef = np.vstack(ecef).T
    Rinv = cameras.transform.Rinv
    Sinv = cameras.transform.Sinv
    T = cameras.transform.T.T

    points[:, :3] = Rinv.dot(Sinv).dot( (ecef - T).T ).T

    return points

class Projector(object):
    def __init__(self, sensor, pose):
        self.sensor = sensor
        self.pose = pose

    def to_image(self, point, distort=True):
        """ project point into the image """
        return self.sensor.project(self.pose, point, distort=distort)

    def position(self):
        """ return the position of the camera """
        Rinv = self.pose[0:3, 0:3].T
        C = -Rinv.dot(self.pose[0:3, 3])
        return C.reshape(3).tolist()[0]

    def orientation(self):
        """ return the position of the camera """
        Rinv = self.pose[0:3, 0:3].T
        return Rinv

    def look(self):
        """ return the look vector """
        return self.pose[0:3, 2].reshape(3).tolist()

    def up(self):
        """ return the up vector """
        return self.pose[0:3, 1].reshape(3).tolist()

    def right(self):
        """ return the up vector """
        return self.pose[0:3, 0].reshape(3).tolist()



    def pose_from_RC(self, C, Rc):
        C = np.array(C)
        R = Rc.T
        t = -R.dot(C)
        self.pose = np.vstack([
            np.hstack([R, t.reshape(3, 1)]),
            np.array([0, 0, 0, 1]),
        ])



class Camera(object):
    def __init__(self, name, K, size):
        self.name = name
        self.K = K
        self.size = size

    def c2i(self, x):
        x[:, 0] = self.K[0, 2] + self.K[0, 0] * x[:, 0]
        x[:, 1] = self.K[1, 2] + self.K[1, 1] * x[:, 1]
        return x

    def i2cp(self, x):
        return x


class PinholeCamera(Camera):
    def __init__(self, K, size):
        super(PinholeCamera, self).__init__('Pinhole', K, size)

    def distort(self, x, distort=True):
        return x

    def project(self, pose, point, distort=True):
        p = np.hstack( (point, np.ones((point.shape[0], 1)))).T
        x = pose * p
        x = x.T
        x = np.asarray(x)
        x[:, 0] /= x[:, 2]
        x[:, 1] /= x[:, 2]
        x = x[:, :2]
        return self.c2i(self.distort(x[:, :2], distort=distort))
        return x

class BrownCamera(PinholeCamera):
    def __init__(self, params, K, size):
        super(BrownCamera, self).__init__(K, size)
        self.name = 'Brown'
        self.k1 = params.get('k1', 0)
        self.k2 = params.get('k2', 0)
        self.k3 = params.get('k3', 0)
        self.k4 = params.get('k4', 0)
        self.p1 = params.get('p1', 0)
        self.p2 = params.get('p2', 0)

    @property
    def params(self):
        return {'k1': self.k1,
                'k2': self.k2,
                'k3': self.k3,
                'k4': self.k4,
                'p1': self.p1,
                'p2': self.p2,
                }

    def distort(self, x, distort=True):

        if not distort:
            return x
        # if points are far outside the camera they colud wrap back into the camera
        x2 = x[:, 0] * x[:, 0]
        y2 = x[:, 1] * x[:, 1]
        xy = x[:, 0] * x[:, 1]
        r2 = x2 + y2
        coeff = 1.0 + r2 * (self.k1 + r2 * (self.k2 + r2 * (self.k3 + r2 * self.k4)))
        x[:, 0] = x[:, 0] * coeff + self.p1 * xy * 2.0 + self.p2 * (r2 + x2 * 2.0)
        x[:, 1] = x[:, 1] * coeff + self.p1 * (r2 + y2 * 2.0) + self.p2 * xy * 2.0
        return x

    def i2cp(self, x):
        xp = (x[0] - self.K[0, 2]) / self.K[0, 0]
        yp = (x[1] - self.K[1, 2]) / self.K[1, 1]
        return (xp, yp)
