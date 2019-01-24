import argparse
import numpy as np
import yaml

import meshcat.transformations as tf
# from iterative_closest_point import RunICP
# from visualization_utils import ThresholdArray

from pydrake.util.eigen_geometry import Isometry3
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import (AbstractValue, BasicVector,
                                       DiagramBuilder, LeafSystem,
                                       PortDataType)
from pydrake.examples.manipulation_station import (
    ManipulationStation, ManipulationStationHardwareInterface)

import numpy as np
import yaml

from pydrake.systems.framework import AbstractValue, LeafSystem
from pydrake.common.eigen_geometry import Isometry3, AngleAxis
from pydrake.systems.sensors import CameraInfo, ImageRgba8U
import pydrake.perception as mut
from pydrake.systems.sensors import PixelType

import meshcat.transformations as tf

class PoseRefinement(LeafSystem):

    def __init__(self, config_file, model_file, viz=False, segment_scene_function=None,
                 get_pose_function=None):
        """
        A system that takes in a point cloud, an initial pose guess, and an
        object model and calculates a refined pose of the object. The user can
        optionally supply a custom segmentation function and pose alignment
        function used to determine the pose. If these functions aren't supplied,
        the default functions in this class will be used.

        # TODO(kmuhlrad): add something about frames

        @param model_file str. A path to a .npy file containing the object mesh.
        @param model_file str. A path to a .npy file containing the object mesh.
        @param viz bool. If True, save the aligned and segmented point clouds
            as serialized numpy arrays.
        @param segment_scene_function A Python function that returns a subset of
            the scene point cloud. See self.SegmentScene for more details.
        @param get_pose_function A Python function that calculates a pose from a
            segmented point cloud. See self.GetPose for more details.

        @system{
          @input_port{point_cloud},
          @input_port{X_WObject_guess},
          @output_port{X_WObject_refined}
        }
        """
        LeafSystem.__init__(self)

        self.point_cloud_port = self._DeclareAbstractInputPort(
            "point_cloud", AbstractValue.Make(mut.PointCloud()))
        self.init_pose_port = self._DeclareAbstractInputPort(
            "X_WObject_guess", AbstractValue.Make(Isometry3))

        self._DeclareAbstractOutputPort("X_WObject_refined",
                                        lambda: AbstractValue.Make(
                                            Isometry3.Identity()),
                                        self._DoCalcOutput)

        self.segment_scene_function = segment_scene_function
        self.get_pose_function = get_pose_function
        self.model = np.load(model_file)

        self._LoadConfigFile(config_file)

        self.viz = viz

    def _LoadConfigFile(self, config_file):
        with open(config_file, 'r') as stream:
            try:
                config = yaml.load(stream)
                self.camera_configs = {}
                for camera in config:
                    serial_no, X_WCamera, X_CameraDepth, camera_info = \
                        self._ParseCameraConfig(config[camera])
                    self.camera_configs[camera + "_serial"] = str(serial_no)
                    self.camera_configs[camera + "_pose_world"] = X_WCamera
                    self.camera_configs[camera + "_pose_internal"] = \
                        X_CameraDepth
                    self.camera_configs[camera + "_info"] = camera_info
            except yaml.YAMLError as exc:
                print "could not parse config file"
                print exc

    def _ParseCameraConfig(self, camera_config):
        # extract serial number
        serial_no = camera_config["serial_no"]

        # construct the transformation matrix
        world_transform = camera_config["world_transform"]
        X_WCamera = tf.euler_matrix(world_transform["roll"],
                                    world_transform["pitch"],
                                    world_transform["yaw"])
        X_WCamera[:3, 3] = \
            [world_transform["x"], world_transform["y"], world_transform["z"]]

        # construct the transformation matrix
        internal_transform = camera_config["internal_transform"]
        X_CameraDepth = tf.euler_matrix(internal_transform["roll"],
                                        internal_transform["pitch"],
                                        internal_transform["yaw"])
        X_CameraDepth[:3, 3] = ([internal_transform["x"],
                                 internal_transform["y"],
                                 internal_transform["z"]])

        # construct the camera info
        camera_info_data = camera_config["camera_info"]
        if "fov_y" in camera_info_data:
            camera_info = CameraInfo(camera_info_data["width"],
                                     camera_info_data["height"],
                                     camera_info_data["fov_y"])
        else:
            camera_info = CameraInfo(
                camera_info_data["width"], camera_info_data["height"],
                camera_info_data["focal_x"], camera_info_data["focal_y"],
                camera_info_data["center_x"], camera_info_data["center_y"])

        return serial_no, X_WCamera, X_CameraDepth, camera_info

    def _TransformPointCloud(self, point_cloud, colors):
        # transform the point cloud according to the config file
        pc_h = np.ones((4, point_cloud.shape[1]))
        pc_h[:3, :] = np.copy(point_cloud)

        X_WDepth = self.camera_configs["left_camera_pose_world"].dot(
            self.camera_configs["left_camera_pose_internal"])
        point_cloud_transformed = X_WDepth.dot(pc_h)

        # Filter the final point cloud for NaNs and infs
        nan_indices = np.logical_not(np.isnan(point_cloud_transformed))

        point_cloud_transformed = point_cloud_transformed[:, nan_indices[0, :]]
        filtered_colors = colors[:, nan_indices[0, :]]

        inf_indices = np.logical_not(np.isinf(point_cloud_transformed))

        point_cloud_transformed = point_cloud_transformed[:, inf_indices[0, :]]
        filtered_colors = filtered_colors[:, inf_indices[0, :]]

        return point_cloud_transformed[:3, :].T, filtered_colors.T

    def SegmentScene(self, scene_points, scene_colors, model, init_pose):
        """
        Returns a subset of the scene point cloud representing the segmentation
        of the object of interest.

        @param scene_points An Nx3 numpy array representing a scene.
        @param scene_colors An Nx3 numpy array of rgb values corresponding to
            the points in scene_points.
        @param model A Px3 numpy array representing the object model.
        @param init_pose A 4x4 numpy array representing the initial guess of the
            pose of the object.

        @return segmented_points An Mx3 numpy array of segmented object points.
        @return segmented_colors An Mx3 numpy array of corresponding segmented
            object colors.
        """
        if self.segment_scene_function:
            return self.segment_scene_function(
                scene_points, scene_colors, model, init_pose)

        # TODO(kmuhlrad): fill in default segmentation function
        return scene_points, scene_colors

    def GetPose(self, segmented_scene_points, segmented_scene_colors,
                model, init_pose):
        """Returns the pose of the object of interest.

        Args:
        @param segmented_scene_points An Nx3 numpy array of the segmented object
            points.
        @param segmented_scene_colors An Nx3 numpy array of the segmented object
            colors.
        @param model A Px3 numpy array representing the object model.
        @param init_pose A 4x4 numpy array representing the initial guess of the
            pose of the object.

        Returns:
        @return A 4x4 numpy array representing the pose of the object. The
            default is the identity matrix if a get_pose_function is not
            supplied.
        """
        if self.get_pose_function:
            return self.get_pose_function(
                segmented_scene_points, segmented_scene_colors,
                model, init_pose)

        # TODO(kmuhlrad): fill in default alignment function
        return init_pose

    def _DoCalcOutput(self, context, output):
        init_pose = self.EvalAbstractInput(
            context, self.init_pose_port.get_index()).get_value()
        point_cloud = self.EvalAbstractInput(
            context, self.point_cloud_port.get_index()).get_value()

        scene_points, scene_colors = self._TransformPointCloud(
            point_cloud.xyzs(), point_cloud.rgbs())

        segmented_scene_points, segmented_scene_colors = \
            self.SegmentScene(scene_points, scene_colors, self.model, init_pose)

        if self.viz:
            np.save("saved_point_clouds/scene_points", scene_points)
            np.save("saved_point_clouds/scene_colors", scene_colors)
            np.save("saved_point_clouds/segmented_scene_points",
                    segmented_scene_points)
            np.save("saved_point_clouds/segmented_scene_colors",
                    segmented_scene_colors)

        X_WObject_refined = self.GetPose(
            segmented_scene_points, segmented_scene_colors,
            self.model, init_pose)

        output.get_mutable_value().set_matrix(X_WObject_refined.matrix())


def SegmentCrackerBoxByDopePose(scene_points, scene_colors):
    """Removes all points that aren't a part of the cracker box.

    @param scene_points An Nx3 numpy array representing a scene.
    @param scene_colors An Nx3 numpy array of rgb values corresponding to the
        points in scene_points.

    @return brick_points An Mx3 numpy array of points in the box.
    @return brick_colors An Mx3 numpy array of the colors of the brick box.
    """

    # TODO(kmuhlrad): read in cracker box dimensions from .sdf
    # TODO(kmuhlrad): don't hard code initial pose

    init_guess_x = -0.26
    init_guess_y = -0.59
    init_guess_z = 0.42

    cracker_box_height = 0.21

    x_min = init_guess_x - cracker_box_height
    x_max = init_guess_x + cracker_box_height

    y_min = init_guess_y - cracker_box_height
    y_max = init_guess_y + cracker_box_height

    z_min = init_guess_z - cracker_box_height
    z_max = init_guess_z + cracker_box_height

    x_indices = ThresholdArray(scene_points[:, 0], x_min, x_max)
    y_indices = ThresholdArray(scene_points[:, 1], y_min, y_max)
    z_indices = ThresholdArray(scene_points[:, 2], z_min, z_max)

    indices = reduce(np.intersect1d, (x_indices, y_indices, z_indices))

    cracker_box_points = scene_points[indices, :]
    cracker_box_colors = scene_colors[indices, :]

    # TODO(kmuhlrad): don't hardcode rgb

    # with counting black
    # r_avg = 0.364705882353
    # g_avg = 0.188235294118
    # b_avg = 0.152941176471

    # without counting black
    r_avg = 0.647058823529
    g_avg = 0.333333333333
    b_avg = 0.274509803922

    delta = 0.2

    r_min = r_avg - delta
    r_max = r_avg + delta

    g_min = g_avg - delta
    g_max = g_avg + delta

    b_min = b_avg - delta
    b_max = b_avg + delta

    r_indices = ThresholdArray(cracker_box_colors[:, 0], r_min, r_max)
    g_indices = ThresholdArray(cracker_box_colors[:, 1], g_min, g_max)
    b_indices = ThresholdArray(cracker_box_colors[:, 2], b_min, b_max)

    indices = reduce(np.intersect1d, (r_indices, g_indices, b_indices))

    cracker_box_points = cracker_box_points[indices, :]
    cracker_box_colors = cracker_box_colors[indices, :]

    return cracker_box_points, cracker_box_colors

def SegmentCrackerBoxByDopeClusters(scene_points, scene_colors):
    """Removes all points that aren't a part of the cracker box.

    @param scene_points An Nx3 numpy array representing a scene.
    @param scene_colors An Nx3 numpy array of rgb values corresponding to the
        points in scene_points.

    @return brick_points An Mx3 numpy array of points in the box.
    @return brick_colors An Mx3 numpy array of the colors of the brick box.
    """

    # TODO(kmuhlrad): read in cracker box dimensions from .sdf
    # TODO(kmuhlrad): don't hard code initial pose

    init_guess_x = -0.26
    init_guess_y = -0.59
    init_guess_z = 0.42

    cracker_box_height = 0.21

    x_min = init_guess_x - cracker_box_height
    x_max = init_guess_x + cracker_box_height

    y_min = init_guess_y - cracker_box_height
    y_max = init_guess_y + cracker_box_height

    z_min = init_guess_z - cracker_box_height
    z_max = init_guess_z + cracker_box_height

    x_indices = ThresholdArray(scene_points[:, 0], x_min, x_max)
    y_indices = ThresholdArray(scene_points[:, 1], y_min, y_max)
    z_indices = ThresholdArray(scene_points[:, 2], z_min, z_max)

    indices = reduce(np.intersect1d, (x_indices, y_indices, z_indices))

    cracker_box_points = scene_points[indices, :]
    cracker_box_colors = scene_colors[indices, :]

    # TODO(kmuhlrad): don't hardcode rgb

    # with counting black
    # r_avg = 0.364705882353
    # g_avg = 0.188235294118
    # b_avg = 0.152941176471

    # without counting black
    r1 = 149./255
    g1 = 34./255
    b1 = 27./255

    r2 = -100 #178./255
    g2 = -100 #164./255
    b2 = -100 #156./255

    r3 = 190./255
    g3 = 126./255
    b3 = 52./255

    delta = 0.15

    r1_min = r1 - delta
    r1_max = r1 + delta

    g1_min = g1 - delta
    g1_max = g1 + delta

    b1_min = b1 - delta
    b1_max = b1 + delta


    r2_min = r2 - delta
    r2_max = r2 + delta

    g2_min = g2 - delta
    g2_max = g2 + delta

    b2_min = b2 - delta
    b2_max = b2 + delta


    r3_min = r3 - delta
    r3_max = r3 + delta

    g3_min = g3 - delta
    g3_max = g3 + delta

    b3_min = b3 - delta
    b3_max = b3 + delta


    r1_indices = ThresholdArray(cracker_box_colors[:, 0], r1_min, r1_max)
    g1_indices = ThresholdArray(cracker_box_colors[:, 1], g1_min, g1_max)
    b1_indices = ThresholdArray(cracker_box_colors[:, 2], b1_min, b1_max)

    indices1 = reduce(np.intersect1d, (r1_indices, g1_indices, b1_indices))

    r2_indices = ThresholdArray(cracker_box_colors[:, 0], r2_min, r2_max)
    g2_indices = ThresholdArray(cracker_box_colors[:, 1], g2_min, g2_max)
    b2_indices = ThresholdArray(cracker_box_colors[:, 2], b2_min, b2_max)

    indices2 = reduce(np.intersect1d, (r2_indices, g2_indices, b2_indices))

    r3_indices = ThresholdArray(cracker_box_colors[:, 0], r3_min, r3_max)
    g3_indices = ThresholdArray(cracker_box_colors[:, 1], g3_min, g3_max)
    b3_indices = ThresholdArray(cracker_box_colors[:, 2], b3_min, b3_max)

    indices3 = reduce(np.intersect1d, (r3_indices, g3_indices, b3_indices))

    indices = reduce(np.union1d, (indices1, indices2, indices3))

    cracker_box_points = cracker_box_points[indices, :]
    cracker_box_colors = cracker_box_colors[indices, :]

    return cracker_box_points, cracker_box_colors

def SegmentCrackerBoxByDopeHSV(scene_points, scene_colors):
    """Removes all points that aren't a part of the cracker box.

    @param scene_points An Nx3 numpy array representing a scene.
    @param scene_colors An Nx3 numpy array of rgb values corresponding to the
        points in scene_points.

    @return brick_points An Mx3 numpy array of points in the box.
    @return brick_colors An Mx3 numpy array of the colors of the brick box.
    """

    # TODO(kmuhlrad): read in cracker box dimensions from .sdf
    # TODO(kmuhlrad): don't hard code initial pose

    init_guess_x = -0.26
    init_guess_y = -0.59
    init_guess_z = 0.42

    cracker_box_height = 0.21

    x_min = init_guess_x - cracker_box_height
    x_max = init_guess_x + cracker_box_height

    y_min = init_guess_y - cracker_box_height
    y_max = init_guess_y + cracker_box_height

    z_min = init_guess_z - cracker_box_height
    z_max = init_guess_z + cracker_box_height

    x_indices = ThresholdArray(scene_points[:, 0], x_min, x_max)
    y_indices = ThresholdArray(scene_points[:, 1], y_min, y_max)
    z_indices = ThresholdArray(scene_points[:, 2], z_min, z_max)

    indices = reduce(np.intersect1d, (x_indices, y_indices, z_indices))

    cracker_box_points = scene_points[indices, :]
    cracker_box_colors = scene_colors[indices, :]

    # TODO(kmuhlrad): don't hardcode hsv

    import matplotlib.colors as mcolors

    cracker_box_colors_hsv = mcolors.rgb_to_hsv(cracker_box_colors)

    print cracker_box_colors[10000]
    print cracker_box_colors_hsv[10000]

    # with counting black
    # r_avg = 0.364705882353
    # g_avg = 0.188235294118
    # b_avg = 0.152941176471

    # without counting black
    h_avg = 0.384970019226538 / (2 * np.pi)
    s_avg = 0.620146624099
    v_avg = 0.650754493633

    h1 = 3 / 360.
    h2 = 21 / 360.
    h3 = 32 / 360.

    delta = 1 / 360.

    h1_min = h1 - delta
    h1_max = h1 + delta

    h2_min = h2 - delta
    h2_max = h2 + delta

    h3_min = h3 - delta
    h3_max = h3 + delta

    s_min = s_avg - delta
    s_max = s_avg + delta

    v_min = v_avg - delta
    v_max = v_avg + delta

    h1_indices = ThresholdArray(cracker_box_colors_hsv[:, 0], h1_min, h1_max)
    h2_indices = ThresholdArray(cracker_box_colors_hsv[:, 0], h2_min, h2_max)
    h3_indices = ThresholdArray(cracker_box_colors_hsv[:, 0], h3_min, h3_max)
    s_indices = ThresholdArray(cracker_box_colors_hsv[:, 1], s_min, s_max)
    v_indices = ThresholdArray(cracker_box_colors_hsv[:, 2], v_min, v_max)

    #indices = reduce(np.intersect1d, (h_indices, s_indices, v_indices))
    indices = reduce(np.union1d, (h1_indices, h2_indices, h3_indices))

    cracker_box_points = cracker_box_points[indices, :]
    cracker_box_colors = cracker_box_colors[indices, :]

    return cracker_box_points, cracker_box_colors

def GetCrackerBoxPose(cracker_box_points, cracker_box_colors, dope_pose):
    """Finds a good 4x4 pose of the cracker box from the segmented points.

    @param cracker_box_points An Nx3 numpy array of brick points.
    @param cracker_box_colors An Nx3 numpy array of corresponding brick colors.

    @return X_MS A 4x4 numpy array of the best-fit brick pose.
    """
    #model = np.load("/home/amazon/cloud-manipulation-station-sim/perception/models/cracker_box_texture.npy")
    model = np.load("models/cracker_box_texture.npy")

    print dope_pose
    X_MS, error, num_iters = RunICP(
        model, cracker_box_points, init_guess=dope_pose,
        max_iterations=100, tolerance=1e-8)

    print "ICP Error:", error
    print "Num ICP Iters:", num_iters
    print

    return X_MS


def SegmentSugarBoxByDopePose(scene_points, scene_colors):
    """Removes all points that aren't a part of the cracker box.

    @param scene_points An Nx3 numpy array representing a scene.
    @param scene_colors An Nx3 numpy array of rgb values corresponding to the
        points in scene_points.

    @return brick_points An Mx3 numpy array of points in the box.
    @return brick_colors An Mx3 numpy array of the colors of the brick box.
    """

    # TODO(kmuhlrad): read in cracker box dimensions from .sdf
    # TODO(kmuhlrad): don't hard code initial pose

    init_guess_x = -0.26
    init_guess_y = -0.75
    init_guess_z = 0.38

    sugar_box_height = 0.18

    x_min = init_guess_x - sugar_box_height
    x_max = init_guess_x + sugar_box_height

    y_min = init_guess_y - sugar_box_height
    y_max = init_guess_y + sugar_box_height

    z_min = init_guess_z - sugar_box_height
    z_max = init_guess_z + sugar_box_height

    x_indices = ThresholdArray(scene_points[:, 0], x_min, x_max)
    y_indices = ThresholdArray(scene_points[:, 1], y_min, y_max)
    z_indices = ThresholdArray(scene_points[:, 2], z_min, z_max)

    indices = reduce(np.intersect1d, (x_indices, y_indices, z_indices))

    cracker_box_points = scene_points[indices, :]
    cracker_box_colors = scene_colors[indices, :]

    # TODO(kmuhlrad): don't hardcode rgb

    # without counting black
    r_avg = 0.709803921569
    g_avg = 0.694117647059
    b_avg = 0.458823529412

    delta = 0.2

    r_min = r_avg - delta
    r_max = r_avg + delta

    g_min = g_avg - delta
    g_max = g_avg + delta

    b_min = b_avg - delta
    b_max = b_avg + delta

    r_indices = ThresholdArray(cracker_box_colors[:, 0], r_min, r_max)
    g_indices = ThresholdArray(cracker_box_colors[:, 1], g_min, g_max)
    b_indices = ThresholdArray(cracker_box_colors[:, 2], b_min, b_max)

    indices = reduce(np.intersect1d, (r_indices, g_indices, b_indices))

    cracker_box_points = cracker_box_points[indices, :]
    cracker_box_colors = cracker_box_colors[indices, :]

    return cracker_box_points, cracker_box_colors

def SegmentSugarBoxByDopeHSV(scene_points, scene_colors):
    """Removes all points that aren't a part of the cracker box.

    @param scene_points An Nx3 numpy array representing a scene.
    @param scene_colors An Nx3 numpy array of rgb values corresponding to the
        points in scene_points.

    @return brick_points An Mx3 numpy array of points in the box.
    @return brick_colors An Mx3 numpy array of the colors of the brick box.
    """

    # TODO(kmuhlrad): read in cracker box dimensions from .sdf
    # TODO(kmuhlrad): don't hard code initial pose

    init_guess_x = -0.26
    init_guess_y = -0.75
    init_guess_z = 0.38

    sugar_box_height = 0.18

    x_min = init_guess_x - sugar_box_height
    x_max = init_guess_x + sugar_box_height

    y_min = init_guess_y - sugar_box_height
    y_max = init_guess_y + sugar_box_height

    z_min = init_guess_z - sugar_box_height
    z_max = init_guess_z + sugar_box_height

    x_indices = ThresholdArray(scene_points[:, 0], x_min, x_max)
    y_indices = ThresholdArray(scene_points[:, 1], y_min, y_max)
    z_indices = ThresholdArray(scene_points[:, 2], z_min, z_max)

    indices = reduce(np.intersect1d, (x_indices, y_indices, z_indices))

    cracker_box_points = scene_points[indices, :]
    cracker_box_colors = scene_colors[indices, :]

    # TODO(kmuhlrad): don't hardcode rgb

    import matplotlib.colors as mcolors

    cracker_box_colors_hsv = mcolors.rgb_to_hsv(cracker_box_colors)

    h1 = 61 / 360.
    h2 = 56 / 360.
    h3 = 245 / 360.

    delta = 1 / 360.

    h1_min = h1 - delta
    h1_max = h1 + delta

    h2_min = h2 - delta
    h2_max = h2 + delta

    h3_min = h3 - delta
    h3_max = h3 + delta

    # s_min = s_avg - delta
    # s_max = s_avg + delta

    # v_min = v_avg - delta
    # v_max = v_avg + delta

    h1_indices = ThresholdArray(cracker_box_colors_hsv[:, 0], h1_min, h1_max)
    h2_indices = ThresholdArray(cracker_box_colors_hsv[:, 0], h2_min, h2_max)
    h3_indices = ThresholdArray(cracker_box_colors_hsv[:, 0], h3_min, h3_max)
    # s_indices = ThresholdArray(cracker_box_colors_hsv[:, 1], s_min, s_max)
    # v_indices = ThresholdArray(cracker_box_colors_hsv[:, 2], v_min, v_max)

    #indices = reduce(np.intersect1d, (h_indices, s_indices, v_indices))
    indices = reduce(np.union1d, (h1_indices, h2_indices, h3_indices))

    cracker_box_points = cracker_box_points[indices, :]
    cracker_box_colors = cracker_box_colors[indices, :]

    return cracker_box_points, cracker_box_colors

def GetSugarBoxPose(cracker_box_points, cracker_box_colors, dope_pose):
    """Finds a good 4x4 pose of the cracker box from the segmented points.

    @param cracker_box_points An Nx3 numpy array of brick points.
    @param cracker_box_colors An Nx3 numpy array of corresponding brick colors.

    @return X_MS A 4x4 numpy array of the best-fit brick pose.
    """
    #model = np.load("/home/amazon/cloud-manipulation-station-sim/perception/models/cracker_box_texture.npy")
    model = np.load("models/sugar_box_texture.npy")

    print dope_pose
    X_MS, error, num_iters = RunICP(
        model, cracker_box_points, init_guess=dope_pose,
        max_iterations=100, tolerance=1e-8)

    print "ICP Error:", error
    print "Num ICP Iters:", num_iters
    print

    return X_MS

def SegmentGelatinBoxByDopePose(scene_points, scene_colors):
    """Removes all points that aren't a part of the cracker box.

    @param scene_points An Nx3 numpy array representing a scene.
    @param scene_colors An Nx3 numpy array of rgb values corresponding to the
        points in scene_points.

    @return brick_points An Mx3 numpy array of points in the box.
    @return brick_colors An Mx3 numpy array of the colors of the brick box.
    """

    # TODO(kmuhlrad): read in cracker box dimensions from .sdf
    # TODO(kmuhlrad): don't hard code initial pose

    init_guess_x = -0.11
    init_guess_y = -0.75
    init_guess_z = 0.53

    gelatin_box_height = 0.18

    x_min = init_guess_x - gelatin_box_height
    x_max = init_guess_x + gelatin_box_height

    y_min = init_guess_y - gelatin_box_height
    y_max = init_guess_y + gelatin_box_height

    z_min = init_guess_z - gelatin_box_height
    z_max = init_guess_z + gelatin_box_height

    x_min = -0.2
    x_max = -0.1

    y_min = -0.65
    y_max = -0.6

    z_min = 0.35
    z_max = 0.6

    x_indices = ThresholdArray(scene_points[:, 0], x_min, x_max)
    y_indices = ThresholdArray(scene_points[:, 1], y_min, y_max)
    z_indices = ThresholdArray(scene_points[:, 2], z_min, z_max)

    indices = reduce(np.intersect1d, (x_indices, y_indices, z_indices))

    cracker_box_points = scene_points[indices, :]
    cracker_box_colors = scene_colors[indices, :]

    # TODO(kmuhlrad): don't hardcode rgb

    # without counting black
    r_avg = 0.690196078431
    g_avg = 0.458823529412
    b_avg = 0.439215686275

    delta = 0.2

    r_min = r_avg - delta
    r_max = r_avg + delta

    g_min = g_avg - delta
    g_max = g_avg + delta

    b_min = b_avg - delta
    b_max = b_avg + delta

    r_min = 0
    r_max = 1

    g_min = 0
    g_max = 0.2

    b_min = 0
    b_max = 0.2

    r_indices = ThresholdArray(cracker_box_colors[:, 0], r_min, r_max)
    g_indices = ThresholdArray(cracker_box_colors[:, 1], g_min, g_max)
    b_indices = ThresholdArray(cracker_box_colors[:, 2], b_min, b_max)

    indices = reduce(np.intersect1d, (r_indices, g_indices, b_indices))

    cracker_box_points = cracker_box_points[indices, :]
    cracker_box_colors = cracker_box_colors[indices, :]

    return cracker_box_points, cracker_box_colors

def SegmentGelatinBoxByDopeClusters(scene_points, scene_colors):
    """Removes all points that aren't a part of the cracker box.

    @param scene_points An Nx3 numpy array representing a scene.
    @param scene_colors An Nx3 numpy array of rgb values corresponding to the
        points in scene_points.

    @return brick_points An Mx3 numpy array of points in the box.
    @return brick_colors An Mx3 numpy array of the colors of the brick box.
    """

    # TODO(kmuhlrad): read in cracker box dimensions from .sdf
    # TODO(kmuhlrad): don't hard code initial pose

    init_guess_x = -0.11
    init_guess_y = -0.75
    init_guess_z = 0.53

    gelatin_box_height = 0.18

    x_min = init_guess_x - gelatin_box_height
    x_max = init_guess_x + gelatin_box_height

    y_min = init_guess_y - gelatin_box_height
    y_max = init_guess_y + gelatin_box_height

    z_min = init_guess_z - gelatin_box_height
    z_max = init_guess_z + gelatin_box_height

    x_indices = ThresholdArray(scene_points[:, 0], x_min, x_max)
    y_indices = ThresholdArray(scene_points[:, 1], y_min, y_max)
    z_indices = ThresholdArray(scene_points[:, 2], z_min, z_max)

    indices = reduce(np.intersect1d, (x_indices, y_indices, z_indices))

    cracker_box_points = scene_points[indices, :]
    cracker_box_colors = scene_colors[indices, :]

    # TODO(kmuhlrad): don't hardcode rgb

    # with counting black
    # r_avg = 0.364705882353
    # g_avg = 0.188235294118
    # b_avg = 0.152941176471

    # without counting black
    r1 = 179./255
    g1 = 178./255
    b1 = 174./255

    r2 = 166./255
    g2 = 48./255
    b2 = 42./255

    r3 = 175./255
    g3 = 108./255
    b3 = 103./255

    delta = 0.15

    r1_min = r1 - delta
    r1_max = r1 + delta

    g1_min = g1 - delta
    g1_max = g1 + delta

    b1_min = b1 - delta
    b1_max = b1 + delta


    r2_min = r2 - delta
    r2_max = r2 + delta

    g2_min = g2 - delta
    g2_max = g2 + delta

    b2_min = b2 - delta
    b2_max = b2 + delta


    r3_min = r3 - delta
    r3_max = r3 + delta

    g3_min = g3 - delta
    g3_max = g3 + delta

    b3_min = b3 - delta
    b3_max = b3 + delta


    r1_indices = ThresholdArray(cracker_box_colors[:, 0], r1_min, r1_max)
    g1_indices = ThresholdArray(cracker_box_colors[:, 1], g1_min, g1_max)
    b1_indices = ThresholdArray(cracker_box_colors[:, 2], b1_min, b1_max)

    indices1 = reduce(np.intersect1d, (r1_indices, g1_indices, b1_indices))

    r2_indices = ThresholdArray(cracker_box_colors[:, 0], r2_min, r2_max)
    g2_indices = ThresholdArray(cracker_box_colors[:, 1], g2_min, g2_max)
    b2_indices = ThresholdArray(cracker_box_colors[:, 2], b2_min, b2_max)

    indices2 = reduce(np.intersect1d, (r2_indices, g2_indices, b2_indices))

    r3_indices = ThresholdArray(cracker_box_colors[:, 0], r3_min, r3_max)
    g3_indices = ThresholdArray(cracker_box_colors[:, 1], g3_min, g3_max)
    b3_indices = ThresholdArray(cracker_box_colors[:, 2], b3_min, b3_max)

    indices3 = reduce(np.intersect1d, (r3_indices, g3_indices, b3_indices))

    indices = reduce(np.union1d, (indices1, indices2, indices3))

    cracker_box_points = cracker_box_points[indices, :]
    cracker_box_colors = cracker_box_colors[indices, :]

    return cracker_box_points, cracker_box_colors

def SegmentGelatinBoxByDopeHSV(scene_points, scene_colors):
    """Removes all points that aren't a part of the cracker box.

    @param scene_points An Nx3 numpy array representing a scene.
    @param scene_colors An Nx3 numpy array of rgb values corresponding to the
        points in scene_points.

    @return brick_points An Mx3 numpy array of points in the box.
    @return brick_colors An Mx3 numpy array of the colors of the brick box.
    """

    # TODO(kmuhlrad): read in cracker box dimensions from .sdf
    # TODO(kmuhlrad): don't hard code initial pose

    init_guess_x = -0.26
    init_guess_y = -0.59
    init_guess_z = 0.42

    cracker_box_height = 0.21

    x_min = init_guess_x - cracker_box_height
    x_max = init_guess_x + cracker_box_height

    y_min = init_guess_y - cracker_box_height
    y_max = init_guess_y + cracker_box_height

    z_min = init_guess_z - cracker_box_height
    z_max = init_guess_z + cracker_box_height

    x_indices = ThresholdArray(scene_points[:, 0], x_min, x_max)
    y_indices = ThresholdArray(scene_points[:, 1], y_min, y_max)
    z_indices = ThresholdArray(scene_points[:, 2], z_min, z_max)

    indices = reduce(np.intersect1d, (x_indices, y_indices, z_indices))

    cracker_box_points = scene_points[indices, :]
    cracker_box_colors = scene_colors[indices, :]

    # TODO(kmuhlrad): don't hardcode hsv

    import matplotlib.colors as mcolors

    cracker_box_colors_hsv = mcolors.rgb_to_hsv(cracker_box_colors)

    print cracker_box_colors[10000]
    print cracker_box_colors_hsv[10000]

    # with counting black
    # r_avg = 0.364705882353
    # g_avg = 0.188235294118
    # b_avg = 0.152941176471

    # without counting black
    h_avg = 0.384970019226538 / (2 * np.pi)
    s_avg = 0.620146624099
    v_avg = 0.650754493633

    h1 = 50 / 360.
    h2 = 3 / 360.
    h3 = 4 / 360.

    delta = 1 / 360.

    h1_min = h1 - delta
    h1_max = h1 + delta

    h2_min = h2 - delta
    h2_max = h2 + delta

    h3_min = h3 - delta
    h3_max = h3 + delta

    s_min = s_avg - delta
    s_max = s_avg + delta

    v_min = v_avg - delta
    v_max = v_avg + delta

    h1_indices = ThresholdArray(cracker_box_colors_hsv[:, 0], h1_min, h1_max)
    h2_indices = ThresholdArray(cracker_box_colors_hsv[:, 0], h2_min, h2_max)
    h3_indices = ThresholdArray(cracker_box_colors_hsv[:, 0], h3_min, h3_max)
    s_indices = ThresholdArray(cracker_box_colors_hsv[:, 1], s_min, s_max)
    v_indices = ThresholdArray(cracker_box_colors_hsv[:, 2], v_min, v_max)

    #indices = reduce(np.intersect1d, (h_indices, s_indices, v_indices))
    indices = reduce(np.union1d, (h1_indices, h2_indices, h3_indices))

    cracker_box_points = cracker_box_points[indices, :]
    cracker_box_colors = cracker_box_colors[indices, :]

    return cracker_box_points, cracker_box_colors

def GetGelatinBoxPose(cracker_box_points, cracker_box_colors, dope_pose):
    """Finds a good 4x4 pose of the cracker box from the segmented points.

    @param cracker_box_points An Nx3 numpy array of brick points.
    @param cracker_box_colors An Nx3 numpy array of corresponding brick colors.

    @return X_MS A 4x4 numpy array of the best-fit brick pose.
    """
    #model = np.load("/home/amazon/cloud-manipulation-station-sim/perception/models/cracker_box_texture.npy")
    model = np.load("models/gelatin_box_texture.npy")

    print dope_pose
    X_MS, error, num_iters = RunICP(
        model, cracker_box_points, init_guess=dope_pose,
        max_iterations=100, tolerance=1e-8)

    print "ICP Error:", error
    print "Num ICP Iters:", num_iters
    print

    return X_MS

def read_poses_from_file(filename):
    pose_dict = {}
    row_num = 0
    cur_matrix = np.eye(4)
    with open(filename, "r") as f:
        for line in f:
            line = line.rstrip()
            if not line.lstrip(" ").startswith("["):
                object_name = line
            else:
                row = np.matrix(line)
                cur_matrix[row_num, :] = row
                row_num += 1
                if row_num == 4:
                    pose_dict[object_name] = Isometry3(cur_matrix)
                    cur_matrix = np.eye(4)
                row_num %= 4
    return pose_dict


def read_rgba_values(filename):
    from PIL import Image

    im = Image.open(filename) # Can be many different formats.
    pix = im.load()
    print im.size  # Get the width and hight of the image for iterating over
    print pix[0, 0]  # Get the RGBA Value of the a pixel of an image
    print pix[100, 100]  # Get the RGBA Value of the a pixel of an image

    average_r = 0
    average_g = 0
    average_b = 0

    colored_pixels = 0

    for x in range(im.size[0]):
        for y in range(im.size[1]):
            r, g, b = pix[x, y]
            if r and g and b:
                average_r += r
                average_g += g
                average_b += b
                colored_pixels += 1

    average_r /= colored_pixels
    average_g /= colored_pixels
    average_b /= colored_pixels

    print average_r / 255.
    print average_g / 255.
    print average_b / 255.

def read_hsv_values(filename):
    from PIL import Image
    import colorsys

    im = Image.open(filename) # Can be many different formats.
    pix = im.load()
    print im.size  # Get the width and hight of the image for iterating over
    print pix[0, 0]  # Get the RGBA Value of the a pixel of an image
    print pix[100, 100]  # Get the RGBA Value of the a pixel of an image

    average_h = 0

    sin_h = 0
    cos_h = 0

    average_s = 0
    average_v = 0

    colored_pixels = 0

    for x in range(im.size[0]):
        for y in range(im.size[1]):
            r, g, b = pix[x, y]
            h, s, v = colorsys.rgb_to_hsv(r/255., g/255., b/255.)
            if r and g and b:
                # average_h += h
                sin_h += np.sin(h*2*np.pi)
                cos_h += np.cos(h*2*np.pi)
                average_s += s
                average_v += v
                colored_pixels += 1

    # average_h /= colored_pixels
    sin_h /= colored_pixels
    cos_h /= colored_pixels

    average_h = np.arctan2(sin_h, cos_h)
    average_s /= colored_pixels
    average_v /= colored_pixels

    print colored_pixels
    print average_h
    print average_s
    print average_v

def main(config_file, model_file, dope_pose_file, object_name, viz=True):
    """Estimates the pose of the foam brick in a ManipulationStation setup.

    @param config_file str. The path to a camera configuration file.
    @param viz bool. If True, save point clouds to numpy arrays.

    @return A 4x4 Numpy array representing the pose of the brick.
    """

    segmentation_functions = {
        'cracker': SegmentCrackerBoxByDopeHSV,
        'gelatin': SegmentGelatinBoxByDopeHSV,
        'sugar': SegmentSugarBoxByDopeHSV,
    }

    alignment_functions = {
        'cracker': GetCrackerBoxPose,
        'gelatin': GetGelatinBoxPose,
        'sugar': GetSugarBoxPose,
    }

    builder = DiagramBuilder()

    station = builder.AddSystem(ManipulationStation())
    station.SetupDopeClutterClearingStation()
    station.Finalize()

    # pose_refinement = builder.AddSystem(PoseRefinement(
    #     config_file, model_file, viz, segmentation_functions[object_name],
    #     alignment_functions[object_name]))

    pose_refinement = builder.AddSystem(PoseRefinement(
            config_file, model_file, viz))

    left_camera_info = pose_refinement.camera_configs["left_camera_info"]
    left_name_prefix = \
        "camera_" + pose_refinement.camera_configs["left_camera_serial"]

    dut = builder.AddSystem(
        mut.DepthImageToPointCloud(left_camera_info, PixelType.kDepth16U))

    builder.Connect(station.GetOutputPort(left_name_prefix + "_depth_image"),
                    dut.depth_image_input_port())
    builder.Connect(station.GetOutputPort(left_name_prefix + "_rgb_image"),
                    dut.rgb_image_input_port())

    builder.Connect(dut.point_cloud_output_port(),
                    pose_refinement.GetInputPort("point_cloud"))

    diagram = builder.Build()
    simulator = Simulator(diagram)

    dope_poses = read_poses_from_file(dope_pose_file)
    dope_pose = dope_poses[object_name]

    context = diagram.GetMutableSubsystemContext(
        pose_refinement, simulator.get_mutable_context())

    context.FixInputPort(pose_refinement.GetInputPort(
        "X_WObject_guess").get_index(), AbstractValue.Make(dope_pose))

    return pose_refinement.GetOutputPort("X_WObject_refined").Eval(context)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config_file",
        required=True,
        help="The path to a .yml camera config file")
    parser.add_argument(
        "--model_file",
        required=True,
        help="The path to a .npy model file")
    parser.add_argument(
        "--dope_pose_file",
        required=True,
        help="The path to a .txt file containing poses returned by DOPE")
    parser.add_argument(
        "--object_name",
        required=True,
        help="One of 'cracker', 'sugar', 'gelatin', 'meat', 'soup', 'mustard'")
    parser.add_argument(
        "--viz",
        action="store_true",
        help="Save the aligned and segmented point clouds for visualization")
    args = parser.parse_args()

    print main(
        args.config_file, args.model_file, args.dope_pose_file,
        args.object_name, args.viz)
