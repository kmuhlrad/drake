import argparse
import numpy as np
import yaml

from iterative_closest_point import RunICP
# from visualization_utils import ThresholdArray

from pydrake.util.eigen_geometry import Isometry3
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import (AbstractValue, BasicVector,
                                       DiagramBuilder, LeafSystem,
                                       PortDataType)
from pydrake.examples.manipulation_station import (
    ManipulationStation, ManipulationStationHardwareInterface)

from pydrake.systems.framework import AbstractValue, LeafSystem
from pydrake.common.eigen_geometry import Isometry3
from pydrake.systems.sensors import CameraInfo
import pydrake.perception as mut
from pydrake.systems.sensors import PixelType

from PIL import Image

import meshcat.transformations as tf

class PoseRefinement(LeafSystem):

    def __init__(self, config_file, model_points_file, model_image_file,
                 segment_scene_function=None, get_pose_function=None,
                 viz=False):
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
        self.model = np.load(model_points_file)
        self.model_image = Image.open(model_image_file)

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
        # TODO(kmuhlrad): change this to drake
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

    def _ThresholdArray(self, arr, min_val, max_val):
        """
        Finds where the values of arr are between min_val and max_val inclusive.

        @param arr An (N, ) numpy array containing number values.
        @param min_val number. The minimum value threshold.
        @param max_val number. The maximum value threshold.

        @return An (M, ) numpy array of the integer indices in arr with values
            that are between min_val and max_val.
        """
        return np.where(
            abs(arr - (max_val + min_val) / 2.) < (max_val - min_val) / 2.)[0]

    def _ReadRgbValues(self, image):
        pix = image.load()

        average_r = 0
        average_g = 0
        average_b = 0

        colored_pixels = 0
        black_threshold = 0

        for x in range(image.size[0]):
            for y in range(image.size[1]):
                r, g, b = pix[x, y]
                if r > black_threshold and g > black_threshold and b > black_threshold:
                    average_r += r
                    average_g += g
                    average_b += b
                    colored_pixels += 1

        average_r /= float(colored_pixels)
        average_g /= float(colored_pixels)
        average_b /= float(colored_pixels)

        return average_r, average_g, average_b

    def SegmentScene(
            self, scene_points, scene_colors, model, model_image, init_pose):
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
                scene_points, scene_colors, model, model_image, init_pose)

        # Filter by area around initial pose guess
        max_delta_x = np.abs(np.max(model[:, 0]) - np.min(model[:, 0]))
        max_delta_y = np.abs(np.max(model[:, 1]) - np.min(model[:, 1]))
        max_delta_z = np.abs(np.max(model[:, 2]) - np.min(model[:, 2]))

        max_delta = np.max([max_delta_x, max_delta_y, max_delta_z])

        init_x = init_pose.matrix()[0, 3]
        init_y = init_pose.matrix()[1, 3]
        init_z = init_pose.matrix()[2, 3]

        x_min = init_x - max_delta
        x_max = init_x + max_delta

        y_min = init_y - max_delta
        y_max = init_y + max_delta

        z_min = init_z - max_delta
        z_max = init_z + max_delta

        x_indices = self._ThresholdArray(scene_points[:, 0], x_min, x_max)
        y_indices = self._ThresholdArray(scene_points[:, 1], y_min, y_max)
        z_indices = self._ThresholdArray(scene_points[:, 2], z_min, z_max)

        indices = reduce(np.intersect1d, (x_indices, y_indices, z_indices))

        segmented_points = scene_points[indices, :]
        segmented_colors = scene_colors[indices, :]

        # Filter by average (r, g, b) value in the model texture image
        r_avg, g_avg, b_avg = self._ReadRgbValues(model_image)
        color_delta = 0.2 * 255

        r_min = r_avg - color_delta
        r_max = r_avg + color_delta

        g_min = g_avg - color_delta
        g_max = g_avg + color_delta

        b_min = b_avg - color_delta
        b_max = b_avg + color_delta

        r_indices = self._ThresholdArray(segmented_colors[:, 0], r_min, r_max)
        g_indices = self._ThresholdArray(segmented_colors[:, 1], g_min, g_max)
        b_indices = self._ThresholdArray(segmented_colors[:, 2], b_min, b_max)

        indices = reduce(np.intersect1d, (r_indices, g_indices, b_indices))

        segmented_points = segmented_points[indices, :]
        segmented_colors = segmented_colors[indices, :]

        return segmented_points, segmented_colors

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

        X_MS, error, num_iters = RunICP(
            model, segmented_scene_points, init_guess=init_pose.matrix(),
            max_iterations=100, tolerance=1e-8)

        return X_MS

    def _DoCalcOutput(self, context, output):
        init_pose = self.EvalAbstractInput(
            context, self.init_pose_port.get_index()).get_value()
        point_cloud = self.EvalAbstractInput(
            context, self.point_cloud_port.get_index()).get_value()

        scene_points, scene_colors = self._TransformPointCloud(
            point_cloud.xyzs(), point_cloud.rgbs())

        segmented_scene_points, segmented_scene_colors = self.SegmentScene(
            scene_points, scene_colors, self.model, self.model_image, init_pose)

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

        output.get_mutable_value().set_matrix(X_WObject_refined)

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

def main(config_file, model_points_file, model_image_file, dope_pose_file, object_name, viz=True):
    """Estimates the pose of the foam brick in a ManipulationStation setup.

    @param config_file str. The path to a camera configuration file.
    @param viz bool. If True, save point clouds to numpy arrays.

    @return A 4x4 Numpy array representing the pose of the brick.
    """

    # segmentation_functions = {
    #     'cracker': SegmentCrackerBoxByDopeHSV,
    #     'gelatin': SegmentGelatinBoxByDopeHSV,
    #     'sugar': SegmentSugarBoxByDopeHSV,
    # }
    #
    # alignment_functions = {
    #     'cracker': GetCrackerBoxPose,
    #     'gelatin': GetGelatinBoxPose,
    #     'sugar': GetSugarBoxPose,
    # }

    builder = DiagramBuilder()

    station = builder.AddSystem(ManipulationStation())
    station.SetupDopeClutterClearingStation()
    station.Finalize()

    # pose_refinement = builder.AddSystem(PoseRefinement(
    #     config_file, model_file, viz, segmentation_functions[object_name],
    #     alignment_functions[object_name]))

    pose_refinement = builder.AddSystem(PoseRefinement(
            config_file, model_points_file, model_image_file, viz=viz))

    left_camera_info = pose_refinement.camera_configs["left_camera_info"]
    left_name_prefix = \
        "camera_" + pose_refinement.camera_configs["left_camera_serial"]

    # use scale factor of 1/1000 to convert mm to m
    dut = builder.AddSystem(
        mut.DepthImageToPointCloud(left_camera_info, PixelType.kDepth16U, 1e-3))

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

    station_context = diagram.GetMutableSubsystemContext(
        station, simulator.get_mutable_context())

    station_context.FixInputPort(station.GetInputPort(
        "iiwa_feedforward_torque").get_index(), np.zeros(7))

    station_context.FixInputPort(station.GetInputPort(
        "iiwa_position").get_index(), np.array([-1.57, 0.1, 0, -1.2, 0, 1.6, 0]))

    station_context.FixInputPort(station.GetInputPort(
        "wsg_position").get_index(), np.array([0.1]))

    station_context.FixInputPort(station.GetInputPort(
        "wsg_force_limit").get_index(), np.array([40.0]))

    simulator.set_publish_every_time_step(False)

    simulator.set_target_realtime_rate(1.0)
    simulator.StepTo(0.1)

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
        "--model_image_file",
        required=True,
        help="The path to a .png model texture file")
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
        args.config_file, args.model_file, args.model_image_file,
        args.dope_pose_file, args.object_name, args.viz)

