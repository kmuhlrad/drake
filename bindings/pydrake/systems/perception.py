import numpy as np

from pydrake.math import RigidTransform
from pydrake.systems.framework import AbstractValue, DiagramBuilder, LeafSystem
from pydrake.systems.rendering import PoseBundle
from pydrake.third_party.iterative_closest_point import RunICP
import pydrake.perception as mut


def _TransformPoints(points_Ci, X_CiSi):
    # Make homogenous copy of points
    points_h_Ci = np.vstack((points_Ci,
                            np.ones((1, points_Ci.shape[1]))))

    return X_CiSi.dot(points_h_Ci)[:3, :]


def _TileColors(color, dim):
    # Need manual broadcasting.
    return np.tile(np.array([color]).T, (1, dim))


def _ConcatenatePointClouds(points_dict, colors_dict):
    scene_points = None
    scene_colors = None

    for id in points_dict:
        if scene_points is None:
            scene_points = points_dict[id]
        else:
            scene_points = np.hstack((points_dict[id], scene_points))

        if scene_colors is None:
            scene_colors = colors_dict[id]
        else:
            scene_colors = np.hstack((colors_dict[id], scene_colors))

    valid_indices = np.logical_not(np.isnan(scene_points))

    scene_points = scene_points[:, valid_indices[0, :]]
    scene_colors = scene_colors[:, valid_indices[0, :]]

    return scene_points, scene_colors


class PointCloudConcatenation(LeafSystem):

    def __init__(self, id_list, default_rgb=[255., 255., 255.]):
        """
        A system that takes in N point clouds of points Si in frame Ci, and N
        RigidTransforms from frame Ci to F, to put each point cloud in a common
        frame F. The system returns one point cloud combining all of the
        transformed point clouds. Each point cloud must have XYZs. RGBs are
        optional. If absent, those points will be the provided default color.

        @param id_list A list containing the string IDs of all of the point
            clouds. This is often the serial number of the camera they came
            from, such as "1" for a simulated camera or "805212060373" for a
            real camera.
        @param default_rgb A list of length 3 containing the RGB values to use
            in the absence of PointCloud.rgbs. Values should be between 0 and
            255. The default is white.

        @system{
          @input_port{point_cloud_CiSi_id0}
          @input_port{X_FCi_id0}
          .
          .
          .
          @input_port{point_cloud_CiSi_idN}
          @input_port{X_FCi_idN}
          @output_port{point_cloud_FS}
        }
        """
        LeafSystem.__init__(self)

        self.point_cloud_ports = {}
        self.transform_ports = {}

        self.id_list = id_list

        self._default_rgb = np.array(default_rgb)

        output_fields = mut.Fields(mut.BaseField.kXYZs | mut.BaseField.kRGBs)

        for id in self.id_list:
            self.point_cloud_ports[id] = self.DeclareAbstractInputPort(
                "point_cloud_CiSi_{}".format(id),
                AbstractValue.Make(mut.PointCloud(fields=output_fields)))

            self.transform_ports[id] = self.DeclareAbstractInputPort(
                "X_FCi_{}".format(id),
                AbstractValue.Make(RigidTransform.Identity()))

        self.DeclareAbstractOutputPort("point_cloud_FS",
                                       lambda: AbstractValue.Make(
                                           mut.PointCloud(
                                               fields=output_fields)),
                                       self.DoCalcOutput)

    def _AlignPointClouds(self, context):
        points = {}
        colors = {}

        for id in self.id_list:
            point_cloud = self.EvalAbstractInput(
                context, self.point_cloud_ports[id].get_index()).get_value()
            X_CiSi = self.EvalAbstractInput(
                context, self.transform_ports[id].get_index()).get_value()

            points[id] = _TransformPoints(point_cloud.xyzs(), X_CiSi.matrix())

            if point_cloud.has_rgbs():
                colors[id] = point_cloud.rgbs()
            else:
                colors[id] = _TileColors(
                    self._default_rgb, point_cloud.xyzs().shape[1])

        return _ConcatenatePointClouds(points, colors)

    def DoCalcOutput(self, context, output):
        scene_points, scene_colors = self._AlignPointClouds(context)

        output.get_mutable_value().resize(scene_points.shape[1])
        output.get_mutable_value().mutable_xyzs()[:] = scene_points
        output.get_mutable_value().mutable_rgbs()[:] = scene_colors


def _ThresholdArray(arr, min_val, max_val):
    """
    Finds where the values of arr are between min_val and max_val (inclusive).

    @param arr An (N, ) numpy array containing number values.
    @param min_val number. The minimum value threshold.
    @param max_val number. The maximum value threshold.

    @return An (M, ) numpy array of the integer indices in arr with values that
        are between min_val and max_val.
    """
    return np.where(
        abs(arr - (max_val + min_val) / 2.) < (max_val - min_val) / 2.)[0]


class ObjectInfo(object):

    def __init__(self, object_name, model_points_file,
                 segment_scene_function=None, alignment_function=None):
        """
        ObjectInfo is a data structure containing the relevant information
        about objects whose pose will be refined. More information about its
        use can be found with the PoseRefinement class documentation.

        @param object_name str. The name of the object.
        @param model_points_file str. The path to a .npy file containing a
            point cloud of the model object in world frame.
        @param segment_scene_function function. A function that segments the
            object out of the entire scene point cloud. For more details about
            the method signature and expected behavior, see
            PoseRefinement.DefaultSegmentSceneFunction.
        @param alignment_function function. A function that takes the
            segmented point cloud and aligns it with the model point cloud. For
            more details about the method signature and expected behavior, see
            PoseRefinement.DefaultAlignPoseFunction.
        """
        self.object_name = object_name
        self.model_points_file = model_points_file
        self.segment_scene_function = segment_scene_function
        self.alignment_function = alignment_function


class PoseRefinement(LeafSystem):

    def __init__(self, object_info_dict):
        """
        A system that takes in a point cloud, initial pose guesses, and a dict
        of ObjectInfo objects to calculate refined poses for each of the
        objects. Through the ObjectInfo dictionary, the user can optionally
        provide custom segmentation and pose alignment functions used to
        determine the final pose. If these functions aren't supplied,
        the default functions in this class will be used. All point clouds and
        poses are assumed to be in world frame. The output is a pose_bundle
        containing the refined poses of every object. If the segmented
        point cloud for a given object is empty, the returned pose will be
        the initial guess. There are also output ports of each of the
        segmented point clouds for visualization purposes.

        @param object_info_dict dict. A dictionary of ObjectInfo objects keyed
            by the ObjectInfo.object_name.

        @system{
          @input_port{point_cloud_W},
          @input_port{pose_bundle_W},
          @output_port{refined_pose_bundle_W}
          @output_port{segmented_point_cloud_W_obj_name_1}
          .
          .
          .
          @output_port{segmented_point_cloud_W_obj_name_n}
        }
        """
        LeafSystem.__init__(self)

        self.object_info_dict = object_info_dict

        self.point_cloud_port = self.DeclareAbstractInputPort(
            "point_cloud_W", AbstractValue.Make(mut.PointCloud()))

        self.pose_bundle_port = self.DeclareAbstractInputPort(
            "pose_bundle_W", AbstractValue.Make(PoseBundle(
                num_poses=len(self.object_info_dict))))

        self.DeclareAbstractOutputPort("refined_pose_bundle_W",
                                       lambda: AbstractValue.Make(
                                           PoseBundle(
                                               num_poses=len(
                                                   self.object_info_dict))),
                                       self.DoCalcOutput)

        output_fields = mut.Fields(mut.BaseField.kXYZs | mut.BaseField.kRGBs)
        for object_name in self.object_info_dict:
            self.DeclareAbstractOutputPort(
                "segmented_point_cloud_W_{}".format(object_name),
                lambda: AbstractValue.Make(
                    mut.PointCloud(
                        fields=output_fields)),
                lambda context, output, object_name=object_name:
                self.DoCalcSegmentedPointCloud(
                    context, output, object_name))

    def DefaultSegmentSceneFunction(
            self, scene_points, scene_colors, model, init_pose):
        """
        Returns a subset of the scene point cloud representing the segmentation
        of the object of interest.

        The default segmentation function is an area filter based on the object
        model and initial pose. Points are only included if they are within the
        size of the largest model dimension of either side of the initial pose.
        For example, if the largest dimension of the model was 2 and init_pose
        was located at (0, 3, 4), all points included in the segmentation mask
        would have x-values between [-2, 2], y-values between [1, 5], and
        z-values between [2, 6].

        If a custom scene segmentation function is supplied, it must have this
        method signature.

        @param scene_points An Nx3 numpy array representing a scene.
        @param scene_colors An Nx3 numpy array of rgb values corresponding to
            the points in scene_points.
        @param model A Px3 numpy array representing the object model.
        @param init_pose A RigidTransform representing the initial guess of the
            pose of the object.

        @return segmented_points An Mx3 numpy array of segmented object points.
        @return segmented_colors An Mx3 numpy array of corresponding segmented
            object colors.
        """
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

        x_indices = _ThresholdArray(scene_points[:, 0], x_min, x_max)
        y_indices = _ThresholdArray(scene_points[:, 1], y_min, y_max)
        z_indices = _ThresholdArray(scene_points[:, 2], z_min, z_max)

        indices = reduce(np.intersect1d, (x_indices, y_indices, z_indices))

        return scene_points[indices, :], scene_colors[indices, :]

    def DefaultAlignPoseFunction(self, segmented_object_points,
                                 segmented_object_colors, model, init_pose):
        """Returns the pose of the object of interest.

        The default pose alignment function runs ICP on the segmented scene
        with a maximum of 100 iterations and stopping threshold of 1e-8. If
        the segmented scene point cloud is empty, this function will return the
        initial pose guess.

        If a custom pose alignment function is supplied, it must have this
        method signature.

        Args:
        @param segmented_object_points An Nx3 numpy array of the segmented
            object points.
        @param segmented_object_colors An Nx3 numpy array of the segmented
            object colors.
        @param model A Px3 numpy array representing the object model.
        @param init_pose An RigidTransform representing the initial guess of
            the pose of the object.

        Returns:
        @return A 4x4 numpy array representing the pose of the object.
        """
        if len(segmented_object_points):
            X_MS, _, _ = RunICP(
                model, segmented_object_points, init_guess=init_pose.matrix(),
                max_iterations=100, tolerance=1e-8)
        else:
            X_MS = init_pose.matrix()

        return X_MS

    def _SegmentObject(self, point_cloud, object_info, init_pose):
        model = np.load(object_info.model_points_file)

        scene_points = np.copy(point_cloud.xyzs()).T
        scene_colors = np.copy(point_cloud.rgbs()).T

        if object_info.segment_scene_function:
            return object_info.segment_scene_function(
                scene_points, scene_colors, model, init_pose)
        else:
            return self.DefaultSegmentSceneFunction(
                scene_points, scene_colors, model, init_pose)

    def _RefineSinglePose(self, point_cloud, object_info, init_pose):
        model = np.load(object_info.model_points_file)

        object_points, object_colors = self._SegmentObject(
            point_cloud, object_info, init_pose)

        if object_info.alignment_function:
            return object_info.alignment_function(
                object_points, object_colors, model, init_pose)
        else:
            return self.DefaultAlignPoseFunction(
                object_points, object_colors, model, init_pose)

    def DoCalcOutput(self, context, output):
        pose_bundle = self.EvalAbstractInput(
            context, self.pose_bundle_port.get_index()).get_value()
        point_cloud = self.EvalAbstractInput(
            context, self.point_cloud_port.get_index()).get_value()

        for i in range(pose_bundle.get_num_poses()):
            if pose_bundle.get_name(i):
                object_name = pose_bundle.get_name(i)
                init_pose = pose_bundle.get_pose(i)
                X_WObject_refined = self._RefineSinglePose(
                    point_cloud, self.object_info_dict[object_name], init_pose)
                output.get_mutable_value().set_name(i, object_name)
                output.get_mutable_value().set_pose(i, X_WObject_refined)

    def DoCalcSegmentedPointCloud(self, context, output, object_name):
        pose_bundle = self.EvalAbstractInput(
            context, self.pose_bundle_port.get_index()).get_value()
        point_cloud = self.EvalAbstractInput(
            context, self.point_cloud_port.get_index()).get_value()

        pose_bundle_index = 0
        for i in range(pose_bundle.get_num_poses()):
            if pose_bundle.get_name(i) == object_name:
                pose_bundle_index = i
                break

        init_pose = pose_bundle.get_pose(pose_bundle_index)
        object_points, object_colors = self._SegmentObject(
            point_cloud, self.object_info_dict[object_name], init_pose)

        output.get_mutable_value().resize(object_points.shape[0])
        output.get_mutable_value().mutable_xyzs()[:] = object_points.T
        output.get_mutable_value().mutable_rgbs()[:] = object_colors.T
