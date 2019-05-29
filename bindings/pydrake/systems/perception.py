import numpy as np

from pydrake.math import RigidTransform
from pydrake.systems.framework import AbstractValue, DiagramBuilder, LeafSystem
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
        A system that takes in N point clouds and N RigidTransforms that
        put each point cloud in a common frame F. The system returns one point
        cloud combining all of the transformed point clouds. Each point cloud
        must have XYZs. RGBs are optional. If absent, those points will be the
        provided default color.

        @param id_list A list containing the IDs of all of the point clouds.
            This is often the serial number of the camera they came from.
        @param default_rgb A list of length 3 containing the RGB values to use
            in the absence of PointCloud.rgbs. Values should be between 0 and
            255. The default is white.

        TODO(kmuhlrad): figure out frames and id_list stuff
        @system{
          @input_port{point_cloud_C0S0}
          @input_port{X_FC0}
          .
          .
          .
          @input_port{point_cloud_CNSN}
          @input_port{X_FCN}
          @output_port{point_cloud_FS}
        }
        """
        LeafSystem.__init__(self)

        self.point_cloud_ports = {}
        self.transform_ports = {}

        self.id_list = id_list

        self._default_rgb = np.array(default_rgb)

        output_fields = mut.Fields(mut.BaseField.kXYZs | mut.BaseField.kRGBs)

        # TODO(kmuhlrad): figure out frames and id_list stuff
        for id in self.id_list:
            self.point_cloud_ports[id] = self.DeclareAbstractInputPort(
                "point_cloud_C{}S{}".format(id, id),
                AbstractValue.Make(mut.PointCloud(fields=output_fields)))

            self.transform_ports[id] = self.DeclareAbstractInputPort(
                "X_FC{}".format(id),
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
