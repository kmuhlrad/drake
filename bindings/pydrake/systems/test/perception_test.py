# -*- coding: utf-8 -*-

import unittest
import numpy as np

from pydrake.common import FindResourceOrThrow
from pydrake.math import RigidTransform, RollPitchYaw, RotationMatrix
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import AbstractValue, DiagramBuilder
from pydrake.systems.perception import (
    ObjectInfo, PointCloudConcatenation, PoseRefinement,
    _ConcatenatePointClouds, _TileColors, _TransformPoints)
from pydrake.systems.rendering import PoseBundle

import pydrake.perception as mut


class TestConcatenatePointClouds(unittest.TestCase):
    def setUp(self):
        self.points_0 = np.array([[1.0], [2.0], [3.0]])
        self.colors_0 = np.array([[0], [128], [255]])

        self.points_1 = np.array([[4.0], [5.0], [6.0]])
        self.colors_1 = np.array([[50], [100], [200]])

        self.points_dict = {"0": self.points_0, "1": self.points_1}
        self.colors_dict = {"0": self.colors_0, "1": self.colors_1}

    def test_concatenation(self):
        scene_points, scene_colors = _ConcatenatePointClouds(
            self.points_dict, self.colors_dict)

        self.assertEqual(scene_points.shape, (3, len(self.points_dict)))
        self.assertEqual(scene_colors.shape, (3, len(self.colors_dict)))
        self.assertEqual(scene_points.shape, scene_colors.shape)

        for i, value in enumerate(self.points_0.flatten()):
            self.assertTrue(value in scene_points[i, :])

        for i, value in enumerate(self.points_1.flatten()):
            self.assertTrue(value in scene_points[i, :])

        for i, value in enumerate(self.colors_0.flatten()):
            self.assertTrue(value in scene_colors[i, :])

        for i, value in enumerate(self.colors_0.flatten()):
            self.assertTrue(value in scene_colors[i, :])


class TestTileColors(unittest.TestCase):
    def setUp(self):
        self.red = [255, 0, 0]
        self.blue = [0, 0, 255]

    def testOneDim(self):
        tiled = _TileColors(self.red, 1)
        expected_tiled = np.array([[255], [0], [0]])
        self.assertTrue(np.allclose(tiled, expected_tiled))

    def testThreeDims(self):
        tiled = _TileColors(self.blue, 1)
        expected_tiled = np.array([[0, 0, 0], [0, 0, 0], [255, 255, 255]])
        self.assertTrue(np.allclose(tiled, expected_tiled))


class TestTransformPoints(unittest.TestCase):
    def setUp(self):
        self.points = np.array([[1, 1, 0], [2, 1, 0]]).T
        self.translation = RigidTransform(p=[1, 2, 3])
        self.rotation = RigidTransform(
            RotationMatrix(RollPitchYaw(0, 0, np.pi/2)))

    def test_translation(self):
        transformed_points = _TransformPoints(
            self.points, self.translation.matrix())
        expected_translated_points = np.array([[2, 3, 3], [3, 3, 3]]).T

        self.assertTrue(
            np.allclose(transformed_points, expected_translated_points))

    def test_rotation(self):
        transformed_points = _TransformPoints(
            self.points, self.rotation.matrix())
        expected_rotated_points = np.array([[-1, 1, 0], [-1, 2, 0]]).T

        self.assertTrue(
            np.allclose(transformed_points, expected_rotated_points))


class TestPointCloudConcatenation(unittest.TestCase):
    def setUp(self):
        builder = DiagramBuilder()

        X_WP_0 = RigidTransform.Identity()
        X_WP_1 = RigidTransform.Identity()
        X_WP_1.set_translation([1.0, 0, 0])

        id_list = ["0", "1"]

        self.pc_concat = builder.AddSystem(PointCloudConcatenation(id_list))

        self.num_points = 10000
        xyzs = np.random.uniform(-0.1, 0.1, (3, self.num_points))
        # Only go to 254 to distinguish between point clouds with and without
        # color.
        rgbs = np.random.uniform(0., 254.0, (3, self.num_points))

        self.pc = mut.PointCloud(
            self.num_points,
            mut.Fields(mut.BaseField.kXYZs | mut.BaseField.kRGBs))
        self.pc.mutable_xyzs()[:] = xyzs
        self.pc.mutable_rgbs()[:] = rgbs

        self.pc_no_rgbs = mut.PointCloud(
            self.num_points, mut.Fields(mut.BaseField.kXYZs))
        self.pc_no_rgbs.mutable_xyzs()[:] = xyzs

        diagram = builder.Build()

        simulator = Simulator(diagram)

        self.context = diagram.GetMutableSubsystemContext(
            self.pc_concat, simulator.get_mutable_context())

        self.context.FixInputPort(
            self.pc_concat.GetInputPort("X_FCi_0").get_index(),
            AbstractValue.Make(X_WP_0))
        self.context.FixInputPort(
            self.pc_concat.GetInputPort("X_FCi_1").get_index(),
            AbstractValue.Make(X_WP_1))

    def test_no_rgb(self):
        self.context.FixInputPort(
            self.pc_concat.GetInputPort("point_cloud_CiSi_0").get_index(),
            AbstractValue.Make(self.pc_no_rgbs))
        self.context.FixInputPort(
            self.pc_concat.GetInputPort("point_cloud_CiSi_1").get_index(),
            AbstractValue.Make(self.pc_no_rgbs))

        fused_pc = self.pc_concat.GetOutputPort("point_cloud_FS").Eval(
            self.context)

        self.assertEqual(fused_pc.size(), 2 * self.num_points)

        # The first point cloud should be from [-0.1 to 0.1].
        # Tthe second point cloud should be from [0.9 to 1.1].
        self.assertTrue(np.max(fused_pc.xyzs()[0, :]) >= 1.0)
        self.assertTrue(np.min(fused_pc.xyzs()[0, :]) <= 0.0)

        # Even if both input point clouds don't have rgbs, the fused point
        # cloud should contain rgbs of the default color.
        self.assertTrue(fused_pc.has_rgbs())
        self.assertTrue(
            np.all(fused_pc.rgbs()[:, 0] == np.array([255, 255, 255])))
        self.assertTrue(
            np.all(fused_pc.rgbs()[:, -1] == np.array([255, 255, 255])))

    def test_rgb(self):
        self.context.FixInputPort(
            self.pc_concat.GetInputPort("point_cloud_CiSi_0").get_index(),
            AbstractValue.Make(self.pc))
        self.context.FixInputPort(
            self.pc_concat.GetInputPort("point_cloud_CiSi_1").get_index(),
            AbstractValue.Make(self.pc))

        fused_pc = self.pc_concat.GetOutputPort("point_cloud_FS").Eval(
            self.context)

        self.assertEqual(fused_pc.size(), 2 * self.num_points)

        # The first point cloud should be from [-0.1 to 0.1].
        # The second point cloud should be from [0.9 to 1.1].
        self.assertTrue(np.max(fused_pc.xyzs()[0, :]) >= 1.0)
        self.assertTrue(np.min(fused_pc.xyzs()[0, :]) <= 0.0)

        self.assertTrue(fused_pc.has_rgbs())
        self.assertTrue(
            np.all(fused_pc.rgbs()[:, 0] != np.array([255, 255, 255])))
        self.assertTrue(
            np.all(fused_pc.rgbs()[:, -1] != np.array([255, 255, 255])))

    def test_mix_rgb(self):
        self.context.FixInputPort(
            self.pc_concat.GetInputPort("point_cloud_CiSi_0").get_index(),
            AbstractValue.Make(self.pc))
        self.context.FixInputPort(
            self.pc_concat.GetInputPort("point_cloud_CiSi_1").get_index(),
            AbstractValue.Make(self.pc_no_rgbs))

        fused_pc = self.pc_concat.GetOutputPort("point_cloud_FS").Eval(
            self.context)

        self.assertEqual(fused_pc.size(), 2 * self.num_points)

        # The first point cloud should be from [-0.1 to 0.1].
        # The second point cloud should be from [0.9 to 1.1].
        self.assertTrue(np.max(fused_pc.xyzs()[0, :]) >= 1.0)
        self.assertTrue(np.min(fused_pc.xyzs()[0, :]) <= 0.0)

        self.assertTrue(fused_pc.has_rgbs())

        # We don't know what order the two point clouds will be combined.
        rgb_first = np.all(fused_pc.rgbs()[:, 0] != np.array([255, 255, 255]))
        rgb_last = np.all(fused_pc.rgbs()[:, -1] != np.array([255, 255, 255]))
        no_rgb_first = np.all(
            fused_pc.rgbs()[:, 0] == np.array([255, 255, 255]))
        no_rgb_last = np.all(
            fused_pc.rgbs()[:, -1] == np.array([255, 255, 255]))

        self.assertTrue(
            (rgb_first and no_rgb_last) or (no_rgb_first and rgb_last))


class TestPoseRefinement(unittest.TestCase):
    def setUp(self):
        # Create the input pose bundle of the ground truth poses.
        self.X_WMustard_expected = RigidTransform(
            RollPitchYaw([-1.57, 0, 3.3]), [0.44, -0.16, 0.09])
        self.X_WSoup_expected = RigidTransform(
            RollPitchYaw([-1.57, 0, 3.14]), [0.40, -0.07, 0.03])

        initial_pose_bundle = PoseBundle(num_poses=2)
        initial_pose_bundle.set_name(0, "mustard")
        initial_pose_bundle.set_pose(0, self.X_WMustard_expected)
        initial_pose_bundle.set_name(1, "soup")
        initial_pose_bundle.set_pose(1, self.X_WSoup_expected)

        # Construct the scene point cloud from saved arrays.
        self.test_models_path = "drake/bindings/pydrake/systems/test/"
        # TODO(kmuhlrad): figure out file paths
        scene_points = np.load(FindResourceOrThrow(
            self.test_models_path + "scene_points.npy"))
        scene_colors = np.load(FindResourceOrThrow(
            self.test_models_path + "scene_colors.npy"))

        self.scene_point_cloud = mut.PointCloud(
            fields=mut.Fields(mut.BaseField.kXYZs | mut.BaseField.kRGBs))
        self.scene_point_cloud.resize(scene_points.shape[1])
        self.scene_point_cloud.mutable_xyzs()[:] = scene_points
        # self.scene_point_cloud.mutable_rgbs()[:] = scene_colors

        # Construct two PoseRefinement Systems: One with the default
        # segmentation and alignment functions, and one with custom
        # segmentation and alignment functions.
        builder = DiagramBuilder()

        self.default_pose_refinement = builder.AddSystem(
            PoseRefinement(self.construct_default_object_info_dict()))
        self.custom_pose_refinement = builder.AddSystem(
            PoseRefinement(self.construct_custom_object_info_dict()))

        diagram = builder.Build()
        simulator = Simulator(diagram)

        self.default_context = diagram.GetMutableSubsystemContext(
            self.default_pose_refinement, simulator.get_mutable_context())
        self.custom_context = diagram.GetMutableSubsystemContext(
            self.custom_pose_refinement, simulator.get_mutable_context())

        self.default_context.FixInputPort(
            self.default_pose_refinement.GetInputPort(
                "pose_bundle_W").get_index(),
            AbstractValue.Make(initial_pose_bundle)
        )
        self.custom_context.FixInputPort(
            self.custom_pose_refinement.GetInputPort(
                "pose_bundle_W").get_index(),
            AbstractValue.Make(initial_pose_bundle)
        )

        self.default_context.FixInputPort(
            self.default_pose_refinement.GetInputPort(
                "point_cloud_W").get_index(),
            AbstractValue.Make(self.scene_point_cloud)
        )
        self.custom_context.FixInputPort(
            self.custom_pose_refinement.GetInputPort(
                "point_cloud_W").get_index(),
            AbstractValue.Make(self.scene_point_cloud)
        )

    def construct_default_object_info_dict(self):
        mustard_info = ObjectInfo(
            "mustard",
            FindResourceOrThrow(
                self.test_models_path + "006_mustard_bottle_textured.npy"))
        soup_info = ObjectInfo(
            "soup",
            FindResourceOrThrow(
                self.test_models_path + "005_tomato_soup_can_textured.npy"))

        object_info_dict = {
            "mustard": mustard_info,
            "soup": soup_info,
        }

        return object_info_dict

    def construct_custom_object_info_dict(self):
        mustard_info = ObjectInfo(
            "mustard",
            FindResourceOrThrow(
                self.test_models_path + "006_mustard_bottle_textured.npy"),
            segment_scene_function=self.full_scene_segmentation_function,
            alignment_function=self.identity_alignment_function)
        soup_info = ObjectInfo(
            "soup",
            FindResourceOrThrow(
                self.test_models_path + "005_tomato_soup_can_textured.npy"),
            segment_scene_function=self.empty_segmentation_function)

        object_info_dict = {
            "mustard": mustard_info,
            "soup": soup_info,
        }

        return object_info_dict

    def full_scene_segmentation_function(self, scene_points, scene_colors,
                                         model, init_pose):
        return scene_points, scene_colors

    def empty_segmentation_function(self, scene_points, scene_colors,
                                    model, init_pose):
        return np.array([]), np.array([])

    def identity_alignment_function(self, segmented_scene_points,
                                    segmented_scene_colors, model, init_pose):
        return np.eye(4)

    def test_default_system(self):
        # Evaluate outputs.
        mustard_pc = self.default_pose_refinement.GetOutputPort(
            "segmented_point_cloud_W_mustard").Eval(self.default_context)
        soup_pc = self.default_pose_refinement.GetOutputPort(
            "segmented_point_cloud_W_soup").Eval(self.default_context)
        refined_pose_bundle = self.default_pose_refinement.GetOutputPort(
            "refined_pose_bundle_W").Eval(self.default_context)

        # Check that the refined poses are the ground truth.
        for i in range(refined_pose_bundle.num_poses()):
            X_WObject_actual = refined_pose_bundle.get_pose(i)
            if refined_pose_bundle.get_name(i) == "mustard":
                self.assertTrue(
                    np.allclose(X_WObject_actual, self.X_WMustard_expected))
            elif refined_pose_bundle.get_name(i) == "soup":
                self.assertTrue(
                    np.allclose(X_WObject_actual, self.X_WSoup_expected))

        # Check that the segmented point clouds are smaller than the scene.
        self.assertLess(len(mustard_pc.xyzs()), len(self.scene_point_cloud.xyzs()))
        self.assertLess(len(soup_pc.xyzs()), len(self.scene_point_cloud.xyzs()))