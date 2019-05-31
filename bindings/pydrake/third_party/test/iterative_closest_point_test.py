# -*- coding: utf-8 -*-

import unittest
import numpy as np

from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.third_party.iterative_closest_point import (
    FindNearestNeighbors, CalcLeastSquaresTransform, RunICP)


class TestIterativeClosestPoint(unittest.TestCase):
    def setUp(self):
        self.point_cloud_A = np.array([[0, 0, 0],
                                       [1, 2, 3],
                                       [10, 9, 8],
                                       [-3, -4, -5]])

        point_cloud_Ah = np.ones((4, self.point_cloud_A.shape[0]))
        point_cloud_Ah[:3, :] = np.copy(self.point_cloud_A.T)

        self.expected_X_BA = RigidTransform(
            RollPitchYaw(np.pi/2, 0, np.pi/2), [1, 2, 3]).matrix()

        self.point_cloud_B = self.expected_X_BA.dot(point_cloud_Ah)[:3, :].T

    def test_find_nearest_neighbors(self):
        point_cloud_A_shuffled = np.array([[10, 9, 8],
                                           [-3, -4, -5],
                                           [1, 2, 3],
                                           [0, 0, 0]])

        actual_distances, actual_indices = FindNearestNeighbors(
            self.point_cloud_A, point_cloud_A_shuffled)

        expected_distances = np.zeros(self.point_cloud_A.shape[0])
        expected_indices = np.array([3, 2, 0, 1])

        self.assertTrue(np.allclose(actual_distances, expected_distances))
        self.assertTrue(np.allclose(actual_indices, expected_indices))

    def test_calc_least_squares_transform(self):
        actual_X_BA = CalcLeastSquaresTransform(
            self.point_cloud_A, self.point_cloud_B)

        self.assertTrue(
            np.allclose(
                np.around(actual_X_BA, decimals=8), self.expected_X_BA))

    def test_run_icp_with_initial_guess_as_true_transform(self):
        actual_X_BA, _, _ = RunICP(
            self.point_cloud_A, self.point_cloud_B,
            init_guess=self.expected_X_BA)

        self.assertTrue(
            np.allclose(
                np.around(actual_X_BA, decimals=3), self.expected_X_BA))
