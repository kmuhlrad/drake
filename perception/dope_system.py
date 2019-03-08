import numpy as np
import cv2
import yaml

# import sys
# sys.path.append('inference')
#

# import torch
# TODO(kmuhlrad): make minimal example of error

from PIL import Image, ImageDraw

from pydrake.common.eigen_geometry import Quaternion
from pydrake.examples.manipulation_station import (ManipulationStation,
                                                   CreateDefaultYcbObjectList)
from pydrake.geometry import ConnectDrakeVisualizer
from pydrake.math import RigidTransform
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import (AbstractValue, BasicVector,
                                       DiagramBuilder, LeafSystem)
from pydrake.systems.lcm import LcmPublisherSystem
from pydrake.systems.rendering import PoseBundle
from pydrake.systems.sensors import ImageRgba8U, ImageToLcmImageArrayT, PixelType

from robotlocomotion import image_array_t

from inference.cuboid import Cuboid3d
from inference.cuboid_pnp_solver import CuboidPNPSolver
from inference.detector import ModelData, ObjectDetector

import pdb


class DopeSystem(LeafSystem):
    """
    A system that runs DOPE [INSERT LINK HERE]. Note that this only runs the
    inference step of DOPE, so trained weights must be provided.

    @system{
      @input_port{rgb_input_image},
      @output_port{annotated_rgb_image},
      @output_port{pose_bundle}
    """
    def __init__(self, weights, config_file):
        """
        @param weights a dict mapping object_name to network weight files.
        """
        LeafSystem.__init__(self)

        # TODO(kmuhlrad): get rid of magic numbers

        self.rgb_input_image = self._DeclareAbstractInputPort(
            "rgb_input_image", AbstractValue.Make(ImageRgba8U(848, 480)))

        self._DeclareAbstractOutputPort("annotated_rgb_image",
                                        lambda: AbstractValue.Make(
                                            ImageRgba8U(848, 480)),
                                        self._DoCalcAnnotatedImage)

        self._DeclareAbstractOutputPort("pose_bundle",
                                        lambda: AbstractValue.Make(
                                            PoseBundle(num_poses=6)),
                                        self._DoCalcPoseBundle)

        self.model_names = ['cracker', 'sugar', 'soup', 'mustard', 'gelatin',
                            'meat']
        self.weights = weights
        self.poses = {}
        # TODO(kmuhlrad): probably set size now
        self.image_data = None

        self.g_draw = None

        self.params = None
        with open(config_file, 'r') as stream:
            try:
                print("Loading DOPE parameters from '{}'...".format(config_file))
                self.params = yaml.load(stream)
                print('    Parameters loaded.')
            except yaml.YAMLError as exc:
                print(exc)

    def _ConvertRgbImageToOpenCv(self, rgb_image):
        colors = np.zeros((3, rgb_image.height() * rgb_image.width()))
        cnt = 0
        for i in range(rgb_image.height()):
            for j in range(rgb_image.width()):
                colors[:3, cnt] = rgb_image.at(j, i)[:3]
                cnt += 1
        colors /= 255.
        colors = colors.reshape([rgb_image.height(), rgb_image.width(), 3])

        return colors

    # TODO(kmuhlrad): update documentation
    def _DrawLine(self, point1, point2, lineColor, lineWidth):
        '''Draws line on image'''
        if not point1 is None and point2 is not None:
            self.g_draw.line([point1,point2], fill=lineColor, width=lineWidth)

    def _DrawDot(self, point, pointColor, pointRadius):
        '''Draws dot (filled circle) on image'''
        if point is not None:
            xy = [
                point[0] - pointRadius,
                point[1] - pointRadius,
                point[0] + pointRadius,
                point[1] + pointRadius
            ]
            self.g_draw.ellipse(xy,
                           fill=pointColor,
                           outline=pointColor
                           )

    def _DrawCube(self, points, color=(255, 0, 0)):
        '''
        Draws cube with a thick solid line across
        the front top edge and an X on the top face.
        '''

        lineWidthForDrawing = 2

        # draw front
        self._DrawLine(points[0], points[1], color, lineWidthForDrawing)
        self._DrawLine(points[1], points[2], color, lineWidthForDrawing)
        self._DrawLine(points[3], points[2], color, lineWidthForDrawing)
        self._DrawLine(points[3], points[0], color, lineWidthForDrawing)

        # draw back
        self._DrawLine(points[4], points[5], color, lineWidthForDrawing)
        self._DrawLine(points[6], points[5], color, lineWidthForDrawing)
        self._DrawLine(points[6], points[7], color, lineWidthForDrawing)
        self._DrawLine(points[4], points[7], color, lineWidthForDrawing)

        # draw sides
        self._DrawLine(points[0], points[4], color, lineWidthForDrawing)
        self._DrawLine(points[7], points[3], color, lineWidthForDrawing)
        self._DrawLine(points[5], points[1], color, lineWidthForDrawing)
        self._DrawLine(points[2], points[6], color, lineWidthForDrawing)

        # draw dots
        self._DrawDot(points[0], pointColor=color, pointRadius=4)
        self._DrawDot(points[1], pointColor=color, pointRadius=4)

        # draw x on the top
        self._DrawLine(points[0], points[5], color, lineWidthForDrawing)
        self._DrawLine(points[1], points[4], color, lineWidthForDrawing)


    def _RunDope(self, context):
        # TODO(kmuhlrad): remove setup from here
        models = {}
        pnp_solvers = {}
        draw_colors = {}

        # Initialize parameters
        matrix_camera = np.zeros((3,3))
        matrix_camera[0, 0] = self.params["camera_settings"]['fx']
        matrix_camera[1, 1] = self.params["camera_settings"]['fy']
        matrix_camera[0, 2] = self.params["camera_settings"]['cx']
        matrix_camera[1, 2] = self.params["camera_settings"]['cy']
        matrix_camera[2, 2] = 1
        dist_coeffs = np.zeros((4,1))

        if "dist_coeffs" in self.params["camera_settings"]:
            dist_coeffs = np.array(self.params["camera_settings"]['dist_coeffs'])
        config_detect = lambda: None
        config_detect.mask_edges = 1
        config_detect.mask_faces = 1
        config_detect.vertex = 1
        config_detect.threshold = 0.5
        config_detect.softmax = 1000
        config_detect.thresh_angle = self.params['thresh_angle']
        config_detect.thresh_map = self.params['thresh_map']
        config_detect.sigma = self.params['sigma']
        config_detect.thresh_points = self.params["thresh_points"]

        # For each object to detect, load network model, and create PNP solver.
        for model in self.params['weights']:
            models[model] = \
                ModelData(
                    model,
                    self.weights[model]
                )
            models[model].load_net_model()

            draw_colors[model] = \
                tuple(self.params["draw_colors"][model])
            pnp_solvers[model] = \
                CuboidPNPSolver(
                    model,
                    matrix_camera,
                    Cuboid3d(self.params['dimensions'][model]),
                    dist_coeffs=dist_coeffs
                )

        g_img = self.EvalAbstractInput(
            context, self.rgb_input_image.get_index()).get_value().data[:, :, :3]

        # Copy and draw image
        img_copy = g_img.copy()
        im = Image.fromarray(img_copy)
        self.g_draw = ImageDraw.Draw(im)

        for m in models:
            # Detect object
            results = ObjectDetector.detect_object_in_image(
                models[m].net,
                pnp_solvers[m],
                g_img,
                config_detect
            )

            # Publish pose and overlay cube on image
            for i_r, result in enumerate(results):
                if result["location"] is None:
                    continue
                loc = result["location"]
                ori = result["quaternion"]
                # TODO(kmuhlrad): turn into pose or posebundle for output
                # port.
                CONVERT_SCALE_CM_TO_METERS = 100
                X_WObject = RigidTransform(
                    Quaternion(ori[3], ori[0], ori[1], ori[2]),
                    [loc[0] / CONVERT_SCALE_CM_TO_METERS,
                     loc[1] / CONVERT_SCALE_CM_TO_METERS,
                     loc[2] / CONVERT_SCALE_CM_TO_METERS])

                self.poses[model] = X_WObject.GetAsIsometry3()

                # Draw the cube
                if None not in result['projected_points']:
                    points2d = []
                    for pair in result['projected_points']:
                        points2d.append(tuple(pair))
                    self._DrawCube(points2d, draw_colors[m])


        # unclear if this is bgr or rgb
        self.image_data = np.array(im)[...,::-1]

    def _DoCalcPoseBundle(self, context, output):
        self._RunDope(context)

        print self.poses
        for model in self.poses:
            print model, self.poses[model].matrix()
            # TODO(kmuhlrad): something isn't quite right here
            output.get_mutable_value().set_pose(self.model_names.index(model), self.poses[model])

    def _DoCalcAnnotatedImage(self, context, output):
        self._RunDope(context)

        print self.image_data.shape
        for i in range(self.image_data.shape[0]):
            for j in range(self.image_data.shape[1]):
                output.get_mutable_value().at(j, i)[:3] = self.image_data[i, j, :3]

def main():
    builder = DiagramBuilder()

    weights = {
        'cracker': '/home/amazon/catkin_ws/src/dope/weights/cracker_60.pth',
        'sugar': '/home/amazon/catkin_ws/src/dope/weights/sugar_60.pth',
        'soup': '/home/amazon/catkin_ws/src/dope/weights/soup_60.pth',
        'mustard': '/home/amazon/catkin_ws/src/dope/weights/mustard_60.pth',
        'gelatin': '/home/amazon/catkin_ws/src/dope/weights/gelatin_60.pth',
        'meat': '/home/amazon/catkin_ws/src/dope/weights/meat_20.pth'
    }
    config_file = '/home/amazon/catkin_ws/src/dope/config/config_pose.yaml'


    # Create the DopeSystem.
    dope_system = builder.AddSystem(DopeSystem(weights, config_file))

    # Create the ManipulationStation.
    station = builder.AddSystem(ManipulationStation())
    station.SetupClutterClearingStation()
    ycb_objects = CreateDefaultYcbObjectList()
    for model_file, X_WObject in ycb_objects:
        station.AddManipulandFromFile(model_file, X_WObject)
    station.Finalize()

    # Connect the rgb images to the DopeSystem.
    builder.Connect(station.GetOutputPort("camera_0_rgb_image"),
                    dope_system.GetInputPort("rgb_input_image"))

    diagram = builder.Build()

    simulator = Simulator(diagram)


    station_context = diagram.GetMutableSubsystemContext(
        station, simulator.get_mutable_context())

    context = diagram.GetMutableSubsystemContext(dope_system,
                                                 simulator.get_mutable_context())

    # TODO(kmuhlrad): visualize image as well
    # print dope_system.GetOutputPort("pose_bundle").Eval(context).get_pose(0)

    annotated_image = dope_system.GetOutputPort("annotated_rgb_image").Eval(context).data
    cv2.imshow("dope image", annotated_image)
    cv2.waitKey(0)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace", action='store_true',
        help="Runs this with debugging.")
    args = parser.parse_args()

    # These are temporary, for debugging, so meh for programming style.
    import sys, trace

    # If there are segfaults, it's a good idea to always use stderr as it
    # always prints to the screen, so you should get as much output as
    # possible.
    sys.stdout = sys.stderr

    # Now trace execution:
    if args.trace:
        tracer = trace.Trace(trace=1, count=0, ignoredirs=["/usr", sys.prefix])
        tracer.run('main()')
    else:
        main()