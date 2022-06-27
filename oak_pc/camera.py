import os.path
import shutil

import depthai as dai
import numpy as np
import cv2 as cv
import open3d as o3d
import pickle
import glob
from typing import Any


def get_frame(queue: Any) -> Any:
    """
    Converts an acquired camera frame to OpenCV frame
    Args:
        queue: Depthai queue

    Returns:
        OpenCV frame
    """
    frame = queue.get()
    return frame.getCvFrame()


def generate_depth_disparity_mapping() -> Any:
    """
    Preloaded intrinsic params of Oak-D S2 camera
    Returns:
        depth-disparity mapping matrix and intrinsic camera params for left and right stereo cameras
    """
    l_intrinsic = np.array(
        [
            [795.2091064453125, 0.0, 637.9916381835938],
            [0.0, 795.2091064453125, 404.3976135253906],
            [0.0, 0.0, 1.0],
        ]
    )
    r_intrinsic = np.array(
        [
            [795.259765625, 0.0, 661.4163208007812],
            [0.0, 795.259765625, 391.59222412109375],
            [0.0, 0.0, 1.0],
        ]
    )
    baseline_distance = 7.5

    # h, w = depth.shape
    cx = l_intrinsic[0][2]
    cy = l_intrinsic[1][2]
    cxp = r_intrinsic[0][2]
    f = l_intrinsic[0][0]
    t = baseline_distance * 10
    q = np.float32(
        [
            [1, 0, 0, -cx],
            [0, 1, 0, -cy],
            [0, 0, 0, f],
            [0, 0, -1 / t, (cx - cxp) / t],
        ]
    )
    return q, l_intrinsic, r_intrinsic


def make_video_from_images(
    images_folder: str,
    video_path: str = os.path.join("output", "video"),
    target_fps: int = 30,
    video_codec: Any = -1,
    delete_images: bool = False,
) -> bool:
    """
    Generates a mp4 video file from list of given images.
    Args:
        images_folder: Folder where input images reside
        video_path: Path to output video
        target_fps: Target fps for the output video
        video_codec: Used video codec for video encoding
        delete_images: Delete images after creating video

    Returns:
        True if the process is successful.
    """
    img_array = []
    size = (0, 0)
    for filename in glob.glob(os.path.join(images_folder, "*.png")):
        img = cv.imread(filename)  # pylint: disable=no-member
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    # out = cv.VideoWriter('project.avi', cv.VideoWriter_fourcc(*'DIVX'), 15, size)
    os.makedirs(video_path, exist_ok=True)
    out = cv.VideoWriter(  # pylint: disable=no-member
        os.path.join(video_path, "out.mp4"), video_codec, target_fps, size
    )

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

    if delete_images:
        shutil.rmtree(images_folder)
    return True


class DepthCamera(object):
    def __init__(
        self,
        high_accuracy: bool = False,
        median_filtering: bool = False,
        sub_pixel_accuracy: bool = True,
    ):
        """
        Constructor for DepthCamera object
        Args:
            high_accuracy: Use high accuracy depth measurement
            median_filtering: Use median filtering for smoothing the disparity image
            sub_pixel_accuracy: Sub-pixel accuracy for 3D reconstruction
        """
        self.pipeline = dai.Pipeline()
        self.monoLeft = self.get_mono_camera(is_left=True)
        self.monoRight = self.get_mono_camera(is_left=False)
        self.rgb = self.get_rgb_camera()
        self.accuracy = high_accuracy
        self.filtering = median_filtering
        self.subpixel = sub_pixel_accuracy
        self.stereoDepth = self.get_stereo_depth()
        self.xoutDisp, self.xoutRGB, self.xoutDepth = self.setup_links()
        (
            self.depthDisparityMap,
            self.lIntrinsic,
            self.rIntrinsic,
        ) = generate_depth_disparity_mapping()

    def get_mono_camera(self, is_left: bool) -> Any:
        """
        Constructs the mono camera object in a stereo pair
        Args:
            is_left: Is the camera, the left camera in the stereo pair

        Returns:
            Depthai mono camera object
        """
        mono = self.pipeline.createMonoCamera()
        mono.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)

        if is_left:
            mono.setBoardSocket(dai.CameraBoardSocket.LEFT)
        else:
            mono.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        return mono

    def get_rgb_camera(self) -> Any:
        """
        Constructs an RGB camera object
        Returns:
            Depthai RGB camera object
        """
        rgb = self.pipeline.createColorCamera()
        rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
        return rgb

    def get_stereo_depth(self) -> Any:
        """
        Gets the stereo depth from the stereo camera object
        Returns:
            Depthai stereo depth object containing depth and disparity
        """
        stereo_depth = self.pipeline.createStereoDepth()
        stereo_depth.setLeftRightCheck(True)
        self.monoLeft.out.link(stereo_depth.left)
        self.monoRight.out.link(stereo_depth.right)
        preset = (
            dai.node.StereoDepth.PresetMode.HIGH_DENSITY
            if self.accuracy
            else dai.node.StereoDepth.PresetMode.HIGH_DENSITY
        )
        stereo_depth.setDefaultProfilePreset(preset)
        median_filter = (
            dai.MedianFilter.KERNEL_7x7
            if self.filtering
            else dai.MedianFilter.MEDIAN_OFF
        )
        stereo_depth.initialConfig.setMedianFilter(median_filter)
        stereo_depth.setSubpixel(self.subpixel)
        stereo_depth.initialConfig.setConfidenceThreshold(200)
        stereo_depth.setRectifyEdgeFillColor(0)
        return stereo_depth

    def setup_links(self) -> Any:
        """
        Setup input and output links to the actual camera hardware
        Returns:
            Oak-D S2 I/O links for depth/disparity/RGB
        """
        xout_disp = self.pipeline.createXLinkOut()
        xout_disp.setStreamName("disparity")

        xout_depth = self.pipeline.createXLinkOut()
        xout_depth.setStreamName("depth")

        xout_rgb = self.pipeline.createXLinkOut()
        xout_rgb.setStreamName("color")

        self.stereoDepth.disparity.link(xout_disp.input)
        self.stereoDepth.depth.link(xout_depth.input)

        self.rgb.video.link(xout_rgb.input)
        return xout_disp, xout_rgb, xout_depth

    def run_camera(
        self,
        rgb_path: str = "rgb",
        disparity_path: str = "disparity",
        depth_path: str = "depth",
        save_video_frames: bool = False,
    ) -> None:
        """
        Run camera and show frames and point cloud.
        Args:
            rgb_path: Path to save RGB images
            disparity_path: Path to save disparity images
            depth_path: Path to serialize depth data
            save_video_frames: save sequence of images as video frames

        Returns:
            None
        """
        p_intrinsic = np.round(self.lIntrinsic).astype(int)

        pinhole_camera_intrinsic = (
            o3d.camera.PinholeCameraIntrinsic(  # pylint: disable=no-member
                1280,
                720,
                p_intrinsic[0][0],
                p_intrinsic[1][1],
                p_intrinsic[0][2],
                p_intrinsic[1][2],
            )
        )

        os.makedirs(os.path.join("output", rgb_path), exist_ok=True)
        os.makedirs(os.path.join("output", disparity_path), exist_ok=True)
        os.makedirs(os.path.join("output", depth_path), exist_ok=True)

        with dai.Device(self.pipeline) as device:
            disparity_queue = device.getOutputQueue(
                name="disparity", maxSize=1, blocking=False
            )
            depth_queue = device.getOutputQueue(name="depth", maxSize=1, blocking=False)
            rgb_queue = device.getOutputQueue(name="color", maxSize=1, blocking=False)
            disparity_multiplier = (
                255 / self.stereoDepth.initialConfig.getMaxDisparity()
            )

            capture_num = 0
            vis = o3d.visualization.Visualizer()  # pylint: disable=no-member
            vis.create_window()
            is_first_pc = True
            while True:
                raw_disparity = get_frame(disparity_queue)
                disparity = (raw_disparity * disparity_multiplier).astype(np.uint8)
                disparity = cv.applyColorMap(  # pylint: disable=no-member
                    disparity, cv.COLORMAP_JET  # pylint: disable=no-member
                )  # pylint: disable=no-member

                depth = get_frame(depth_queue)

                rgb = get_frame(rgb_queue)

                rgb_reshaped = cv.resize(  # pylint: disable=no-member
                    rgb, (disparity.shape[1], disparity.shape[0])
                )  # pylint: disable=no-member

                cv.imshow("Disparity", disparity)  # pylint: disable=no-member
                cv.imshow("Color", rgb_reshaped)  # pylint: disable=no-member

                rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(  # pylint: disable=no-member
                    o3d.geometry.Image(rgb_reshaped),  # pylint: disable=no-member
                    o3d.geometry.Image(depth),  # pylint: disable=no-member
                    convert_rgb_to_intensity=False,
                )
                if is_first_pc:
                    pcl = o3d.geometry.PointCloud.create_from_rgbd_image(  # pylint: disable=no-member
                        rgbd_image, pinhole_camera_intrinsic
                    )
                    vis.add_geometry(pcl)
                    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(  # pylint: disable=no-member
                        size=0.3, origin=[0, 0, 0]
                    )
                    vis.add_geometry(origin)
                    is_first_pc = False
                else:
                    vis.remove_geometry(pcl)
                    pcl = o3d.geometry.PointCloud.create_from_rgbd_image(  # pylint: disable=no-member
                        rgbd_image, pinhole_camera_intrinsic
                    )
                    vis.add_geometry(pcl)
                    vis.poll_events()
                    vis.update_renderer()

                disparity_save_path = os.path.join(
                    "output",
                    disparity_path,
                    f"disparity_{capture_num}.png",
                )
                rgb_save_path = os.path.join(
                    "output",
                    rgb_path,
                    f"rgb_{capture_num}.png",
                )
                depth_save_path = os.path.join(
                    "output", depth_path, f"depth_{capture_num}.pkl"
                )

                if save_video_frames:
                    cv.imwrite(  # pylint: disable=no-member
                        disparity_save_path, disparity
                    )  # pylint: disable=no-member
                    cv.imwrite(  # pylint: disable=no-member
                        rgb_save_path, rgb_reshaped
                    )  # pylint: disable=no-member
                    with open(depth_save_path, "wb") as f:
                        pickle.dump(depth, f)
                        pickle.dump(disparity, f)
                        pickle.dump(raw_disparity, f)
                    capture_num += 1

                key = cv.waitKey(1)  # pylint: disable=no-member
                if key == ord("q"):
                    break
                elif key == ord("c"):
                    if save_video_frames:
                        print("Saving video, not capturing...")
                        continue
                    print(f"Capturing frame {capture_num}")
                    cv.imwrite(  # pylint: disable=no-member
                        disparity_save_path, disparity
                    )  # pylint: disable=no-member
                    cv.imwrite(  # pylint: disable=no-member
                        rgb_save_path, rgb_reshaped
                    )  # pylint: disable=no-member
                    with open(depth_save_path, "wb") as f:
                        pickle.dump(depth, f)
                        pickle.dump(disparity, f)
                        pickle.dump(raw_disparity, f)
                    capture_num += 1

            cv.destroyAllWindows()  # pylint: disable=no-member
            vis.destroy_window()
        return None
