import depthai as dai
import numpy as np
import cv2 as cv
import open3d as o3d
import pickle


def get_frame(queue):
    frame = queue.get()
    return frame.getCvFrame()


def generate_depth_disparity_mapping():
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


class DepthCamera(object):
    def __init__(
        self,
        high_accuracy: bool = False,
        median_filtering: bool = False,
        sub_pixel_accuracy: bool = True,
    ):
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

    def get_mono_camera(self, is_left: bool):
        mono = self.pipeline.createMonoCamera()
        mono.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)

        if is_left:
            mono.setBoardSocket(dai.CameraBoardSocket.LEFT)
        else:
            mono.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        return mono

    def get_rgb_camera(self):
        rgb = self.pipeline.createColorCamera()
        rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
        return rgb

    def get_stereo_depth(self):
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

    def setup_links(self):
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

    def run_camera(self):
        p_intrinsic = np.round(self.lIntrinsic).astype(int)

        pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            1280,
            720,
            p_intrinsic[0][0],
            p_intrinsic[1][1],
            p_intrinsic[0][2],
            p_intrinsic[1][2],
        )

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
            vis = o3d.visualization.Visualizer()
            vis.create_window()
            is_first_pc = True
            while True:
                raw_disparity = get_frame(disparity_queue)
                disparity = (raw_disparity * disparity_multiplier).astype(np.uint8)
                disparity = cv.applyColorMap(disparity, cv.COLORMAP_JET)

                depth = get_frame(depth_queue)

                rgb = get_frame(rgb_queue)

                rgb_reshaped = cv.resize(rgb, (disparity.shape[1], disparity.shape[0]))

                cv.imshow("Disparity", disparity)
                cv.imshow("Color", rgb_reshaped)

                rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    o3d.geometry.Image(rgb_reshaped),
                    o3d.geometry.Image(depth),
                    convert_rgb_to_intensity=False,
                )
                if is_first_pc:
                    pcl = o3d.geometry.PointCloud.create_from_rgbd_image(
                        rgbd_image, pinhole_camera_intrinsic
                    )
                    vis.add_geometry(pcl)
                    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(
                        size=0.3, origin=[0, 0, 0]
                    )
                    vis.add_geometry(origin)
                    is_first_pc = False
                else:
                    vis.remove_geometry(pcl)
                    pcl = o3d.geometry.PointCloud.create_from_rgbd_image(
                        rgbd_image, pinhole_camera_intrinsic
                    )
                    vis.add_geometry(pcl)
                    vis.poll_events()
                    vis.update_renderer()

                key = cv.waitKey(1)
                if key == ord("q"):
                    break
                elif key == ord("c"):
                    cv.imwrite(f"disparity_{capture_num}.png", disparity)
                    cv.imwrite(f"rgb_{capture_num}.png", rgb_reshaped)
                    with open(f"depth_{capture_num}.pkl", "wb") as f:
                        pickle.dump(depth, f)
                        pickle.dump(disparity, f)
                        pickle.dump(raw_disparity, f)
                    capture_num += 1

            cv.destroyAllWindows()
            vis.destroy_window()
