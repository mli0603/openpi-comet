import numpy as np

from scipy.spatial.transform import Rotation as R


def create_pose_matrix(pos, ori):
    """ """
    pose = np.eye(4)
    if len(ori) == 3:
        pose[:3, :3] = R.from_euler("XYZ", ori).as_matrix()
    if len(ori) == 4:
        pose[:3, :3] = R.from_quat(ori).as_matrix()
    pose[:3, 3] = pos
    return pose


class PosePerturbator:
    """ """

    def __init__(self, logger, x_range=(-0.15, +0.15), y_range=(-0.15, +0.15), theta_range=(-np.pi / 12, +np.pi / 12)):
        """ """
        self._logger = logger
        self._x_range = x_range
        self._y_range = y_range
        self._theta_range = theta_range

    def perturb_robot_root_pose(self, robot_pos, robot_quat):
        """ """
        self._logger.info("Perturbing robot root pose ...")

        root_to_base_pose = create_pose_matrix(robot_pos, robot_quat)

        s = np.random.uniform(size=3)
        x = s[0] * (self._x_range[1] - self._x_range[0]) + self._x_range[0]
        y = s[1] * (self._y_range[1] - self._y_range[0]) + self._y_range[0]
        theta = s[2] * (self._theta_range[1] - self._theta_range[0]) + self._theta_range[0]
        self._logger.info(f"x: {x:+7.4f} m, y: {y:+7.4f} m, theta: {np.rad2deg(theta):+9.4f} deg")

        pos = [x, y, 0.0]
        ori = [0.0, 0.0, theta]
        delta_pose = create_pose_matrix(pos, ori)

        root_to_base_pose = root_to_base_pose @ delta_pose

        perturbed_pos = root_to_base_pose[:3, 3].tolist()
        perturbed_quat = R.from_matrix(root_to_base_pose[:3, :3]).as_quat().tolist()

        return perturbed_pos, perturbed_quat