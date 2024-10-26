#!/usr/bin/env python

import argparse
import matplotlib.pyplot as plt
import numpy as np

GRAVITY = 9.81  # m/s^2

# in meters
lengths = {"base": 0.1, "main_boom": 0.115, "second_boom": 0.105, "wrist": 0.07, "gripper": 0.08}

# in meters
center_of_masses = {
    "main_boom": lengths["main_boom"] / 2,
    "second_boom": lengths["second_boom"] / 2,
    "wrist": lengths["wrist"] / 2,
    "gripper": lengths["gripper"] / 2,
}

# in kg
weights = {
    "main_boom": 0.167,
    "second_boom": 0.142,
    "wrist": 0.106,
    "gripper": 0.151,
    "motor_base": 0.083,
    "motor_main_boom": 0.083,
    "motor_second_boom": 0.083,
    "motor_wrist1": 0.083,
    "motor_wrist2": 0.083,
    "motor_gripper": 0.083,
}

# in Nm
torques = {
    "base_motor": 2.94,  # 1.62
    "main_boom_motor": 2.94,
    "second_boom_motor": 2.94,
    "wrist_motor_1": 2.94,
    "wrist_motor_2": 2.94,
}

# in degrees
angle_limits = {
    "main_boom": [-20, 185],
    "second_boom": [5, -170],
    "wrist_vertical": [-90, 90],
    "wrist_horizontal": [-180, 180],
}

# Define DH parameters for each link (theta, d, a, alpha)
dh_params = {
    "main_boom": {"a": lengths["main_boom"], "alpha": 0, "d": 0},
    "second_boom": {"a": lengths["second_boom"], "alpha": 0, "d": 0},
    "wrist": {"a": lengths["wrist"], "alpha": 0, "d": 0},
    "gripper": {"a": lengths["gripper"], "alpha": 0, "d": 0},
}


def dh_transform(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
    """Compute a transformation matrix given DH parameters"""
    theta, alpha = np.radians(theta), np.radians(alpha)
    return np.array(
        [
            [np.cos(theta), -np.sin(theta) * np.cos(alpha), np.sin(theta) * np.sin(alpha), a * np.cos(theta)],
            [np.sin(theta), np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
            [0, np.sin(alpha), np.cos(alpha), d],
            [0, 0, 0, 1],
        ]
    )


def forward_kinematics(angles: list) -> tuple:
    """Compute forward kinematics for the robot arm given joint angles"""
    theta_main_boom, theta_second_boom, theta_wrist_vertical, theta_wrist_horizontal = angles
    T_base = np.eye(4)
    T_main_boom = dh_transform(theta_main_boom, **dh_params["main_boom"])
    T_second_boom = T_main_boom @ dh_transform(theta_second_boom, **dh_params["second_boom"])
    T_wrist = T_second_boom @ dh_transform(theta_wrist_vertical, **dh_params["wrist"])
    T_gripper = T_wrist @ dh_transform(theta_wrist_horizontal, **dh_params["gripper"])
    return T_base, T_main_boom, T_second_boom, T_wrist, T_gripper


def get_x_dist_center_of_mass(T: np.ndarray, part: str) -> float:
    """Get the x distance from the center of mass for a given part"""
    length, com = lengths[part], center_of_masses[part]
    angle = np.arctan2(T[1, 0], T[0, 0])
    return T[0, 3] - np.cos(angle) * (length - com)


def calculate_torques(angles: list, payload: float) -> tuple:
    """Calculate the torque on each motor joint given joint angles and payload weight"""
    _, T_main_boom, T_second_boom, T_wrist, T_gripper = forward_kinematics(angles)
    forces = {part: weights[part] * GRAVITY for part in weights}
    forces["payload"] = payload * GRAVITY

    x_main_boom, x_second_boom, x_wrist, x_gripper = (
        T[0, 3] for T in [T_main_boom, T_second_boom, T_wrist, T_gripper]
    )

    def calculate_torque(force, force_position, joint_position):
        return force * abs(force_position - joint_position) * (-1 if force_position < joint_position else 1)

    torque_main_boom_joint = (
        forces["main_boom"] * get_x_dist_center_of_mass(T_main_boom, "main_boom")
        + forces["second_boom"] * get_x_dist_center_of_mass(T_second_boom, "second_boom")
        + forces["wrist"] * get_x_dist_center_of_mass(T_wrist, "wrist")
        + forces["gripper"] * get_x_dist_center_of_mass(T_gripper, "gripper")
        + forces["motor_second_boom"] * x_main_boom
        + forces["motor_wrist1"] * x_second_boom
        + forces["motor_wrist2"] * x_wrist
        + forces["motor_gripper"] * x_wrist
        + forces["payload"] * x_gripper
    )

    # Torque calculation accounting for torque direction based on the relative position
    # since the rotation point may not align with [0,0]
    torque_second_boom_joint = sum(
        [
            calculate_torque(
                forces["second_boom"],
                abs(get_x_dist_center_of_mass(T_second_boom, "second_boom")),
                x_main_boom,
            ),
            calculate_torque(forces["wrist"], abs(get_x_dist_center_of_mass(T_wrist, "wrist")), x_main_boom),
            calculate_torque(
                forces["gripper"], abs(get_x_dist_center_of_mass(T_gripper, "gripper")), x_main_boom
            ),
            calculate_torque(forces["motor_wrist1"], x_second_boom, x_main_boom),
            calculate_torque(forces["motor_wrist2"], x_wrist, x_main_boom),
            calculate_torque(forces["motor_gripper"], x_wrist, x_main_boom),
            calculate_torque(forces["payload"], x_gripper, x_main_boom),
        ]
    )

    torque_wrist = sum(
        [
            calculate_torque(
                forces["wrist"], abs(get_x_dist_center_of_mass(T_wrist, "wrist")), x_second_boom
            ),
            calculate_torque(
                forces["gripper"], abs(get_x_dist_center_of_mass(T_gripper, "gripper")), x_second_boom
            ),
            calculate_torque(forces["motor_wrist2"], x_wrist, x_second_boom),
            calculate_torque(forces["motor_gripper"], x_wrist, x_second_boom),
            calculate_torque(forces["payload"], x_gripper, x_second_boom),
        ]
    )

    return abs(torque_main_boom_joint), abs(torque_second_boom_joint), abs(torque_wrist)


def check_motor_limits(angles: list, payload: float) -> tuple:
    """Check if the torque on each motor is within the specified limits"""
    torque_main_boom, torque_second_boom, torque_wrist = calculate_torques(angles, payload)
    return (
        torque_main_boom <= torques["main_boom_motor"],
        torque_second_boom <= torques["second_boom_motor"],
        torque_wrist <= torques["wrist_motor_1"],
    )


def plot_robot_position_and_limits(angles: list, payload: float):
    """Plot the robot's position and valid reach points within joint limits"""
    _, T_main_boom, T_second_boom, T_wrist, T_gripper = forward_kinematics(angles)

    # Extract x and y positions from transformations
    positions = np.array([[0, 0], T_main_boom[:2, 3], T_second_boom[:2, 3], T_wrist[:2, 3], T_gripper[:2, 3]])

    valid_reach = []

    # Generate valid reach points within joint limits
    for main_angle in np.linspace(*angle_limits["main_boom"], 50):
        for second_angle in np.linspace(*angle_limits["second_boom"], 50):
            for wrist_angle in np.linspace(*angle_limits["wrist_vertical"], 50):
                joint_angles = [main_angle, second_angle, wrist_angle, 0]
                _, _, _, _, T_gripper = forward_kinematics(joint_angles)

                if all(check_motor_limits(joint_angles, payload)):
                    gripper_x, gripper_y = T_gripper[:2, 3]
                    valid_reach.append((gripper_x, gripper_y))

    plt.figure(figsize=(6, 6))
    plt.scatter(*zip(*valid_reach, strict=False), color="yellow", label="Valid Reach Points")
    plt.plot(positions[:, 0], positions[:, 1], "-o", label="Robot Links")
    plt.grid(True)
    plt.legend()
    plt.title("Robot Position")
    plt.show()


def calculate_max_payload(angles: list) -> float:
    """Calculate the maximum payload that can be handled at the given joint angles"""
    torque_main_boom, torque_second_boom, torque_wrist = calculate_torques(angles, 0)
    _, T_main_boom, T_second_boom, _, T_gripper = forward_kinematics(angles)
    x_main_boom = T_main_boom[0, 3]
    x_second_boom = T_second_boom[0, 3]
    x_gripper = T_gripper[0, 3]

    max_payload_main_boom = (torques["main_boom_motor"] - torque_main_boom) / abs(x_gripper)
    max_payload_second_boom = (torques["second_boom_motor"] - torque_second_boom) / abs(
        x_gripper - x_main_boom
    )
    max_payload_wrist = (torques["wrist_motor_1"] - torque_wrist) / abs(x_gripper - x_second_boom)

    max_torque = min(max_payload_main_boom, max_payload_second_boom, max_payload_wrist)

    return max_torque / GRAVITY


if __name__ == "__main__":
    default_angles = [110, -75, -30, 0]
    default_payload = 0.0 

    parser = argparse.ArgumentParser(description="Robot arm simulation")
    parser.add_argument(
        "main_boom_angle",
        type=float,
        nargs="?",
        default=default_angles[0],
        help=f"Angle of the main boom in degrees (default: {default_angles[0]})",
    )
    parser.add_argument(
        "second_boom_angle",
        type=float,
        nargs="?",
        default=default_angles[1],
        help=f"Angle of the second boom in degrees (default: {default_angles[1]})",
    )
    parser.add_argument(
        "wrist_angle",
        type=float,
        nargs="?",
        default=default_angles[2],
        help=f"Vertical angle of the wrist in degrees (default: {default_angles[2]})",
    )
    
    parser.add_argument(
        "--payload",
        type=float,
        nargs="?", 
        const=default_payload,
        default=default_payload,  
        help=f"Payload in kg (default: {default_payload})",
    )

    args = parser.parse_args()
    angles = [args.main_boom_angle, args.second_boom_angle, args.wrist_angle, 0]
    payload = calculate_max_payload(angles)
    print(f"Max Payload for angles {angles}: {payload:.2f} kg")
    print(f"Max Payload fully stretched: {calculate_max_payload([0, 0, 0, 0]):.2f} kg")
    print("Plotting...")
    plot_robot_position_and_limits(angles, payload=args.payload)
