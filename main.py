import os
import sys
import time
import math

import cv2 as cv
import numpy as np

import processor.algorithms.colored_frame_difference as proc_color
import processor.algorithms.frame_difference as proc_naive
import processor.algorithms.dummy_algorithm as dummy

import pantilthat as pth
import controller.pd as pd_mod
import controller.ekf as ekf_mod

from processor.camera import CameraStream, SharedObject
from threading import Thread


def clamp(value, vmin, vmax):
    return max(vmin, min(vmax, value))


def tilt(shared_obj):
    while True:
        if shared_obj.is_exit:
            sys.exit(0)
        loop_start = time.monotonic()
        pth.pan(0)
        pth.tilt(shared_obj.current_tilt)

        elapsed = time.monotonic() - loop_start
        sleep_time = shared_obj.LOOP_DT_TARGET - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)



if __name__ == '__main__':
    # Create shared-memory for capturing, processing and tilting
    shared_obj = SharedObject()

    # Initialize camera
    camera = CameraStream(shared_obj)
    camera.start()

    # Frame dimensions and timing
    FRAME_HEIGHT = 640
    FRAME_RATE = 50
    shared_obj.LOOP_DT_TARGET = 1.0 / FRAME_RATE

    # Compute vertical FoV
    SENSOR_HEIGHT_MM = 4.712
    FOCAL_LENGTH_MM = 35
    vfov_rad = 2 * math.atan(SENSOR_HEIGHT_MM / (2 * FOCAL_LENGTH_MM))
    vfov_deg = math.degrees(vfov_rad)
    deg_per_px = vfov_deg / FRAME_HEIGHT
    setpoint = FRAME_HEIGHT / 2

    # PID and servo settings
    SERVO_MIN, SERVO_MAX = -90, 90

    # Initialize PanTilt HAT
    current_tilt = -20
    try:
        pth.servo_enable(2, True)
        pth.tilt(int(current_tilt))
        print("[INFO] Tilt servo initialized.")

    except Exception as e:
        print(f"[ERROR] Could not initialize PanTilt HAT: {e}")
        camera.stop()
        exit()

    # Camera variables
    # Measurement
    camera_preview_output = None
    camera_prev_gray = None

    # FPS Overlay
    camera_prev_time = time.time_ns()
    camera_diff_time = 0
    camera_frame_per_sec = 0
    camera_frame_cnt_in_sec = 0
    camera_is_one_sec_passed = False
    recording_id = time.strftime('%y%m%d%H%M%S', time.gmtime())

    # Colors
    color_hues = {
        "Red": 0,
        "Green": 60,
        "Blue": 120,
        "Cyan": 90,
        "Magenta": 150,
        "Yellow": 30,
        "Amber": 15,
        "Chartreuse": 45,
        "Spring Green": 75,
        "Azure": 105,
        "Violet": 135,
        "Rose": 165
    }

    pd = pd_mod.PD(kp=0.08, kd=0.01, tau=0.02, setpoint=setpoint, deg_per_px=deg_per_px)

    DROP_DIST_M = 4.5
    world_h_m = 2 * DROP_DIST_M * math.tan(vfov_rad / 2)
    px_per_m = FRAME_HEIGHT / world_h_m
    g_pix = 9.81 * px_per_m

    # Ball properties (example values - YOU MUST MEASURE/DETERMINE YOURS)
    mass = 0.0042     # kg (example mass of a dry ping pong ball)
    rho_air = 1.2         # kg/m^3 (air density at sea level, 15°C)
    Cd = 0.47               # Drag coefficient for a sphere (typical value)
    radius_m = 0.025        # m (for a 4cm diameter ball, e.g., ping pong ball)
    A_m2 = np.pi * radius_m**2 # Cross-sectional area m^2
    # k_drag_world = 0.5 * rho_air * Cd * A_m2 (units: kg/m)
    # k_drag_pix = k_drag_world / (px_per_m * mass) (units: 1/pixel)
    k_drag_pix = (0.5 * rho_air * Cd * A_m2 / px_per_m) / mass

    # initialize with the first measurement (we’ll overwrite vy once we get the 2nd sample)
    initial_y   = 0.0
    initial_vy  = 0.0
    P_y0        = 2500.0         # e.g. measurement variance
    P_vy0       = (2*P_y0)/(shared_obj.LOOP_DT_TARGET**2)
    Qy, Qvy     = 900.0, 925.0
    Ry          = 2500.0

    ekf = ekf_mod.EKF_BallTracker(
        initial_y, initial_vy,
        P_y0, P_vy0,
        Qy, Qvy,
        Ry,
        shared_obj.LOOP_DT_TARGET, g_pix, k_drag_pix, px_per_m
    )

    ekf_initialized = False
    prev_meas = None

    print("[INFO] Starting tracking loop. Press Ctrl+C to exit.")
    pth.idle_timeout(FRAME_RATE)
    # iter_machine = dummy.DummyMeasurements()
    try:
        while True:
            loop_start = time.monotonic()

            # 1) Read frame
            current_frame = shared_obj.frame
            current_gray_frame = cv.cvtColor(current_frame, cv.COLOR_RGB2HSV) if current_frame is not None else None

            # 2) Detect object
            if current_frame is None or camera_prev_gray is None:
                measurement_y = None
            else:
                # camera_preview_output, measurement_y = proc_naive.process_frames(camera_prev_gray, current_gray_frame, current_frame)
                measurement_y, camera_preview_output, _ = proc_color.process_frames(camera_prev_gray, current_gray_frame, current_frame, color_hues["Yellow"], hue_tolerance=10)
                # measurement_y, camera_preview_output, _ = iter_machine.next(), current_frame, None

            print(f"info: y: {measurement_y}")
            camera_prev_gray = current_gray_frame

            # 2.1) FPS overlay
            camera_frame_cnt_in_sec += 1
            camera_curr_time = time.time_ns()
            camera_diff_time += (camera_curr_time - camera_prev_time) / 1e6

            if int(camera_diff_time) >= 1000:
                camera_frame_per_sec = camera_frame_cnt_in_sec
                camera_frame_cnt_in_sec = 0
                camera_diff_time = 0
                camera_is_one_sec_passed = True

            if camera_is_one_sec_passed:
                cv.putText(current_frame, f"FPS: {camera_frame_per_sec}", (10, 25), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv.putText(current_frame, f"FPS: (WAITING...)", (10, 25), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            camera_prev_time = camera_curr_time

            # 5) PID update and servo write when active
            if measurement_y is not None:
                if not ekf_initialized:
                    # bootstrap velocity from first two samples
                    if prev_meas is None:
                        prev_meas = measurement_y
                    else:
                        initial_vy = (measurement_y - prev_meas) / shared_obj.LOOP_DT_TARGET
                        ekf.x_hat = np.array([measurement_y, initial_vy])
                        ekf_initialized = True
                else:
                    # 1) EKF prediction
                    ekf.predict()

                    # 2) EKF update with the new pixel measurement
                    ekf.update(measurement_y)

                    # 3) Get the filtered position & velocity
                    y_filt  = ekf.get_current_position()
                    vy_filt = ekf.get_current_velocity()

                    # 4) Predict next-step y for your PID set-point
                    y_next_pred = ekf.get_predicted()

                    if abs(setpoint - y_next_pred) > 0:
                        cv.line(current_frame, (0, int(setpoint)), (current_frame.shape[1], int(setpoint)), (0, 0, 255), 2)
                        delta_deg = pd.update(y_next_pred)
                    else:
                        delta_deg = 0.0

                    desired = current_tilt + delta_deg
                    current_tilt = clamp(desired, SERVO_MIN, SERVO_MAX)
                    print(f"info: tilt: {current_tilt} deg")

            if current_frame is not None:
                cv.imshow(f'[{recording_id}] [Live] Processed Frame', current_frame)

            pth.tilt(current_tilt)

            # 6) Exit & Store frames
            if cv.waitKey(1) & 0xFF == ord('q'):
                shared_obj.is_exit = True

                output_dir = os.path.join("output_frames", recording_id)
                os.makedirs(output_dir, exist_ok=True)

                for i, frame in enumerate(shared_obj.frame_buffer):
                    filename = os.path.join(output_dir, f"frame_{i:06d}.png")
                    print(f'info: storing frames [{i:06d}/{len(shared_obj.frame_buffer)}]')
                    cv.imwrite(filename, frame)
                sys.exit(0)

            # Fix FPS
            elapsed = time.monotonic() - loop_start
            sleep_time = shared_obj.LOOP_DT_TARGET - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[INFO] Exiting, disabling tilt servo.")

    finally:
        pth.servo_enable(2, False)
        camera.stop()
        cv.destroyAllWindows()
