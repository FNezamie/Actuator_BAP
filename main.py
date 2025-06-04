import os
import sys
import time
import math

import cv2 as cv

import processor.algorithms.colored_frame_difference as proc_color
import processor.algorithms.frame_difference as proc_naive
import processor.algorithms.dummy_algorithm as dummy

import pantilthat as pth
import controller.pid 

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
    setpoint = 0

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

    print("[INFO] Starting tracking loop. Press Ctrl+C to exit.")
    pth.idle_timeout(FRAME_RATE)
    iter_machine = dummy.DummyMeasurements()
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
                measurement_y, camera_preview_output, _ = proc_color.process_frames(camera_prev_gray, current_gray_frame, current_frame, color_hues["Rose"], hue_tolerance=10)
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
                if abs(setpoint - measurement_y) > 0 and measurement_y >= FRAME_HEIGHT / 8:
                    cv.line(current_frame, (0, int(setpoint)), (current_frame.shape[1], int(setpoint)), (0, 0, 255), 2)
                    error = setpoint - measurement_y
                    P = 0.25
                    delta_deg = -P * error * deg_per_px
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
