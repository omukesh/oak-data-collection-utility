import cv2
import depthai as dai
import numpy as np
import time
from datetime import datetime
from pathlib import Path

# 1. Configuration
RES_OPTIONS = {
    0: (dai.ColorCameraProperties.SensorResolution.THE_1080_P, "1080p"),
    1: (dai.ColorCameraProperties.SensorResolution.THE_4_K, "4K"),
    2: (dai.ColorCameraProperties.SensorResolution.THE_12_MP, "12MP")
}

# ArUco Setup
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# State Variables
current_mode = 0  
current_res_idx = 0 
show_aruco = False 
restart_needed = True
is_processing = False 
device = None
q_video = None
cap_sim = None
detected_ids_list = None

def get_save_path(ids):
    date_str = datetime.now().strftime("%Y-%m-%d")
    if ids is None or len(ids) == 0:
        folder_name = "1000_tray"
    else:
        ids_flat = sorted([str(i[0]) for i in ids])
        folder_name = "_".join(ids_flat) + "_tray"
    path = Path("data") / date_str / folder_name
    path.mkdir(parents=True, exist_ok=True)
    return path

def create_pipeline(res_enum):
    pipeline = dai.Pipeline()
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setResolution(res_enum)
    cam.setInterleaved(False)
    cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("video")
    cam.video.link(xout.input)
    return pipeline

def switch_hardware_res(idx):
    global device, q_video
    if current_mode == 0: return
    res_enum = RES_OPTIONS[idx][0]
    if device: device.close()
    device = dai.Device(create_pipeline(res_enum))
    q_video = device.getOutputQueue("video", 4, False)

# UI Callbacks
def on_mode_change(val): global current_mode, restart_needed; current_mode = val; restart_needed = True
def on_res_change(val): global current_res_idx, restart_needed; current_res_idx = val; restart_needed = True

# --- NEW WINDOW SETUP ---
window_name = "MDL Pro Collector"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) # Allow manual resizing
cv2.resizeWindow(window_name, 1080, 960)        # Force your 1080x960 size

cv2.createTrackbar("MODE: 0-Sim 1-OAK", window_name, 0, 1, on_mode_change)
cv2.createTrackbar("RES: 0,1,2, 3(Mode7)", window_name, 0, 3, on_res_change)

try:
    while True:
        if restart_needed:
            is_processing = True
            if device: device.close(); device = None
            if cap_sim: cap_sim.release(); cap_sim = None
            
            temp_idx = 0 if current_res_idx == 3 else current_res_idx
            res_enum, res_name = RES_OPTIONS[temp_idx]
            
            if current_mode == 1:
                try:
                    device = dai.Device(create_pipeline(res_enum))
                    q_video = device.getOutputQueue("video", 4, False)
                except:
                    current_mode = 0; cv2.setTrackbarPos("MODE: 0-Sim 1-OAK", window_name, 0)
            else:
                cap_sim = cv2.VideoCapture(0)
            restart_needed = False
            is_processing = False

        frame = None
        if current_mode == 1 and q_video:
            frame = q_video.get().getCvFrame()
        elif cap_sim:
            ret, sim_frame = cap_sim.read()
            if ret: 
                temp_idx = 0 if current_res_idx == 3 else current_res_idx
                # Simulation scaling
                frame = cv2.resize(sim_frame, (1920, 1080) if temp_idx==0 else (3840, 2160))
            else:
                cap_sim.set(cv2.CAP_PROP_POS_FRAMES, 0)

        if frame is not None:
            # Resize the display frame to fit perfectly in your 1080x960 window
            # We leave a little room for the trackbars at the bottom
            display_frame = cv2.resize(frame, (1080, 800)) 
            
            # Pad the bottom with white/gray so the trackbars don't cover the image
            canvas = np.zeros((960, 1080, 3), dtype=np.uint8) + 50 # Dark gray background
            canvas[0:800, 0:1080] = display_frame
            
            corners, ids, rejected = detector.detectMarkers(display_frame)
            detected_ids_list = ids
            
            # --- UI: STATUS BUBBLE (Relocated for 1080 width) ---
            if current_mode == 0:
                bubble_color = (255, 255, 255) # White for Simulation
                status_text = "SIM-MODE"
            else:
                bubble_color = (0, 0, 255) if is_processing else (0, 255, 0)
                status_text = "BUSY" if is_processing else "READY"

            # Bubble at Top Right of the image area
            cv2.circle(canvas, (1040, 40), 12, bubble_color, -1)
            cv2.putText(canvas, status_text, (940, 45), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, bubble_color, 2)

            # --- UI: ARUCO DISPLAY ---
            if show_aruco:
                if ids is not None and len(ids) > 0:
                    # Draw on the 'canvas' (image part)
                    cv2.aruco.drawDetectedMarkers(canvas[0:800, 0:1080], corners, ids)
                    id_str = " & ".join([str(i[0]) for i in ids])
                else:
                    id_str = "1000"
                cv2.putText(canvas, f"IDs: {id_str}", (20, 90), 2, 0.8, (0, 255, 0), 2)

            # --- UI: MODE OVERLAY ---
            res_display = "Mode 7 (Multi)" if current_res_idx == 3 else RES_OPTIONS[current_res_idx][1]
            cv2.putText(canvas, f"{'REAL' if current_mode==1 else 'SIM'} | {res_display}", 
                        (20, 40), 2, 0.7, (0, 255, 255), 2)
            
            cv2.imshow(window_name, canvas)

        key = cv2.waitKey(1)
        if key == ord('q'): break
        elif key == ord('a'): show_aruco = not show_aruco
        elif key in [ord('s'), ord('b')] and frame is not None and not is_processing:
            is_processing = True
            save_dir = get_save_path(detected_ids_list)
            num_snaps = 5 if key == ord('b') else 1
            caps_to_run = [0, 1, 2] if current_res_idx == 3 else [current_res_idx]
            
            for r_idx in caps_to_run:
                if current_res_idx == 3 and current_mode == 1:
                    switch_hardware_res(r_idx)
                    time.sleep(1.0)
                    frame = q_video.get().getCvFrame()
                
                res_name = RES_OPTIONS[r_idx][1]
                for i in range(num_snaps):
                    ts = datetime.now().strftime("%H-%M-%S-%f")[:-3]
                    prefix = "burst" if key == ord('b') else "snap"
                    fn = f"{prefix}_{res_name}_{ts}.png"
                    cv2.imwrite(str(save_dir / fn), frame)
                    print(f"Saved: {fn}")
                    if num_snaps > 1: time.sleep(0.05)

            if current_res_idx == 3 and current_mode == 1: restart_needed = True 
            is_processing = False

finally:
    if device: device.close()
    if cap_sim: cap_sim.release()
    cv2.destroyAllWindows()
