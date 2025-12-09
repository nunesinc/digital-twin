import cv2
import numpy as np
import csv
from ultralytics import YOLO
import supervision as sv
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
AREA_WIDTH_M = 6.0
AREA_HEIGHT_M = 5.0
VIDEO_PATH = r"D:\Stuff\ok\MASTER\VC\cam1_2min.mp4"
MODEL_PATH = r"D:\Stuff\ok\MASTER\VC\yolov8l.pt"
# MODEL_PATH = r"D:\Stuff\ok\MASTER\VC\yolo11m.pt"

# Calibration points (filled by mouse callback if needed)
calibration_points = []
homography_matrix = None

def draw_grid(frame, H, width_m, height_m, step_m=1.0):
    """Draws a real-world grid on the frame using homography."""
    if H is None:
        return frame
    
    # Inverse homography: real world (X,Y) -> pixels
    H_inv = np.linalg.inv(H)

    # Draw vertical grid lines (X fixed, Y varies)
    for x in np.arange(0, width_m + 0.001, step_m):
        pts_real = np.array([[[x, 0]], [[x, height_m]]], dtype=np.float32)
        pts_img = cv2.perspectiveTransform(pts_real, H_inv)
        p1, p2 = pts_img[0][0], pts_img[1][0]
        cv2.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 255, 0), 1)

    # Draw horizontal grid lines (Y fixed, X varies)
    for y in np.arange(0, height_m + 0.001, step_m):
        pts_real = np.array([[[0, y]], [[width_m, y]]], dtype=np.float32)
        pts_img = cv2.perspectiveTransform(pts_real, H_inv)
        p1, p2 = pts_img[0][0], pts_img[1][0]
        cv2.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (255, 0, 0), 1)

    return frame


def click_event(event, x, y, flags, param):
    """Mouse callback to collect 4 calibration points. Start from top-left, clockwise."""
    global calibration_points
    if event == cv2.EVENT_LBUTTONDOWN:
        calibration_points.append((x, y))
        print(f"Point selected: {x},{y}")

def get_homography(frame):
    """Collect 4 points and compute homography for perspective correction."""
    global calibration_points, homography_matrix
    calibration_points = []
    clone = frame.copy()
    cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Calibration", 800, 600)
    cv2.setMouseCallback("Calibration", click_event)

    while True:
        cv2.imshow("Calibration", clone)
        key = cv2.waitKey(1) & 0xFF
        if len(calibration_points) == 4:
            break
        if key == 27:  # ESC to cancel
            break
    cv2.destroyWindow("Calibration")

    if len(calibration_points) != 4:
        print("Calibration canceled.")
        return None

    # Real-world square coordinates (meters, scaled to pixels)
    dst_pts = np.array([
        [0, AREA_HEIGHT_M],             # bottom-left
        [AREA_WIDTH_M, AREA_HEIGHT_M], # bottom-right
        [AREA_WIDTH_M, 0], # top-right
        [0, 0]   # top-left
    ], dtype=np.float32)


    src_pts = np.array(calibration_points, dtype=np.float32)
    H, _ = cv2.findHomography(src_pts, dst_pts)
    homography_matrix = H
    print("Homography matrix:\n", H)
    return H

def pixel_to_meters(point, H, frame_shape):
    """Apply homography if available, else simple scaling with Y inversion."""
    H_img, W_img = frame_shape[:2]
    if H is not None:
        pts = np.array([[point]], dtype=np.float32)
        dst = cv2.perspectiveTransform(pts, H)[0][0]
        return dst[0], dst[1]
    else:
        # Simple linear scaling
        x_pix, y_pix = point
        real_x = x_pix * (AREA_WIDTH_M / W_img)
        real_y = y_pix * (AREA_HEIGHT_M / H_img)
        return real_x, real_y

def main():
    global homography_matrix
    model = YOLO(MODEL_PATH)
    box_annotator = sv.BoxAnnotator(thickness=2, text_thickness=1, text_scale=0.5)

    # Prepare CSV
    csv_file = open("positions.csv", "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["frame", "id", "x_m", "y_m"])

    # Dict to store trajectories
    trajectories = {}

    # Setup windows
    cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Tracking", 1000, 700)

    # # Matplotlib live plot
    # plt.ion()
    # fig, ax = plt.subplots()
    # ax.set_xlim(0, AREA_WIDTH_M)
    # ax.set_ylim(0, AREA_HEIGHT_M)
    # ax.set_xlabel("X (m)")
    # ax.set_ylabel("Y (m)")
    # ax.set_title("Trajectories")
    
    frame_count = 0
    for result in model.track(source=VIDEO_PATH, show=False, stream=True, agnostic_nms=True):
        frame = result.orig_img
        frame_count += 1

        if homography_matrix is None and frame_count == 1:
            # Ask once for calibration if desired
            print("Click 4 corners of the area (starting top-left, clockwise). Press ESC to skip.")
            get_homography(frame)

        detections = sv.Detections.from_yolov8(result)
        if result.boxes.id is not None:
            detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)
        detections = detections[detections.class_id == 0]  # only persons

        labels = []
        for (xyxy, conf, class_id, tracker_id) in detections:
            x1, y1, x2, y2 = xyxy
            px = (x1 + x2) / 2
            py = (y1 + y2) / 2
            real_x, real_y = pixel_to_meters((px, py), homography_matrix, frame.shape)

            # Save trajectory
            trajectories.setdefault(int(tracker_id), []).append((real_x, real_y))

            # Save to CSV
            writer.writerow([frame_count, tracker_id, f"{real_x:.2f}", f"{real_y:.2f}"])

            labels.append(f"ID {tracker_id} ({real_x:.2f},{real_y:.2f})m")
            cv2.circle(frame, (int(px), int(py)), 8, (255, 0, 0), -1)

        frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)

        # Draw the real-world grid overlay (after calibration)
        frame = draw_grid(frame, homography_matrix, AREA_WIDTH_M, AREA_HEIGHT_M, step_m=1.0)

        cv2.imshow("Tracking", frame)

        if cv2.waitKey(1) == 27:  # ESC to quit
            break

        # # Press 'q' to quit, or space to go frame-by-frame
        # key = cv2.waitKey(0)  # 0 = wait indefinitely (frame by frame)
        # if key == ord("q"):
        #     break

        # # Update trajectories plot
        # ax.clear()
        # ax.set_xlim(0, AREA_WIDTH_M)
        # ax.set_ylim(0, AREA_HEIGHT_M)
        # ax.set_xlabel("X (m)")
        # ax.set_ylabel("Y (m)")
        # ax.set_title("Trajectories")
        # for pid, pts in trajectories.items():
        #     xs, ys = zip(*pts)
        #     ax.plot(xs, ys, marker="o", label=f"ID {pid}")
        # ax.legend(loc="upper right", fontsize="small")
        # plt.pause(0.001)

    csv_file.close()
    cv2.destroyAllWindows()
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
