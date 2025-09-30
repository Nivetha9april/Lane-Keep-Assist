import cv2
import numpy as np
import time
from collections import deque
import psutil   # for memory usage

# ---------------- Lane Detection ---------------- #

last_left, last_right = None, None
left_history = deque(maxlen=5)
right_history = deque(maxlen=5)
angle_history = deque(maxlen=15)


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, [vertices], 255)
    return cv2.bitwise_and(img, mask)


def smooth_line(line, history):
    if line is not None:
        history.append(line)
    if not history:
        return None
    avg = np.mean(history, axis=0).astype(int).tolist()
    return avg


def draw_lane_lines(img, left_line, right_line, color=(0, 255, 0)):
    global last_left, last_right
    if left_line is None:
        left_line = last_left
    else:
        last_left = left_line
    if right_line is None:
        right_line = last_right
    else:
        last_right = right_line

    if left_line is None or right_line is None:
        return img, None, None

    left_line = smooth_line(left_line, left_history)
    right_line = smooth_line(right_line, right_history)

    if left_line is None or right_line is None:
        return img, None, None

    line_img = np.zeros_like(img)
    pts = np.array([[(
        left_line[0], left_line[1]),
        (left_line[2], left_line[3]),
        (right_line[2], right_line[3]),
        (right_line[0], right_line[1])
    ]], dtype=np.int32)

    cv2.fillPoly(line_img, pts, color)
    return cv2.addWeighted(img, 0.8, line_img, 0.5, 0.0), left_line, right_line


def measure_curvature_and_angle(left_line, right_line, frame_width, frame_height):
    if left_line is None or right_line is None:
        return None, None, None

    xm_per_pix = 3.7 / 700

    # Lane center vs vehicle center
    lane_center = (left_line[0] + right_line[0]) // 2
    vehicle_center = frame_width // 2
    offset = (lane_center - vehicle_center) * xm_per_pix

    # ✅ FIX: Reverse sign so LEFT = negative, RIGHT = positive
    lookahead_y = 15.0
    steering_angle = -np.arctan2(offset, lookahead_y) * 180 / np.pi

    # Smooth angle
    angle_history.append(steering_angle)
    steering_angle = float(np.median(angle_history))

    # Clamp small drift
    if abs(steering_angle) < 2.0:
        steering_angle = 0.0

    # Curvature estimation (clamped)
    if steering_angle == 0.0:
        curvature = 5000.0   # treat straight as ~5 km radius
    else:
        curvature = abs(50 / (steering_angle / 45.0))

    curvature = np.clip(curvature, 100.0, 5000.0)  # min 100 m, max 5 km

    return curvature, steering_angle, offset


def draw_curvature_arc(img, curvature, steering_angle, offset, frame_height):
    """Draw a short arc representing road curvature ahead"""
    h, w = img.shape[:2]
    overlay = img.copy()

    center_x = w // 2
    base_y = h - 50

    if steering_angle == 0.0:  # Straight line
        cv2.line(overlay, (center_x, base_y),
                 (center_x, base_y - 150), (255, 0, 0), 3)
    else:
        radius = int(abs(curvature))
        radius = min(radius, 800)  # keep realistic

        direction = np.sign(steering_angle)  # -1 = LEFT, +1 = RIGHT
        circle_center = (int(center_x + direction * radius), base_y)

        # Draw only a small arc segment instead of the full circle
        start_angle = 200 if direction > 0 else -20
        end_angle = 260 if direction > 0 else 40
        cv2.ellipse(overlay, circle_center, (radius, radius),
                    0, start_angle, end_angle, (255, 0, 0), 3)

    return cv2.addWeighted(overlay, 0.7, img, 0.3, 0)


def lane_pipeline(image):
    height, width = image.shape[:2]
    roi_vertices = np.array([
        (0, height),
        (width // 2, int(height * 0.55)),
        (width, height)
    ], dtype=np.int32)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    cropped = region_of_interest(edges, roi_vertices)

    lines = cv2.HoughLinesP(cropped, 2, np.pi / 180, 50,
                            minLineLength=30, maxLineGap=50)

    if lines is None:
        return image, None, None, None

    lx, ly, rx, ry = [], [], [], []
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1) if x2 != 0 else 0
            if abs(slope) < 0.5:
                continue
            if slope < 0:
                lx += [x1, x2]; ly += [y1, y2]
            else:
                rx += [x1, x2]; ry += [y1, y2]

    min_y, max_y = int(height * 0.6), height
    l_line, r_line = None, None
    if lx and ly:
        pl = np.poly1d(np.polyfit(ly, lx, 1))
        l_line = [int(pl(max_y)), max_y, int(pl(min_y)), min_y]
    if rx and ry:
        pr = np.poly1d(np.polyfit(ry, rx, 1))
        r_line = [int(pr(max_y)), max_y, int(pr(min_y)), min_y]

    lane_img, l_line, r_line = draw_lane_lines(image, l_line, r_line)
    curvature, angle, offset = measure_curvature_and_angle(l_line, r_line, width, height)

    if curvature and angle is not None:
        if angle == 0.0:
            direction = "STRAIGHT"
        elif angle > 0:
            direction = "RIGHT"
        else:
            direction = "LEFT"

        lane_img = draw_curvature_arc(lane_img, curvature, angle, offset, height)

        if curvature >= 850:
            curv_text = "Straight"
        else:
            curv_text = f"{curvature:.0f} m"

        # Show signed angle
        cv2.putText(lane_img, f"Curvature: {curv_text}",
                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(lane_img, f"Steering: {angle:.1f}° ({direction})",
                    (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    return lane_img, angle, curvature, offset


# ---------------- Run ---------------- #
def process_video(video_path=r"D:\bis\left.mp4"):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: cannot open video")
        return

    target_fps = 30
    frame_time = 1.0 / target_fps
    prev_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (1280, 720))
        lane_frame, steer, curv, offset = lane_pipeline(frame.copy())

        # FPS & Memory usage
        now = time.time()
        fps = 1 / (now - prev_time)
        prev_time = now
        mem_usage = psutil.Process().memory_info().rss / (1024 * 1024)  # in MB

        # Overlay stats
        cv2.putText(lane_frame, f"FPS: {fps:.1f}", (30, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(lane_frame, f"RAM: {mem_usage:.1f} MB", (30, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Console logs
        if steer is not None:
            print(f"[INFO] Steering={steer:.2f}°, Curvature={curv:.1f}m, Offset={offset:.2f}m, FPS={fps:.1f}, RAM={mem_usage:.1f}MB")

        cv2.imshow("Lane Detection with Curvature", lane_frame)
        time.sleep(frame_time)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    process_video()
