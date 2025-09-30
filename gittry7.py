import cv2
import numpy as np
import math
from collections import deque

class LaneDetection:
    def __init__(self, video_path):
        self.video = cv2.VideoCapture(video_path)
        self.angle_history = deque(maxlen=10)  # median smoothing

    def detect_edges(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        return cv2.Canny(blurred, 50, 150)

    def region_of_interest(self, edges):
        height, width = edges.shape
        mask = np.zeros_like(edges)
        vertices = np.array([[(0, height), (width/3, height/2),
                              (width*2/3, height/2), (width, height)]], dtype=np.int32)
        cv2.fillPoly(mask, [vertices], 255)
        return cv2.bitwise_and(edges, mask)

    def detect_line_segments(self, cropped_edges):
        rho, theta, threshold = 1, np.pi/180, 30
        return cv2.HoughLinesP(cropped_edges, rho, theta, threshold, np.array([]),
                               minLineLength=5, maxLineGap=150)

    def average_slope_intercept(self, frame, line_segments):
        lane_lines = []
        if line_segments is None:
            return lane_lines

        height, width, _ = frame.shape
        left_fit, right_fit = [], []
        boundary = 1/3
        left_boundary, right_boundary = width*(1-boundary), width*boundary

        for line_segment in line_segments:
            for x1, y1, x2, y2 in line_segment:
                if x1 == x2:
                    continue
                slope = (y2-y1)/(x2-x1)
                intercept = y1 - slope*x1
                if slope < 0 and x1 < left_boundary and x2 < left_boundary:
                    left_fit.append((slope, intercept))
                elif slope > 0 and x1 > right_boundary and x2 > right_boundary:
                    right_fit.append((slope, intercept))

        if left_fit:
            left_avg = np.average(left_fit, axis=0)
            lane_lines.append(self.make_points(frame, left_avg))
        if right_fit:
            right_avg = np.average(right_fit, axis=0)
            lane_lines.append(self.make_points(frame, right_avg))

        return lane_lines

    def make_points(self, frame, line):
        height, width, _ = frame.shape
        slope, intercept = line
        y1, y2 = height, int(height/2)
        if slope == 0: slope = 0.1
        x1, x2 = int((y1-intercept)/slope), int((y2-intercept)/slope)
        return [[x1, y1, x2, y2]]

    def draw_lane_area(self, frame, lane_lines):
        if len(lane_lines) == 2:
            left, right = lane_lines[0][0], lane_lines[1][0]
            pts = np.array([[left[0], left[1]], [left[2], left[3]],
                            [right[2], right[3]], [right[0], right[1]]], np.int32)
            cv2.fillPoly(frame, [pts], (0, 255, 64))
        return frame

    def display_lines(self, frame, lines, color=(0,255,0), width=6):
        line_img = np.zeros_like(frame)
        if lines is not None:
            for line in lines:
                for x1,y1,x2,y2 in line:
                    cv2.line(line_img, (x1,y1), (x2,y2), color, width)
        return cv2.addWeighted(frame, 0.8, line_img, 1, 1)

    def display_heading_line(self, frame, steering_angle, color=(0,0,255), width=5):
        heading_img = np.zeros_like(frame)
        height, width_f, _ = frame.shape
        angle_rad = steering_angle * math.pi / 180.0

        x1, y1 = width_f//2, height
        x2 = int(x1 + height/2 * math.tan(angle_rad))
        y2 = int(height/2)

        cv2.line(heading_img, (x1,y1), (x2,y2), color, width)
        return cv2.addWeighted(frame, 0.8, heading_img, 1, 1)

    # ✅ New function
    def measure_curvature_and_angle(self, left_line, right_line, frame_width, frame_height):
        if left_line is None or right_line is None:
            return None, None, None

        xm_per_pix = 3.7 / 700  # lane width scaling

        # Lane center vs vehicle center
        lane_center = (left_line[0] + right_line[0]) // 2
        vehicle_center = frame_width // 2
        offset = (lane_center - vehicle_center) * xm_per_pix

        # Reverse sign → Left = negative, Right = positive
        lookahead_y = 15.0
        steering_angle = -np.arctan2(offset, lookahead_y) * 180 / np.pi

        # Median smoothing
        self.angle_history.append(steering_angle)
        steering_angle = float(np.median(self.angle_history))

        # Clamp tiny drift
        if abs(steering_angle) < 2.0:
            steering_angle = 0.0

        # Curvature estimation
        if steering_angle == 0.0:
            curvature = 5000.0   # treat straight as 5 km radius
        else:
            curvature = abs(50 / (steering_angle / 45.0))

        curvature = np.clip(curvature, 100.0, 5000.0)

        return curvature, steering_angle, offset

    def driving_direction(self, steering_angle, curvature):
        if curvature and curvature > 500:
            return "Go Straight"
        if steering_angle > 5:
            return "Right"
        elif steering_angle < -5:
            return "Left"
        else:
            return "Go Straight"

    def run(self):
        while True:
            ret, frame = self.video.read()
            if not ret: break

            h, w, _ = frame.shape

            edges = self.detect_edges(frame)
            roi = self.region_of_interest(edges)
            line_segments = self.detect_line_segments(roi)
            lane_lines = self.average_slope_intercept(frame, line_segments)

            frame = self.draw_lane_area(frame, lane_lines)
            frame = self.display_lines(frame, lane_lines)

            curvature, steering_angle, offset = None, 0, 0
            if len(lane_lines) == 2:
                left_line, right_line = lane_lines[0][0], lane_lines[1][0]
                curvature, steering_angle, offset = self.measure_curvature_and_angle(
                    left_line, right_line, w, h
                )

            frame = self.display_heading_line(frame, steering_angle)

            direction = self.driving_direction(steering_angle, curvature)

            if curvature:
                cv2.putText(frame, f"Radius: {curvature:.2f}m", (30,60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            cv2.putText(frame, f"Steering: {steering_angle:.1f} deg", (30,100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.putText(frame, f"Direction: {direction}", (30,140),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

            cv2.imshow("Lane Detection with Curvature + Steering", frame)
            if cv2.waitKey(1) & 0xFF == 27: break

        self.video.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    ld = LaneDetection(r"D:\bis\right.mp4")
    ld.run()
