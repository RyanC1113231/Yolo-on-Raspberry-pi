"""
任务3：多目标定位 + 智能加速减速转向 + Flask实时串流
访问：http://100.102.90.76:5000

依赖安装：
    pip install ultralytics opencv-python-headless numpy filterpy flask --break-system-packages
"""

import cv2
import numpy as np
from picamera2 import Picamera2
from filterpy.kalman import KalmanFilter
from dataclasses import dataclass
from typing import List, Tuple
import time
import threading
from flask import Flask, Response, render_template_string

# ─────────────────────────────────────────────
# 配置区
# ─────────────────────────────────────────────
FRAME_WIDTH  = 320
FRAME_HEIGHT = 240
FOCAL_LENGTH = 600.0
REAL_WIDTH   = 45.0    # 人肩宽(cm)
DANGER_DIST  = 80.0    # 紧急停止距离(cm)
SLOW_DIST    = 150.0   # 减速距离(cm)
ROBOT_SPEED  = 30.0    # 机器人当前速度(cm/s)，后期从编码器读


# ─────────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────────
@dataclass
class Target:
    id: int
    bbox: Tuple
    distance: float
    abs_velocity: float
    rel_velocity: float
    state: str
    center: Tuple

@dataclass
class MotionCommand:
    speed: float
    turn: float
    action: str
    reason: str


# ─────────────────────────────────────────────
# 单目测距
# ─────────────────────────────────────────────
def estimate_distance(bbox_width_px: float) -> float:
    if bbox_width_px < 1:
        return 9999.0
    return (REAL_WIDTH * FOCAL_LENGTH) / bbox_width_px


# ─────────────────────────────────────────────
# 卡尔曼滤波器
# ─────────────────────────────────────────────
class TargetKalman:
    def __init__(self, init_dist: float):
        self.kf = KalmanFilter(dim_x=2, dim_z=1)
        dt = 0.05
        self.kf.F = np.array([[1, dt], [0, 1]])
        self.kf.H = np.array([[1, 0]])
        self.kf.Q = np.array([[0.1, 0], [0, 1.0]])
        self.kf.R = np.array([[50.0]])
        self.kf.x = np.array([[init_dist], [0]])
        self.kf.P *= 100

    def update(self, dist_measured: float) -> Tuple[float, float]:
        self.kf.predict()
        self.kf.update(np.array([[dist_measured]]))
        return float(self.kf.x[0]), float(self.kf.x[1])


# ─────────────────────────────────────────────
# 多目标跟踪器
# ─────────────────────────────────────────────
class MultiTargetTracker:
    def __init__(self):
        self.kalman_filters = {}
        self.last_seen = {}
        self.TIMEOUT = 2.0

    def update(self, detections: List[dict], robot_speed: float) -> List[Target]:
        now = time.time()
        targets = []
        for det in detections:
            tid = det["id"]
            x1, y1, x2, y2 = det["bbox"]
            raw_dist = estimate_distance(x2 - x1)

            if tid not in self.kalman_filters:
                self.kalman_filters[tid] = TargetKalman(raw_dist)

            smooth_dist, abs_vel = self.kalman_filters[tid].update(raw_dist)
            self.last_seen[tid] = now
            rel_vel = abs_vel - robot_speed
            state = self._classify(smooth_dist, rel_vel, abs_vel)
            center = ((x1 + x2) // 2, (y1 + y2) // 2)

            targets.append(Target(
                id=tid, bbox=(x1, y1, x2, y2),
                distance=smooth_dist, abs_velocity=abs_vel,
                rel_velocity=rel_vel, state=state, center=center
            ))

        # 清理超时目标
        for tid in [t for t, s in self.last_seen.items() if now - s > self.TIMEOUT]:
            self.kalman_filters.pop(tid, None)
            self.last_seen.pop(tid, None)

        return targets

    def _classify(self, dist, rel_vel, abs_vel) -> str:
        V = 5.0
        if abs(abs_vel) < V:       return "static"
        elif rel_vel > V:          return "approaching"
        elif abs_vel < 0:          return "away"
        else:                      return "same_dir"


# ─────────────────────────────────────────────
# 运动决策
# ─────────────────────────────────────────────
def make_decision(targets: List[Target]) -> MotionCommand:
    if not targets:
        return MotionCommand(50, 0, "go", "无目标，正常行驶")

    closest = sorted(targets, key=lambda t: t.distance)[0]

    if closest.distance < DANGER_DIST:
        return MotionCommand(0, 0, "stop",
            f"ID:{closest.id} 距离{closest.distance:.0f}cm 紧急停止")

    if closest.state == "approaching" and closest.distance < SLOW_DIST:
        cx = closest.center[0]
        mid = FRAME_WIDTH // 2
        turn = 0.5 if cx < mid - 50 else (-0.5 if cx > mid + 50 else 0)
        direction = "右转避让" if turn > 0 else ("左转避让" if turn < 0 else "减速等待")
        speed = max(10, (closest.distance / SLOW_DIST) * 50)
        return MotionCommand(speed, turn, "slow",
            f"ID:{closest.id} 靠近中 {direction}")

    if closest.state == "static" and closest.distance < SLOW_DIST:
        speed = max(20, (closest.distance / SLOW_DIST) * 60)
        return MotionCommand(speed, 0, "slow",
            f"ID:{closest.id} 静止目标 减速接近")

    if closest.state == "same_dir" and closest.distance < SLOW_DIST:
        return MotionCommand(30, 0, "slow",
            f"ID:{closest.id} 同向移动 保持距离")

    return MotionCommand(50, 0, "go", "正常行驶")


# ─────────────────────────────────────────────
# YOLO检测
# ─────────────────────────────────────────────
class YOLODetector:
    def __init__(self):
        from ultralytics import YOLO
        self.model = YOLO("yolov8n.pt")
        self.next_id = 0
        self.prev_boxes = {}

    def detect(self, frame) -> List[dict]:
        results = self.model(frame, classes=[0], conf=0.4, verbose=False)
        detections = []
        new_boxes = {}
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cx, cy = (x1+x2)//2, (y1+y2)//2
                best_id, best_d = None, 80
                for tid, (px, py) in self.prev_boxes.items():
                    d = ((cx-px)**2 + (cy-py)**2)**0.5
                    if d < best_d:
                        best_d, best_id = d, tid
                if best_id is None:
                    best_id = self.next_id
                    self.next_id += 1
                new_boxes[best_id] = (cx, cy)
                detections.append({"id": best_id, "bbox": (x1,y1,x2,y2)})
        self.prev_boxes = new_boxes
        return detections


# ─────────────────────────────────────────────
# 可视化
# ─────────────────────────────────────────────
STATE_COLORS = {
    "static":      (0, 255, 255),
    "approaching": (0, 0, 255),
    "same_dir":    (0, 255, 0),
    "away":        (128, 128, 128),
}
ACTION_COLORS = {
    "go":   (0, 255, 0),
    "slow": (0, 165, 255),
    "stop": (0, 0, 255),
}

def draw_frame(frame, targets: List[Target], cmd: MotionCommand):
    for t in targets:
        x1, y1, x2, y2 = t.bbox
        color = STATE_COLORS.get(t.state, (255,255,255))
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        label = f"ID:{t.id} {t.distance:.0f}cm {t.rel_velocity:+.0f}cm/s"
        cv2.putText(frame, label, (x1, y1-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        cv2.putText(frame, t.state, (x1, y2+14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    # 决策栏背景
    cv2.rectangle(frame, (0,0), (FRAME_WIDTH, 40), (30,30,30), -1)
    action_color = ACTION_COLORS.get(cmd.action, (255,255,255))
    cv2.putText(frame, f"[{cmd.action.upper()}] spd={cmd.speed:.0f} turn={cmd.turn:+.1f}",
                (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, action_color, 1)
    cv2.putText(frame, cmd.reason,
                (6, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200,200,200), 1)

    # 目标数量
    cv2.putText(frame, f"targets:{len(targets)}",
                (FRAME_WIDTH-90, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)
    return frame


# ─────────────────────────────────────────────
# Flask 串流
# ─────────────────────────────────────────────
app = Flask(__name__)
latest_frame = None
frame_lock = threading.Lock()

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Task3 - Multi Target</title>
    <style>
        body { background:#111; color:#eee; font-family:monospace;
               display:flex; flex-direction:column; align-items:center; margin:0; padding:20px; }
        h2 { color:#0f0; margin-bottom:10px; }
        img { border:2px solid #333; border-radius:4px; width:640px; }
        .legend { display:flex; gap:20px; margin-top:12px; font-size:13px; }
        .dot { width:12px; height:12px; border-radius:50%; display:inline-block; margin-right:5px; }
    </style>
</head>
<body>
    <h2>🤖 Task3 Multi-Target Tracker</h2>
    <img src="/video_feed">
    <div class="legend">
        <span><span class="dot" style="background:#ff0"></span>Static</span>
        <span><span class="dot" style="background:#f00"></span>Approaching</span>
        <span><span class="dot" style="background:#0f0"></span>Same Dir</span>
        <span><span class="dot" style="background:#888"></span>Away</span>
    </div>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    while True:
        with frame_lock:
            if latest_frame is None:
                time.sleep(0.05)
                continue
            frame = latest_frame.copy()
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
               + buf.tobytes() + b'\r\n')
        time.sleep(0.03)


# ─────────────────────────────────────────────
# 主循环（独立线程）
# ─────────────────────────────────────────────
def camera_loop():
    global latest_frame

    print("初始化摄像头...")
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(1)

    print("加载YOLO...")
    detector = YOLODetector()
    tracker  = MultiTargetTracker()

    print("摄像头运行中，访问 http://100.102.90.76:5000")
    fps_time = time.time()
    count = 0

    while True:
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        detections = detector.detect(frame)
        targets    = tracker.update(detections, ROBOT_SPEED)
        cmd        = make_decision(targets)

        frame = draw_frame(frame, targets, cmd)

        with frame_lock:
            latest_frame = frame

        count += 1
        if count % 30 == 0:
            fps = 30 / (time.time() - fps_time)
            fps_time = time.time()
            print(f"FPS:{fps:.1f} | 目标:{len(targets)} | {cmd.action} - {cmd.reason}")

        del frame


# ─────────────────────────────────────────────
# 启动
# ─────────────────────────────────────────────
if __name__ == "__main__":
    t = threading.Thread(target=camera_loop, daemon=True)
    t.start()
    app.run(host="0.0.0.0", port=5000, threaded=True)
