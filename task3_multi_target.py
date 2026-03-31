"""
Task3：Multi-target-detection + Autonomous speed control
硬件：Raspberry pi4B + Pi Camera (picamera2) 
Installation：
    pip install ultralytics opencv-python numpy --break-system-packages
    pip install filterpy --break-system-packages  
    # SORT: git clone https://github.com/abewley/sort
"""

import cv2
import numpy as np
from picamera2 import Picamera2
from filterpy.kalman import KalmanFilter
from dataclasses import dataclass
from typing import List, Tuple, Optional
import time

# ─────────────────────────────────────────────
# 1.Set up
# ─────────────────────────────────────────────
FRAME_WIDTH   = 640
FRAME_HEIGHT  = 480
FOCAL_LENGTH  = 600.0    # CHANGE ACCORDINGLY WITH YOUR CAMERA
REAL_WIDTH    = 45.0     # Human width (cm)
DANGER_DIST   = 80.0     # Threshold dangerous distance between robot and human(cm)
SLOW_DIST     = 150.0    # Threshold slowing distance(cm)
ROBOT_SPEED   = 30.0     # Robot's speed(cm/s)



@dataclass
class Target:
    id: int                    # tracking ID
    bbox: Tuple                # (x1, y1, x2, y2)
    distance: float            # distance(cm)
    abs_velocity: float        # target speed
    rel_velocity: float        # relative speed = abs_velocity - robot_speed
    state: str                 # "static" / "approaching" / "same_dir" / "away"
    center: Tuple              # center 


# ─────────────────────────────────────────────
# 2. Distance calculation
# ─────────────────────────────────────────────
def estimate_distance(bbox_width_px: float) -> float:
    
    if bbox_width_px < 1:
        return 9999.0
    return (REAL_WIDTH * FOCAL_LENGTH) / bbox_width_px


# ─────────────────────────────────────────────
# 3. Single target Kalman filter
# ─────────────────────────────────────────────
class TargetKalman:
    
    def __init__(self, init_dist: float):
        self.kf = KalmanFilter(dim_x=2, dim_z=1)
        dt = 0.05  # 假设20Hz采样

        # 状态转移矩阵 x = [dist, vel]
        self.kf.F = np.array([[1, dt],
                               [0, 1]])
        # 观测矩阵（只观测距离）
        self.kf.H = np.array([[1, 0]])
        # 过程噪声
        self.kf.Q = np.array([[0.1, 0],
                               [0,   1.0]])
        # 观测噪声（单目测距误差较大，调大R）
        self.kf.R = np.array([[50.0]])
        # 初始状态
        self.kf.x = np.array([[init_dist], [0]])
        self.kf.P *= 100

    def update(self, dist_measured: float) -> Tuple[float, float]:
        """返回 (平滑距离, 估计速度)"""
        self.kf.predict()
        self.kf.update(np.array([[dist_measured]]))
        dist = float(self.kf.x[0])
        vel  = float(self.kf.x[1])
        return dist, vel


# ─────────────────────────────────────────────
# 4. MultiTargetTracker
# ─────────────────────────────────────────────
class MultiTargetTracker:
    def __init__(self):
        self.kalman_filters = {}   # {track_id: TargetKalman}
        self.last_seen = {}        # {track_id: timestamp}
        self.TIMEOUT = 2.0         # 超时移除(s)

    def update(self, detections: List[dict], robot_speed: float) -> List[Target]:
        """
        detections: YOLO输出的检测列表
        [{"id": int, "bbox": (x1,y1,x2,y2), "class": str}, ...]
        """
        now = time.time()
        targets = []

        for det in detections:
            tid = det["id"]
            x1, y1, x2, y2 = det["bbox"]
            bbox_w = x2 - x1

            # 原始距离估算
            raw_dist = estimate_distance(bbox_w)

            # 初始化或更新卡尔曼
            if tid not in self.kalman_filters:
                self.kalman_filters[tid] = TargetKalman(raw_dist)

            smooth_dist, abs_vel = self.kalman_filters[tid].update(raw_dist)
            self.last_seen[tid] = now

            # 相对速度 = 目标绝对速度 - 机器人速度
            # 正值=目标在靠近，负值=目标在远离
            rel_vel = abs_vel - robot_speed

            # 目标状态分类
            state = self._classify_state(smooth_dist, rel_vel, abs_vel)

            center = ((x1 + x2) // 2, (y1 + y2) // 2)

            targets.append(Target(
                id=tid,
                bbox=(x1, y1, x2, y2),
                distance=smooth_dist,
                abs_velocity=abs_vel,
                rel_velocity=rel_vel,
                state=state,
                center=center,
            ))

        # 清理超时目标
        expired = [tid for tid, t in self.last_seen.items()
                   if now - t > self.TIMEOUT]
        for tid in expired:
            del self.kalman_filters[tid]
            del self.last_seen[tid]

        return targets

    def _classify_state(self, dist: float, rel_vel: float,
                         abs_vel: float) -> str:
        """
        区分三种状态（对应任务5，这里先定义好接口）：
        static     : 静止目标（机器人在靠近）
        approaching: 朝机器人移动
        same_dir   : 同向移动
        away       : 远离
        """
        VEL_THRESHOLD = 5.0  # cm/s，低于此认为静止

        if abs(abs_vel) < VEL_THRESHOLD:
            return "static"           # 目标本身不动
        elif rel_vel > VEL_THRESHOLD:
            return "approaching"      # 目标在靠近机器人
        elif rel_vel < -VEL_THRESHOLD:
            if abs_vel < 0:
                return "away"         # 目标在远离
            else:
                return "same_dir"     # 同向移动
        else:
            return "static"


# ─────────────────────────────────────────────
# 5. Robot motion command
# ─────────────────────────────────────────────
@dataclass
class MotionCommand:
    speed: float        # 目标速度 0~100
    turn: float         # 转向 -1(左)~1(右)
    action: str         # "go" / "slow" / "stop" / "turn"
    reason: str         # 决策原因（调试用）


def make_motion_decision(targets: List[Target]) -> MotionCommand:
    """
    根据所有目标状态做出运动决策
    优先级：紧急停止 > 减速 > 转向 > 正常行驶
    """
    if not targets:
        return MotionCommand(speed=50, turn=0, action="go",
                             reason="无目标，正常行驶")

    # 按距离排序，最近的优先处理
    targets_sorted = sorted(targets, key=lambda t: t.distance)
    closest = targets_sorted[0]

    # 紧急停止
    if closest.distance < DANGER_DIST:
        return MotionCommand(speed=0, turn=0, action="stop",
                             reason=f"目标{closest.id}距离{closest.distance:.0f}cm，紧急停止")

    # 有目标正在靠近
    if closest.state == "approaching":
        if closest.distance < SLOW_DIST:
            # 判断目标偏左还是偏右，考虑转向避让
            frame_center_x = FRAME_WIDTH // 2
            target_x = closest.center[0]
            if target_x < frame_center_x - 50:
                turn = 0.5   # 目标偏左，机器人右转避让
                reason = f"目标{closest.id}靠近且偏左，右转避让"
            elif target_x > frame_center_x + 50:
                turn = -0.5  # 目标偏右，机器人左转避让
                reason = f"目标{closest.id}靠近且偏右，左转避让"
            else:
                turn = 0
                reason = f"目标{closest.id}正面靠近，减速等待"
            speed = max(10, (closest.distance / SLOW_DIST) * 50)
            return MotionCommand(speed=speed, turn=turn,
                                 action="slow", reason=reason)

    # 静止目标在前方
    if closest.state == "static" and closest.distance < SLOW_DIST:
        speed = max(20, (closest.distance / SLOW_DIST) * 60)
        return MotionCommand(speed=speed, turn=0, action="slow",
                             reason=f"目标{closest.id}静止，减速接近")

    # 同向目标，保持跟随距离
    if closest.state == "same_dir":
        if closest.distance < SLOW_DIST:
            return MotionCommand(speed=30, turn=0, action="slow",
                                 reason=f"目标{closest.id}同向，保持距离")

    # 正常行驶
    return MotionCommand(speed=50, turn=0, action="go",
                         reason="正常行驶")


# ─────────────────────────────────────────────
# 6. Target moving status
# ─────────────────────────────────────────────
STATE_COLORS = {
    "static":      (0, 255, 255),   # yellow -> static
    "approaching": (0, 0, 255),     # red -> approaching
    "same_dir":    (0, 255, 0),     # green -> moving in same direction
    "away":        (128, 128, 128), # grey -> moving away
}

def draw_targets(frame: np.ndarray, targets: List[Target],
                 cmd: MotionCommand) -> np.ndarray:
    for t in targets:
        x1, y1, x2, y2 = t.bbox
        color = STATE_COLORS.get(t.state, (255, 255, 255))

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = (f"ID:{t.id} {t.distance:.0f}cm "
                 f"rv:{t.rel_velocity:+.0f}cm/s [{t.state}]")
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    # 决策信息
    cv2.putText(frame,
                f"CMD: {cmd.action} spd={cmd.speed:.0f} turn={cmd.turn:+.1f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(frame, cmd.reason,
                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    return frame


# ─────────────────────────────────────────────
# 7. YOLO检测封装（带简易ID分配）
# ─────────────────────────────────────────────
class YOLODetector:
    def __init__(self, model_path: str = "yolov8n.pt"):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.next_id = 0
        # 简易IoU跟踪（正式版换SORT）
        self.prev_boxes = {}

    def detect(self, frame: np.ndarray) -> List[dict]:
        results = self.model(frame, classes=[0],  # 只检测人
                             conf=0.4, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                # 简易ID：用中心点最近的上一帧目标匹配
                tid = self._assign_id(x1, y1, x2, y2)
                detections.append({
                    "id": tid,
                    "bbox": (x1, y1, x2, y2),
                    "class": "person",
                    "conf": float(box.conf[0]),
                })
        return detections

    def _assign_id(self, x1, y1, x2, y2) -> int:
        """简易中心点匹配，正式版请换SORT"""
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        best_id, best_dist = None, 80  # 80px阈值
        for tid, (px, py) in self.prev_boxes.items():
            d = ((cx - px)**2 + (cy - py)**2) ** 0.5
            if d < best_dist:
                best_dist = d
                best_id = tid
        if best_id is None:
            best_id = self.next_id
            self.next_id += 1
        self.prev_boxes[best_id] = (cx, cy)
        return best_id


# ─────────────────────────────────────────────
# 8. Arduino motion command
# ─────────────────────────────────────────────
class ArduinoComm:
    def __init__(self, port: str = "/dev/ttyUSB0", baud: int = 9600):
        try:
            import serial
            self.ser = serial.Serial(port, baud, timeout=1)
            self.available = True
            print(f"[Arduino] 连接成功: {port}")
        except Exception as e:
            print(f"[Arduino] 未连接，仅打印指令: {e}")
            self.available = False

    def send(self, cmd: MotionCommand):
        msg = f"S{int(cmd.speed):03d}T{int(cmd.turn*100):+04d}\n"
        if self.available:
            self.ser.write(msg.encode())
        else:
            print(f"[模拟指令] {msg.strip()} | {cmd.reason}")


# ─────────────────────────────────────────────
# 9. Main execution
# ─────────────────────────────────────────────
def main():
    print("初始化摄像头...")
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(1)

    print("加载YOLO模型...")
    detector = YOLODetector("yolov8n.pt")

    tracker  = MultiTargetTracker()
    arduino  = ArduinoComm()

    print("开始运行，按 'q' 退出")
    fps_time = time.time()
    frame_count = 0

    while True:
        # 抓帧
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # 检测
        detections = detector.detect(frame)

        # 跟踪 + 卡尔曼
        targets = tracker.update(detections, robot_speed=ROBOT_SPEED)

        # 决策
        cmd = make_motion_decision(targets)

        # 发送控制指令
        arduino.send(cmd)

        # 可视化
        frame = draw_targets(frame, targets, cmd)

        # FPS显示
        frame_count += 1
        if frame_count % 30 == 0:
            fps = 30 / (time.time() - fps_time)
            fps_time = time.time()
            print(f"FPS: {fps:.1f} | 目标数: {len(targets)}")

        cv2.imshow("Task3 - Multi Target", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    picam2.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
