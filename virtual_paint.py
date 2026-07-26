import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
import math
import os
import urllib.request
import threading

# ---------------------------------------------------------
# 1. Threading: Dedicated Video Capture Thread
# ---------------------------------------------------------
class CameraThread(threading.Thread):
    def __init__(self, src=0):
        super().__init__()
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera initialized at {self.w}x{self.h}")
        
        self.frame = None
        self.running = True
        self.lock = threading.Lock()
        
    def run(self):
        while self.running:
            success, frame = self.cap.read()
            if success:
                frame = cv2.flip(frame, 1) 
                with self.lock:
                    self.frame = frame.copy()
            else:
                time.sleep(0.01)
                
    def get_frame(self):
        with self.lock:
            if self.frame is not None:
                return self.frame.copy()
            return None
            
    def stop(self):
        self.running = False
        self.cap.release()

# ---------------------------------------------------------
# 2. Threading: Dedicated MediaPipe Inference Thread
# ---------------------------------------------------------
class InferenceThread(threading.Thread):
    def __init__(self, camera_thread):
        super().__init__()
        self.camera_thread = camera_thread
        self.running = True
        self.lock = threading.Lock()
        self.results = None
        self.timestamp_ms = 0
        
        model_path = 'hand_landmarker.task'
        if not os.path.exists(model_path):
            print("Downloading MediaPipe model...")
            url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
            urllib.request.urlretrieve(url, model_path)
            
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2, # Enabled 2 hands for spatial zoom/pan
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=vision.RunningMode.VIDEO
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def run(self):
        while self.running:
            frame = self.camera_thread.get_frame()
            if frame is not None:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                
                ts = int(time.time() * 1000)
                if ts <= self.timestamp_ms:
                    ts = self.timestamp_ms + 1
                self.timestamp_ms = ts
                
                res = self.detector.detect_for_video(mp_image, ts)
                with self.lock:
                    self.results = res
            else:
                time.sleep(0.01)

    def get_results(self):
        with self.lock:
            return self.results
            
    def stop(self):
        self.running = False

# ---------------------------------------------------------
# 3. Graphical UI Manager (Glassmorphism UI)
# ---------------------------------------------------------
class UIManager:
    def __init__(self, w, h):
        self.w = w
        self.h = h
        self.tools = ["Brush", "Eraser", "Auto-Shape", "Rect", "Circle", "Mask", "Unmask", "Clear"]
        self.active_tool = "Brush"
        self.color = (0, 0, 255) # Red default
        self.thickness = 10
        
        self.tool_boxes = []
        bw, bh = 80, 40
        start_x = 20
        for i, t in enumerate(self.tools):
            self.tool_boxes.append({"name": t, "rect": (start_x + i*(bw+10), 20, start_x + i*(bw+10) + bw, 20 + bh)})
            
        self.grad_x1 = start_x + len(self.tools)*(bw+10) + 20
        self.grad_y1 = 20
        self.grad_x2 = self.w - 100 
        self.grad_y2 = 60
        
        gradient_base = np.arange(256, dtype=np.uint8).reshape(1, 256)
        gradient_color = cv2.applyColorMap(gradient_base, cv2.COLORMAP_HSV)
        self.grad_img = cv2.resize(gradient_color, (self.grad_x2 - self.grad_x1, self.grad_y2 - self.grad_y1))
        
        self.thick_x1 = self.w - 60
        self.thick_y1 = 100
        self.thick_x2 = self.w - 20
        self.thick_y2 = self.h - 100

    def draw(self, frame):
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.w, 80), (20, 20, 20), cv2.FILLED)
        cv2.rectangle(overlay, (self.w - 80, 80), (self.w, self.h), (20, 20, 20), cv2.FILLED)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        for box in self.tool_boxes:
            t = box["name"]
            x1, y1, x2, y2 = box["rect"]
            is_active = self.active_tool == t
            
            if is_active:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 120, 120), cv2.FILLED)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 60, 60), cv2.FILLED)
                
            cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), 1) 
            text_size = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
            tx = x1 + (x2 - x1 - text_size[0]) // 2
            ty = y1 + (y2 - y1 + text_size[1]) // 2
            cv2.putText(frame, t, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            
        frame[self.grad_y1:self.grad_y2, self.grad_x1:self.grad_x2] = self.grad_img
        cv2.rectangle(frame, (self.grad_x1, self.grad_y1), (self.grad_x2, self.grad_y2), (200, 200, 200), 1)
        
        center_x = (self.thick_x1 + self.thick_x2) // 2
        cv2.line(frame, (center_x, self.thick_y1), (center_x, self.thick_y2), (150, 150, 150), 2)
        
        handle_y = int(np.interp(self.thickness, [1, 50], [self.thick_y1, self.thick_y2]))
        cv2.circle(frame, (center_x, handle_y), 12, (255, 255, 255), cv2.FILLED, cv2.LINE_AA)
        cv2.circle(frame, (center_x, handle_y), 12, (100, 100, 100), 2, cv2.LINE_AA)
        
        preview_x, preview_y = self.w - 40, self.h - 40
        cv2.circle(frame, (preview_x, preview_y), self.thickness, self.color, cv2.FILLED, cv2.LINE_AA)
        cv2.circle(frame, (preview_x, preview_y), self.thickness, (255, 255, 255), 2, cv2.LINE_AA)

    def handle_interaction(self, cx, cy):
        for box in self.tool_boxes:
            x1, y1, x2, y2 = box["rect"]
            if x1 < cx < x2 and y1 < cy < y2:
                return "TOOL", box["name"]
                
        if self.grad_x1 < cx < self.grad_x2 and self.grad_y1 < cy < self.grad_y2:
            color = self.grad_img[cy - self.grad_y1, cx - self.grad_x1]
            return "COLOR", (int(color[0]), int(color[1]), int(color[2]))
            
        if self.thick_x1 < cx < self.thick_x2 and self.thick_y1 < cy < self.thick_y2:
            t = int(np.interp(cy, [self.thick_y1, self.thick_y2], [1, 50]))
            return "THICKNESS", max(1, min(50, t))
            
        return None, None

# ---------------------------------------------------------
# 4. Advanced Geometry & Shaders
# ---------------------------------------------------------
def apply_neon_effect(view_layer, frame):
    """Blends a BGR layer onto the main frame with a fast additive neon glow."""
    # Fast check if canvas is completely empty to save time
    if not np.any(view_layer):
        return frame
        
    h, w = view_layer.shape[:2]
    
    # Downscale for ultra-fast blurring (<5ms budget)
    small = cv2.resize(view_layer, (w//4, h//4), interpolation=cv2.INTER_LINEAR)
    blur_small = cv2.GaussianBlur(small, (15, 15), 0)
    glow = cv2.resize(blur_small, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # Pure OpenCV Additive Blending (no floats, extreme speed)
    frame = cv2.add(frame, view_layer)
    frame = cv2.add(frame, glow)
    
    return frame

def recognize_and_snap_shape(points, canvas, color, thickness):
    """Approximates geometric shapes from rough drawn lines and snaps them to vectors."""
    if len(points) < 5:
        return
        
    pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    arc_len = cv2.arcLength(pts, closed=True)
    epsilon = 0.02 * arc_len
    approx = cv2.approxPolyDP(pts, epsilon, closed=True)
    vertices = len(approx)
    bgr_color = (int(color[0]), int(color[1]), int(color[2]))
    
    try:
        if vertices == 3:
            # Triangle
            cv2.drawContours(canvas, [approx], 0, bgr_color, thickness, cv2.LINE_AA)
        elif vertices == 4:
            # Square or Rectangle
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h if h != 0 else 0
            if 0.85 <= aspect_ratio <= 1.15:
                size = (w + h) // 2
                cv2.rectangle(canvas, (x, y), (x + size, y + size), bgr_color, thickness, cv2.LINE_AA)
            else:
                cv2.rectangle(canvas, (x, y), (x + w, y + h), bgr_color, thickness, cv2.LINE_AA)
        else:
            # Circle or Ellipse
            area = cv2.contourArea(pts)
            perimeter = cv2.arcLength(pts, closed=True)
            if perimeter > 0:
                circularity = 4 * np.pi * (area / (perimeter * perimeter))
                if circularity > 0.7 and len(pts) >= 5:
                    ellipse = cv2.fitEllipse(pts)
                    cv2.ellipse(canvas, ellipse, bgr_color, thickness, cv2.LINE_AA)
                else:
                    cv2.drawContours(canvas, [approx], 0, bgr_color, thickness, cv2.LINE_AA)
            else:
                cv2.drawContours(canvas, [approx], 0, bgr_color, thickness, cv2.LINE_AA)
    except Exception as e:
        print(f"Shape fitting error: {e}")
        cv2.drawContours(canvas, [approx], 0, bgr_color, thickness, cv2.LINE_AA)

class VirtualCamera:
    """Manages the global affine transformation for Canvas Zoom & Pan."""
    def __init__(self, screen_w, screen_h, canvas_w, canvas_h):
        self.scale = 1.0
        self.pan_x = (canvas_w - screen_w) / 2.0
        self.pan_y = (canvas_h - screen_h) / 2.0
        
    def get_matrix(self):
        return np.float32([
            [self.scale, 0, -self.pan_x * self.scale],
            [0, self.scale, -self.pan_y * self.scale]
        ])
        
    def screen_to_canvas(self, sx, sy):
        cx = (sx / self.scale) + self.pan_x
        cy = (sy / self.scale) + self.pan_y
        return int(cx), int(cy)
        
    def canvas_to_screen(self, cx, cy):
        sx = (cx - self.pan_x) * self.scale
        sy = (cy - self.pan_y) * self.scale
        return int(sx), int(sy)

# ---------------------------------------------------------
# 5. Utilities (Filters, Skeletons, State)
# ---------------------------------------------------------
class EmaFilter:
    def __init__(self, alpha=0.5): 
        self.alpha = alpha
        self.x = None
        self.y = None
    
    def update(self, x, y):
        if self.x is None or self.y is None:
            self.x, self.y = x, y
        else:
            self.x = self.alpha * x + (1 - self.alpha) * self.x
            self.y = self.alpha * y + (1 - self.alpha) * self.y
        return int(self.x), int(self.y)

    def reset(self):
        self.x = None
        self.y = None

class SpatialGestureState:
    def __init__(self):
        self.mode = None # "ZOOM" or "PAN"
        self.initial_D = 0
        self.initial_scale = 1.0
        self.initial_pan = (0, 0)
        self.initial_midpoint = (0, 0)
        self.zoom_center_canvas = (0, 0)
        self.zoom_center_screen = (0, 0)
        
    def reset(self):
        self.mode = None

def get_finger_states(lm_list):
    fingers = []
    fingers.append(1 if lm_list[4][0] < lm_list[3][0] else 0)
    tips = [8, 12, 16, 20]
    for id in tips:
        fingers.append(1 if lm_list[id][1] < lm_list[id - 2][1] else 0)
    return fingers

def draw_skeleton(frame, lm_list):
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
        (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
    ]
    for connection in HAND_CONNECTIONS:
        cv2.line(frame, lm_list[connection[0]], lm_list[connection[1]], (200, 200, 200), 1, cv2.LINE_AA)
    for pt in lm_list:
        cv2.circle(frame, pt, 2, (255, 255, 255), cv2.FILLED, cv2.LINE_AA)

# ---------------------------------------------------------
# 6. Main Application Loop
# ---------------------------------------------------------
def main():
    cam_thread = CameraThread()
    cam_thread.start()
    
    print("Warming up camera...")
    while cam_thread.get_frame() is None:
        time.sleep(0.1)
        
    inf_thread = InferenceThread(cam_thread)
    inf_thread.start()
    
    w, h = cam_thread.w, cam_thread.h
    ui = UIManager(w, h)
    
    # 4K Infinite Canvas (BGR Additive)
    VIRTUAL_W, VIRTUAL_H = 3840, 2160
    canvas = np.zeros((VIRTUAL_H, VIRTUAL_W, 3), dtype=np.uint8)
    space_mask = np.ones((VIRTUAL_H, VIRTUAL_W), np.uint8) * 255 
    
    camera = VirtualCamera(w, h, VIRTUAL_W, VIRTUAL_H)
    cursor_filter = EmaFilter(alpha=0.6)
    spatial_state = SpatialGestureState()
    
    px, py = 0, 0
    drawing_shape = False
    shape_start_pt = None
    current_stroke = []
    
    cv2.namedWindow("Virtual AR Canvas", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Virtual AR Canvas", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    while True:
        frame = cam_thread.get_frame()
        if frame is None: continue
            
        results = inf_thread.get_results()
        
        if results and results.hand_landmarks:
            num_hands = len(results.hand_landmarks)
            hands_data = []
            
            for i in range(num_hands):
                lm_list = [(int(lm.x * w), int(lm.y * h)) for lm in results.hand_landmarks[i]]
                fingers = get_finger_states(lm_list)
                t_x, t_y = lm_list[4]
                i_x, i_y = lm_list[8]
                is_pinching = math.hypot(i_x - t_x, i_y - t_y) < 60
                is_open = sum(fingers) == 5
                
                hands_data.append({
                    "pinching": is_pinching,
                    "open": is_open,
                    "idx": (i_x, i_y),
                    "palm": (lm_list[0][0], lm_list[0][1])
                })
                draw_skeleton(frame, lm_list)
                
            # -- Spatial Gestures (2 Hands) --
            if num_hands == 2:
                h1, h2 = hands_data[0], hands_data[1]
                
                if h1["pinching"] and h2["pinching"]:
                    dist = math.hypot(h1["idx"][0] - h2["idx"][0], h1["idx"][1] - h2["idx"][1])
                    if spatial_state.mode != "ZOOM":
                        spatial_state.mode = "ZOOM"
                        spatial_state.initial_D = dist
                        spatial_state.initial_scale = camera.scale
                        cx = (h1["idx"][0] + h2["idx"][0]) / 2.0
                        cy = (h1["idx"][1] + h2["idx"][1]) / 2.0
                        spatial_state.zoom_center_screen = (cx, cy)
                        spatial_state.zoom_center_canvas = camera.screen_to_canvas(cx, cy)
                    else:
                        if spatial_state.initial_D > 10:
                            new_scale = spatial_state.initial_scale * (dist / spatial_state.initial_D)
                            camera.scale = max(0.2, min(5.0, new_scale))
                            zx, zy = spatial_state.zoom_center_canvas
                            sx, sy = spatial_state.zoom_center_screen
                            camera.pan_x = zx - (sx / camera.scale)
                            camera.pan_y = zy - (sy / camera.scale)
                            
                elif h1["open"] and h2["open"]:
                    mid_x = (h1["palm"][0] + h2["palm"][0]) / 2.0
                    mid_y = (h1["palm"][1] + h2["palm"][1]) / 2.0
                    if spatial_state.mode != "PAN":
                        spatial_state.mode = "PAN"
                        spatial_state.initial_midpoint = (mid_x, mid_y)
                        spatial_state.initial_pan = (camera.pan_x, camera.pan_y)
                    else:
                        dx = mid_x - spatial_state.initial_midpoint[0]
                        dy = mid_y - spatial_state.initial_midpoint[1]
                        camera.pan_x = spatial_state.initial_pan[0] - (dx / camera.scale)
                        camera.pan_y = spatial_state.initial_pan[1] - (dy / camera.scale)
                else:
                    spatial_state.reset()
            else:
                spatial_state.reset()
                
            # -- Standard Drawing (Hand 0) --
            if spatial_state.mode is None and num_hands > 0:
                h1 = hands_data[0]
                raw_x, raw_y = h1["idx"]
                cx, cy = cursor_filter.update(raw_x, raw_y)
                
                cv2.circle(frame, (cx, cy), 8, ui.color, cv2.FILLED, cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), 8, (255, 255, 255), 2, cv2.LINE_AA)
                
                if h1["pinching"]:
                    action_type, action_val = ui.handle_interaction(cx, cy)
                    
                    if action_type == "TOOL":
                        if action_val == "Clear":
                            canvas.fill(0)
                        elif action_val == "Unmask":
                            space_mask.fill(255)
                        else:
                            ui.active_tool = action_val
                    elif action_type == "COLOR":
                        ui.color = action_val
                    elif action_type == "THICKNESS":
                        ui.thickness = action_val
                    else:
                        canvas_cx, canvas_cy = camera.screen_to_canvas(cx, cy)
                        draw_color = (0, 0, 0) if ui.active_tool == "Eraser" else ui.color
                        
                        if ui.active_tool in ["Brush", "Eraser"]:
                            if px == 0 and py == 0:
                                px, py = canvas_cx, canvas_cy
                                
                            x_min = max(0, min(px, canvas_cx) - ui.thickness)
                            x_max = min(VIRTUAL_W, max(px, canvas_cx) + ui.thickness)
                            y_min = max(0, min(py, canvas_cy) - ui.thickness)
                            y_max = min(VIRTUAL_H, max(py, canvas_cy) + ui.thickness)

                            if x_min < x_max and y_min < y_max:
                                sub_canvas = canvas[y_min:y_max, x_min:x_max]
                                sub_temp = sub_canvas.copy()
                                sub_mask = space_mask[y_min:y_max, x_min:x_max]
                                
                                local_px, local_py = px - x_min, py - y_min
                                local_cx, local_cy = canvas_cx - x_min, canvas_cy - y_min
                                
                                cv2.line(sub_temp, (local_px, local_py), (local_cx, local_cy), draw_color, ui.thickness, cv2.LINE_AA)
                                cv2.circle(sub_temp, (local_cx, local_cy), ui.thickness//2, draw_color, cv2.FILLED, cv2.LINE_AA)
                                
                                mask_2d = sub_mask == 255
                                sub_canvas[mask_2d] = sub_temp[mask_2d]
                                canvas[y_min:y_max, x_min:x_max] = sub_canvas
                                
                            px, py = canvas_cx, canvas_cy
                            
                        elif ui.active_tool in ["Rect", "Circle", "Mask", "Auto-Shape"]:
                            if not drawing_shape:
                                drawing_shape = True
                                shape_start_pt = (cx, cy)
                                current_stroke = []
                                
                            current_stroke.append((canvas_cx, canvas_cy))
                            
                            # Live Previews rendered to screen frame
                            if ui.active_tool == "Rect":
                                cv2.rectangle(frame, shape_start_pt, (cx, cy), ui.color, ui.thickness, cv2.LINE_AA)
                            elif ui.active_tool == "Circle":
                                radius = int(math.hypot(cx - shape_start_pt[0], cy - shape_start_pt[1]))
                                cv2.circle(frame, shape_start_pt, radius, ui.color, ui.thickness, cv2.LINE_AA)
                            elif ui.active_tool == "Mask":
                                cv2.rectangle(frame, shape_start_pt, (cx, cy), (255, 0, 255), 2, cv2.LINE_AA)
                            elif ui.active_tool == "Auto-Shape":
                                screen_stroke = [camera.canvas_to_screen(pt[0], pt[1]) for pt in current_stroke]
                                for j in range(1, len(screen_stroke)):
                                    cv2.line(frame, screen_stroke[j-1], screen_stroke[j], ui.color, ui.thickness, cv2.LINE_AA)
                else:
                    px, py = 0, 0
                    if drawing_shape:
                        canvas_shape_start = camera.screen_to_canvas(*shape_start_pt)
                        canvas_shape_end = camera.screen_to_canvas(cx, cy)
                        
                        if ui.active_tool in ["Rect", "Circle", "Auto-Shape"]:
                            temp = np.zeros_like(canvas)
                            if ui.active_tool == "Rect":
                                cv2.rectangle(temp, canvas_shape_start, canvas_shape_end, ui.color, ui.thickness, cv2.LINE_AA)
                            elif ui.active_tool == "Circle":
                                radius = int(math.hypot(canvas_shape_end[0] - canvas_shape_start[0], canvas_shape_end[1] - canvas_shape_start[1]))
                                cv2.circle(temp, canvas_shape_start, radius, ui.color, ui.thickness, cv2.LINE_AA)
                            elif ui.active_tool == "Auto-Shape":
                                recognize_and_snap_shape(current_stroke, temp, ui.color, ui.thickness)
                                
                            shape_mask_2d = (temp.sum(axis=2) > 0)
                            combined_mask = (space_mask == 255) & shape_mask_2d
                            canvas[combined_mask] = temp[combined_mask]
                            
                        elif ui.active_tool == "Mask":
                            space_mask = np.zeros((VIRTUAL_H, VIRTUAL_W), np.uint8)
                            r_x1 = max(0, min(canvas_shape_start[0], canvas_shape_end[0]))
                            r_y1 = max(0, min(canvas_shape_start[1], canvas_shape_end[1]))
                            r_x2 = min(VIRTUAL_W, max(canvas_shape_start[0], canvas_shape_end[0]))
                            r_y2 = min(VIRTUAL_H, max(canvas_shape_start[1], canvas_shape_end[1]))
                            cv2.rectangle(space_mask, (r_x1, r_y1), (r_x2, r_y2), 255, cv2.FILLED)
                        
                        drawing_shape = False
                        current_stroke = []
        else:
            px, py = 0, 0
            if drawing_shape: drawing_shape = False
            cursor_filter.reset()
            spatial_state.reset()

        # ---------------------------------------------------------
        # 7. Render Pipeline (Virtual Camera -> Neon -> UI)
        # ---------------------------------------------------------
        M = camera.get_matrix()
        view_canvas = cv2.warpAffine(canvas, M, (w, h), flags=cv2.INTER_LINEAR)
        view_mask = cv2.warpAffine(space_mask, M, (w, h), flags=cv2.INTER_NEAREST)
        
        frame = apply_neon_effect(view_canvas, frame)
        
        if np.mean(space_mask) < 255:
            contours, _ = cv2.findContours(view_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame, contours, -1, (0, 255, 255), 2, cv2.LINE_AA)

        ui.draw(frame)
        cv2.imshow("Virtual AR Canvas", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27: break

    cam_thread.stop()
    inf_thread.stop()
    cam_thread.join()
    inf_thread.join()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
