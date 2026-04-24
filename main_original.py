import cv2
import numpy as np
import threading
import subprocess
import os
import time
import math

# Sound effects using system commands (works on macOS, Linux, Windows)
SOUND_ENABLED = True

# MediaPipe import - handle both old and new API
import mediapipe as mp

# Check if old API is available
try:
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    USE_OLD_API = True
    print("✅ Using MediaPipe old API")
except AttributeError:
    # MediaPipe 0.10+ - use new API with compatibility layer
    USE_OLD_API = False
    print("⚠️ MediaPipe 0.10+ detected. Using new API with compatibility layer...")
    
    # Import new API
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
    from mediapipe.tasks.python.core import base_options as base_options_module
    from mediapipe import ImageFormat
    
    class HandsCompat:
        def __init__(self, static_image_mode=False, max_num_hands=1, 
                     min_detection_confidence=0.7, min_tracking_confidence=0.7):
            model_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            model_file = "hand_landmarker.task"
            
            if not os.path.exists(model_file):
                print("📥 Downloading MediaPipe hand landmarker model...")
                try:
                    import urllib.request
                    import ssl
                    ssl_context = ssl.create_default_context()
                    ssl_context.check_hostname = False
                    ssl_context.verify_mode = ssl.CERT_NONE
                    with urllib.request.urlopen(model_url, context=ssl_context) as response:
                        with open(model_file, 'wb') as out_file:
                            out_file.write(response.read())
                    print("✅ Model downloaded successfully!")
                except Exception as e:
                    print(f"❌ Failed to download model: {e}")
                    raise Exception("Model file required.")
            
            base_options = base_options_module.BaseOptions(model_asset_path=model_file)
            options = HandLandmarkerOptions(
                base_options=base_options,
                num_hands=max_num_hands,
                min_hand_detection_confidence=min_detection_confidence,
                min_hand_presence_confidence=min_tracking_confidence
            )
            self.detector = HandLandmarker.create_from_options(options)
        
        def process(self, image):
            mp_image = mp.Image(image_format=ImageFormat.SRGB, data=image)
            result = self.detector.detect(mp_image)
            
            class OldResult:
                def __init__(self, landmarks_list, handedness_list):
                    self.multi_hand_landmarks = landmarks_list
                    self.multi_handedness = handedness_list
            
            landmarks_list = []
            handedness_list = []
            
            if result.hand_landmarks:
                for hand_landmarks in result.hand_landmarks:
                    class HandLandmarks:
                        def __init__(self, landmarks):
                            self.landmark = landmarks
                    
                    landmark_list = []
                    for lm in hand_landmarks:
                        class Landmark:
                            def __init__(self, x, y, z):
                                self.x = x
                                self.y = y
                                self.z = z
                        landmark_list.append(Landmark(lm.x, lm.y, lm.z if hasattr(lm, 'z') else 0))
                    
                    landmarks_list.append(HandLandmarks(landmark_list))
            
            if result.handedness:
                for hand_info in result.handedness:
                    class Classification:
                        def __init__(self, label):
                            self.label = label
                    class Handedness:
                        def __init__(self, classification):
                            self.classification = classification
                    category_name = hand_info[0].category_name if hand_info else "Unknown"
                    handedness_list.append(Handedness([Classification(category_name)]))
            
            return OldResult(landmarks_list, handedness_list)
    
    class DrawingUtilsCompat:
        @staticmethod
        def draw_landmarks(image, landmark_list, connections, 
                       landmark_drawing_spec=None, connection_drawing_spec=None):
            if landmark_list is None:
                return
            
            h, w = image.shape[:2]
            
            if connections and connection_drawing_spec:
                color = connection_drawing_spec.color if hasattr(connection_drawing_spec, 'color') else (0, 255, 0)
                thickness = connection_drawing_spec.thickness if hasattr(connection_drawing_spec, 'thickness') else 2
                
                for connection in connections:
                    start_idx = connection[0]
                    end_idx = connection[1]
                    if start_idx < len(landmark_list.landmark) and end_idx < len(landmark_list.landmark):
                        start = landmark_list.landmark[start_idx]
                        end = landmark_list.landmark[end_idx]
                        cv2.line(image, 
                                (int(start.x * w), int(start.y * h)),
                                (int(end.x * w), int(end.y * h)),
                                color, thickness)
            
            if landmark_drawing_spec:
                color = landmark_drawing_spec.color if hasattr(landmark_drawing_spec, 'color') else (0, 0, 255)
                radius = landmark_drawing_spec.circle_radius if hasattr(landmark_drawing_spec, 'circle_radius') else 5
                
                for landmark in landmark_list.landmark:
                    cv2.circle(image, 
                             (int(landmark.x * w), int(landmark.y * h)),
                             radius, color, -1)
    
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
        (5, 9), (9, 13), (13, 17)
    ]
    
    class MPHandsCompat:
        Hands = HandsCompat
        HAND_CONNECTIONS = HAND_CONNECTIONS
    
    mp_hands = MPHandsCompat()
    mp_drawing = DrawingUtilsCompat()

# Load hand template image
template = cv2.imread('hand_template.png', cv2.IMREAD_UNCHANGED)
if template is None:
    print("Warning: hand_template.png not found, creating default template...")
    template = np.zeros((250, 250, 4), dtype=np.uint8)
    cv2.circle(template, (125, 125), 100, (0, 255, 0, 180), 3)
    cv2.circle(template, (125, 140), 60, (0, 255, 0, 120), 2)
    cv2.line(template, (125, 80), (125, 40), (0, 255, 0, 200), 8)
    cv2.line(template, (105, 85), (95, 45), (0, 255, 0, 200), 6)
    cv2.line(template, (145, 85), (155, 45), (0, 255, 0, 200), 6)
    cv2.line(template, (85, 120), (60, 100), (0, 255, 0, 200), 6)
    cv2.line(template, (165, 90), (175, 50), (0, 255, 0, 200), 5)
    cv2.putText(template, "ALIGN", (85, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0, 255), 2)

template_h, template_w = template.shape[:2]

# Window size
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

def find_best_camera():
    """Find the best available camera"""
    print("\n📹 Scanning for available cameras...")
    available_cameras = []
    
    for i in range(10):
        cap_test = cv2.VideoCapture(i)
        if cap_test.isOpened():
            available_cameras.append((i, f"Camera {i}"))
            print(f"  ✅ Found: Camera {i}")
            cap_test.release()
    
    if not available_cameras:
        print("❌ No cameras found! Using default camera 0")
        return 0
    
    if len(available_cameras) > 1:
        chosen_idx = available_cameras[-1][0]
        print(f"✅ Using camera: Camera {chosen_idx}")
        return chosen_idx
    
    print(f"✅ Using camera: {available_cameras[0][1]}")
    return available_cameras[0][0]

camera_idx = find_best_camera()
cap = cv2.VideoCapture(camera_idx)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WINDOW_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if USE_OLD_API:
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
else:
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

# Global variables
video_playing = False
video_thread = None
verification_progress = 0.0
verifying = False
verification_start_time = 0
verification_duration = 1.5
correct_hand_detected = False
wrong_hand_attempt = False
wrong_hand_message_time = 0
last_wrong_hand_sound_time = 0
wrong_hand_sound_cooldown = 1.0
last_valid_hand_center = None

# ACCEPTED_HAND = "Left" means we ACCEPT the Left hand
ACCEPTED_HAND = "Left"

def play_sound_effect(sound_type):
    """Play sound effects for different events"""
    if not SOUND_ENABLED:
        return
    
    def play_beep(frequency, duration):
        try:
            import platform
            system = platform.system()
            
            if system == "Darwin":
                try:
                    os.system(f'sox -n -d synth {duration} sine {frequency} 2>/dev/null &')
                except:
                    try:
                        if frequency < 300:
                            os.system('afplay /System/Library/Sounds/Basso.aiff 2>/dev/null &')
                        elif frequency < 500:
                            os.system('afplay /System/Library/Sounds/Glass.aiff 2>/dev/null &')
                        else:
                            os.system('afplay /System/Library/Sounds/Ping.aiff 2>/dev/null &')
                    except:
                        os.system('printf "\a"')
            elif system == "Linux":
                os.system(f'beep -f {frequency} -l {int(duration*1000)} 2>/dev/null || speaker-test -t sine -f {frequency} -l {int(duration*1000)} >/dev/null 2>&1 &')
            else:
                try:
                    import winsound
                    winsound.Beep(frequency, int(duration * 1000))
                except:
                    os.system('echo \a')
        except:
            try:
                os.system('printf "\a"')
            except:
                pass
    
    def play_sound():
        if sound_type == "verification_start":
            play_beep(400, 0.2)
        elif sound_type == "verification_complete":
            play_beep(600, 0.15)
            time.sleep(0.1)
            play_beep(800, 0.15)
        elif sound_type == "wrong_hand":
            play_beep(200, 0.4)
            time.sleep(0.1)
            play_beep(200, 0.3)
        elif sound_type == "video_start":
            play_beep(800, 0.2)
    
    sound_thread = threading.Thread(target=play_sound)
    sound_thread.daemon = True
    sound_thread.start()

def check_video_file():
    """Check if video/login.mp4 exists"""
    video_path = 'video/login.mp4'
    if not os.path.exists(video_path):
        print(f"Warning: {video_path} not found!")
        return False
    return True

def play_video_with_audio():
    """Play video"""
    global video_playing
    
    video_path = 'video/login.mp4'
    print(f"🎬 Opening video: {video_path}")
    
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"❌ Error: Could not open {video_path}")
        video_playing = False
        return

    try:
        print(f"✅ Video opened successfully! Starting playback...")
        play_sound_effect("video_start")

        fps = video.get(cv2.CAP_PROP_FPS)
        frame_delay = 1.0 / fps if fps > 0 else 1.0 / 30
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        current_frame = 0

        print(f"🎬 Playing video: {total_frames} frames at {fps} FPS")
        start_time = time.time()

        while video.isOpened() and video_playing:
            ret_vid, frame_vid = video.read()
            if not ret_vid:
                print("📹 Video ended")
                break

            current_frame += 1
            frame_vid = cv2.resize(frame_vid, (WINDOW_WIDTH, WINDOW_HEIGHT))

            overlay_height = 80
            overlay = np.zeros((overlay_height, WINDOW_WIDTH, 3), dtype=np.uint8)

            progress = current_frame / total_frames
            bar_width = int(WINDOW_WIDTH * 0.6)
            bar_x = (WINDOW_WIDTH - bar_width) // 2
            cv2.rectangle(overlay, (bar_x, 45), (bar_x + bar_width, 60), (50, 50, 50), -1)
            cv2.rectangle(overlay, (bar_x, 45), (bar_x + int(bar_width * progress), 60), (0, 255, 0), -1)

            cv2.putText(overlay, "🎬 PLAYING VIDEO", (30, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(overlay, "Press ESC to return", (WINDOW_WIDTH - 200, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            alpha = 0.8
            frame_vid[0:overlay_height] = cv2.addWeighted(
                frame_vid[0:overlay_height], alpha, overlay, 1-alpha, 0
            )

            cv2.imshow('Hand Scanner', frame_vid)

            elapsed = time.time() - start_time
            expected_time = current_frame * frame_delay
            if elapsed < expected_time:
                time.sleep(expected_time - elapsed)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                video_playing = False
                break
    except Exception as e:
        print(f"❌ Video thread error: {e}")
        video_playing = False
    finally:
        video.release()
        video_playing = False
        print("🛑 Video playback stopped")

def overlay_image_alpha(img, img_overlay, pos):
    """Overlay image with alpha channel"""
    x, y = pos
    h, w = img.shape[:2]
    overlay_h, overlay_w = img_overlay.shape[:2]
    
    if x < 0 or y < 0 or x + overlay_w > w or y + overlay_h > h:
        return
    
    if img_overlay.shape[2] == 4:
        alpha_overlay = img_overlay[:, :, 3] / 255.0
        alpha_background = 1.0 - alpha_overlay
        
        for c in range(0, 3):
            img[y:y+overlay_h, x:x+overlay_w, c] = (
                alpha_overlay * img_overlay[:, :, c] +
                alpha_background * img[y:y+overlay_h, x:x+overlay_w, c]
            )
    else:
        img[y:y+overlay_h, x:x+overlay_w] = img_overlay

def get_hand_center(hand_landmarks, w, h):
    """Get hand center from landmarks"""
    landmarks = hand_landmarks.landmark
    palm_indices = [0, 1, 5, 9, 13, 17]
    
    x = sum([landmarks[i].x for i in palm_indices]) / len(palm_indices)
    y = sum([landmarks[i].y for i in palm_indices]) / len(palm_indices)
    
    return int(x * w), int(y * h)

def is_hand_aligned(hand_landmarks, template_center, template_size, w, h):
    """Check if hand is aligned with template"""
    center_x, center_y = get_hand_center(hand_landmarks, w, h)
    template_x, template_y = template_center
    template_w, template_h = template_size
    
    template_cx = template_x + template_w // 2
    template_cy = template_y + template_h // 2
    
    tolerance = 80
    center_aligned = (abs(center_x - template_cx) < tolerance and
                     abs(center_y - template_cy) < tolerance)
    
    key_landmarks = [0, 4, 8, 12, 16, 20]
    landmarks_in_template = 0
    
    for landmark_idx in key_landmarks:
        lm = landmarks[landmark_idx]
        lm_x, lm_y = int(lm.x * w), int(lm.y * h)
        
        if (template_x - 30 < lm_x < template_x + template_w + 30 and
            template_y - 30 < lm_y < template_y + template_h + 30):
            landmarks_in_template += 1
    
    landmarks_aligned = landmarks_in_template >= 3
    
    return center_aligned and landmarks_aligned, center_x, center_y

def draw_hand_skeleton(frame, hand_landmarks, w, h):
    """Draw hand skeleton with red dots and white lines"""
    landmarks = hand_landmarks.landmark
    
    connections = mp_hands.HAND_CONNECTIONS
    for connection in connections:
        start_idx = connection[0]
        end_idx = connection[1]
        
        start_point = (int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h))
        end_point = (int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h))
        
        cv2.line(frame, start_point, end_point, (255, 255, 255), 2)
    
    for landmark in landmarks:
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
        cv2.circle(frame, (x, y), 6, (255, 255, 255), 1)

def draw_verification_circle(frame, center_x, center_y, progress, radius=150):
    """Draw circular progress indicator"""
    cv2.circle(frame, (center_x, center_y), radius, (50, 50, 50), 8)
    cv2.circle(frame, (center_x, center_y), radius - 30, (50, 50, 50), 3)
    
    if progress > 0:
        start_angle = -90
        end_angle = start_angle + (360 * progress)
        cv2.ellipse(frame, (center_x, center_y), (radius, radius), 0, 
                   start_angle, end_angle, (0, 255, 0), 8)
    
    cv2.circle(frame, (center_x, center_y), 12, (255, 255, 255), -1)
    cv2.circle(frame, (center_x, center_y), 8, (50, 50, 50), -1)

def draw_verifying_text(frame, center_x, center_y):
    """Draw 'Verifying...' text below the hand"""
    text = "Verifying..."
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 3
    color = (255, 255, 255)
    
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = center_x - text_size[0] // 2
    text_y = center_y + 200
    
    cv2.putText(frame, text, (text_x + 2, text_y + 2), font, font_scale, (0, 0, 0), thickness)
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness)

def draw_ui_panel(frame):
    """Draw main UI panel"""
    h, w = frame.shape[:2]
    
    cv2.rectangle(frame, (0, 0), (w, 80), (0, 0, 0), -1)
    cv2.rectangle(frame, (0, 0), (w, 80), (100, 100, 100), 3)
    
    cv2.putText(frame, "HAND VERIFICATION SYSTEM", (30, 45), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    
    cv2.putText(frame, "Place your LEFT hand on the template to verify", (30, 70), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    
    cv2.rectangle(frame, (0, h-60), (w, h), (0, 0, 0), -1)
    cv2.rectangle(frame, (0, h-60), (w, h), (100, 100, 100), 3)
    
    cv2.putText(frame, "ESC: Exit", (30, h-25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, "SPACE: Reset", (180, h-25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

# Initialize
print("🚀 Initializing Hand Verification System...")
video_exists = check_video_file()

if not video_exists:
    print("❌ video/login.mp4 not found! Please add your video file.")
else:
    print("✅ video/login.mp4 found!")

cv2.namedWindow('Hand Scanner', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Hand Scanner', WINDOW_WIDTH, WINDOW_HEIGHT)

print("\n🎯 Hand Verification System Started!")
print("Place your LEFT hand on the template to verify!")

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to read from camera")
        break
    
    frame_count += 1
    
    frame = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    center_x = w // 2 - template_w // 2
    center_y = h // 2 - template_h // 2
    
    draw_ui_panel(frame)
    
    if not video_playing:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
    else:
        results = None
    
    hand_aligned = False
    hand_center = None
    current_handedness = None
    
    if results and results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            if results.multi_handedness:
                current_handedness = results.multi_handedness[idx].classification[0].label
            
            draw_hand_skeleton(frame, hand_landmarks, w, h)
            
            hand_aligned, hand_center_x, hand_center_y = is_hand_aligned(
                hand_landmarks, (center_x, center_y), (template_w, template_h), w, h
            )
            
            hand_center = (hand_center_x, hand_center_y)
            
            if hand_aligned:
                if current_handedness == ACCEPTED_HAND:
                    if not verifying and not video_playing:
                        verifying = True
                        verification_start_time = time.time()
                        verification_progress = 0.0
                        correct_hand_detected = True
                        wrong_hand_attempt = False
                        print(f"✅ Correct hand ({ACCEPTED_HAND}) detected! Starting verification...")
                        play_sound_effect("verification_start")
                    elif verifying:
                        correct_hand_detected = True
                    if hand_center:
                        last_valid_hand_center = hand_center
                else:
                    current_time = time.time()
                    if not wrong_hand_attempt or (current_time - wrong_hand_message_time) > 2.0:
                        wrong_hand_attempt = True
                        wrong_hand_message_time = current_time
                        verifying = False
                        verification_progress = 0.0
                        correct_hand_detected = False
                        print(f"❌ Wrong hand ({current_handedness}) detected! Use {ACCEPTED_HAND} hand.")
                    
                    if (current_time - last_wrong_hand_sound_time) >= wrong_hand_sound_cooldown:
                        play_sound_effect("wrong_hand")
                        last_wrong_hand_sound_time = current_time
            else:
                if verifying:
                    time_since_start = time.time() - verification_start_time
                    if time_since_start > 5.0 and verification_progress < 0.1:
                        verifying = False
                        verification_progress = 0.0
                        correct_hand_detected = False
                        last_valid_hand_center = None
                        print("⚠️ Hand alignment lost for too long. Please reposition your hand.")
                if wrong_hand_attempt and (time.time() - wrong_hand_message_time) > 1.0:
                    wrong_hand_attempt = False
    
    if verifying:
        elapsed = time.time() - verification_start_time
        verification_progress = min(elapsed / verification_duration, 1.0)
        
        draw_center = hand_center if hand_center else last_valid_hand_center
        
        if draw_center:
            draw_verification_circle(frame, draw_center[0], draw_center[1], verification_progress)
            draw_verifying_text(frame, draw_center[0], draw_center[1])
        else:
            draw_verification_circle(frame, w // 2, h // 2, verification_progress)
            draw_verifying_text(frame, w // 2, h // 2)
        
        progress_percent = int(verification_progress * 100)
        if progress_percent > 0 and progress_percent % 25 == 0:
            print(f"📊 Verification progress: {progress_percent}%")
        
        if verifying and elapsed >= verification_duration and not video_playing:
            if video_exists:
                print("🎯 Verification complete! Starting video...")
                
                verifying = False
                verification_progress = 0.0
                correct_hand_detected = False
                
                video_playing = True
                
                play_sound_effect("verification_complete")
                
                try:
                    print("🎬 Creating video thread...")
                    video_thread = threading.Thread(target=play_video_with_audio, daemon=True)
                    video_thread.start()
                    print("✅ Video thread started successfully!")
                    time.sleep(0.1)
                except Exception as e:
                    print(f"❌ Error starting video thread: {e}")
                    video_playing = False
            else:
                print("⚠️ Video file not found! Cannot play video.")
                verifying = False
                verification_progress = 0.0
                correct_hand_detected = False
    
    if wrong_hand_attempt and (time.time() - wrong_hand_message_time) < 3.0:
        text = f"❌ Wrong hand! Use {ACCEPTED_HAND} hand."
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        color = (0, 0, 255)
        
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = w // 2 - text_size[0] // 2
        text_y = h // 2 + 250
        
        cv2.rectangle(frame, 
                     (text_x - 10, text_y - text_size[1] - 10),
                     (text_x + text_size[0] + 10, text_y + 10),
                     (0, 0, 0), -1)
        
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness)
    
    if not video_playing:
        overlay_image_alpha(frame, template, (center_x, center_y))
        
        if hand_center:
            cv2.circle(frame, hand_center, 8, (255, 0, 255), -1)
            cv2.circle(frame, hand_center, 12, (255, 255, 255), 2)
        
        if current_handedness:
            hand_text = f"Hand: {current_handedness}"
            text_size = cv2.getTextSize(hand_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.putText(frame, hand_text, (w - text_size[0] - 20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                       (0, 255, 0) if current_handedness == ACCEPTED_HAND else (0, 0, 255), 2)
        
        cv2.imshow('Hand Scanner', frame)
        
        key = cv2.waitKey(5) & 0xFF
        if key == 27:
            break
        elif key == 32:
            print("🔄 Detector reset")
            verifying = False
            verification_progress = 0.0
            wrong_hand_attempt = False
    else:
        cv2.waitKey(1)
        if not video_playing and video_thread and not video_thread.is_alive():
            print("✅ Video finished - Exiting application")
            break

print("🛑 Shutting down...")
video_playing = False
cap.release()
cv2.destroyAllWindows()
print("✅ Hand Verification System closed successfully!")
