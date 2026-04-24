import cv2
import numpy as np
import os
import time
import argparse
import platform
import shutil
import subprocess
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe import ImageFormat
import mediapipe as mp

print("✅ Using MediaPipe new API (0.10+)")

# Download model if needed
model_file = "hand_landmarker.task"
if not os.path.exists(model_file):
    print("📥 Downloading MediaPipe hand landmarker model...")
    import urllib.request
    import ssl
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    model_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    try:
        with urllib.request.urlopen(model_url, context=ssl_context) as response:
            with open(model_file, 'wb') as out_file:
                out_file.write(response.read())
        print("✅ Model downloaded!")
    except Exception as e:
        print(f"❌ Download failed: {e}")
        raise

# Create hand detector
base_options = base_options_module.BaseOptions(model_asset_path=model_file)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
hands = vision.HandLandmarker.create_from_options(options)
USE_NEW_API = True

# Settings
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
VERIFICATION_DURATION = 2.0
WRONG_HAND_WARNING_DURATION = 1.2
ACCEPTED_HAND = "Left"
SWAP_HANDEDNESS = True

# Global variables
video_playing = False
verification_progress = 0.0
verifying = False
verification_start_time = 0
wrong_hand_shown = False
wrong_hand_time = 0
wrong_hand_sound_time = 0
wrong_hand_progress = 0.0
wrong_hand_tick_time = 0

# Load template
template = cv2.imread('hand_template.png', cv2.IMREAD_UNCHANGED)
if template is None:
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

# Hand connections for drawing skeleton
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]

def play_sound_effect(sound_type):
    """Play simple feedback sounds."""
    def beep_fallback():
        try:
            print("\a", end="", flush=True)
        except Exception:
            pass

    system_name = platform.system()

    try:
        if system_name == "Darwin":
            if sound_type == "wrong_hand":
                # Hard "dat dat" style warning sound.
                subprocess.Popen(
                    [
                        "/bin/sh",
                        "-c",
                        "afplay /System/Library/Sounds/Basso.aiff >/dev/null 2>&1; "
                        "sleep 0.07; "
                        "afplay /System/Library/Sounds/Basso.aiff >/dev/null 2>&1",
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            elif sound_type == "success":
                os.system('afplay /System/Library/Sounds/Glass.aiff >/dev/null 2>&1 &')
            else:
                os.system('afplay /System/Library/Sounds/Ping.aiff >/dev/null 2>&1 &')
        elif system_name == "Windows":
            import winsound
            if sound_type == "wrong_hand":
                winsound.Beep(180, 180)
                time.sleep(0.07)
                winsound.Beep(180, 180)
            elif sound_type == "success":
                winsound.Beep(900, 180)
            else:
                winsound.Beep(700, 120)
        else:
            beep_fallback()
    except Exception:
        beep_fallback()

def play_video():
    """Play login video in fullscreen mode with audio when possible."""
    global video_playing
    audio_process = None
    try:
        video_path = 'video/login.mp4'
        if not os.path.exists(video_path):
            print(f"❌ Video not found: {video_path}")
            video_playing = False
            return

        ffplay_path = shutil.which("ffplay")
        if ffplay_path:
            print("🎬 Playing video with audio in fullscreen (ffplay)...")
            subprocess.run(
                [
                    ffplay_path,
                    "-autoexit",
                    "-fs",
                    "-loglevel", "error",
                    "-hide_banner",
                    video_path,
                ],
                check=False,
            )
            print("✅ Video finished")
            return

        print("⚠️ ffplay not found. Falling back to OpenCV playback (no audio).")
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            print(f"❌ Failed to open video: {video_path}")
            video_playing = False
            return

        # macOS fallback: play audio track with afplay while OpenCV renders frames.
        if platform.system() == "Darwin" and shutil.which("afplay"):
            try:
                audio_process = subprocess.Popen(
                    ["afplay", video_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                print("🔊 Audio playback started with afplay.")
            except Exception as audio_error:
                print(f"⚠️ Audio fallback failed: {audio_error}")

        cv2.namedWindow('Hand Scanner', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('Hand Scanner', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_delay = 1.0 / fps if fps > 0 else 1.0 / 30
        display_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or WINDOW_WIDTH
        display_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or WINDOW_HEIGHT

        while video.isOpened() and video_playing:
            ret, frame = video.read()
            if not ret:
                break

            frame = cv2.resize(frame, (display_width, display_height))
            cv2.imshow('Hand Scanner', frame)
            if cv2.waitKey(int(frame_delay * 1000)) & 0xFF == 27:
                break

        video.release()
        cv2.setWindowProperty('Hand Scanner', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        print("✅ Video finished")
    except Exception as e:
        print(f"❌ Video error: {e}")
    finally:
        if audio_process and audio_process.poll() is None:
            try:
                audio_process.terminate()
            except Exception:
                pass
        video_playing = False

def draw_hand_skeleton(frame, hand_landmarks, w, h):
    """Draw enhanced hand skeleton with glow particles."""
    landmarks = hand_landmarks.landmark
    pulse = 0.5 + 0.5 * np.sin(time.time() * 8.0)
    glow_radius = int(8 + 4 * pulse)

    # Draw soft glow on separate layer for smoother particles.
    glow_layer = frame.copy()
    
    # Draw connections (white lines)
    for start_idx, end_idx in HAND_CONNECTIONS:
        start = (int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h))
        end = (int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h))
        cv2.line(glow_layer, start, end, (180, 220, 255), 4)
        cv2.line(frame, start, end, (255, 255, 255), 2)
    
    # Draw landmarks with pulse glow + tiny orbit particles.
    for idx, landmark in enumerate(landmarks):
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        orbit_angle = time.time() * 4.0 + idx * 0.4
        orbit_x = int(x + np.cos(orbit_angle) * (glow_radius + 2))
        orbit_y = int(y + np.sin(orbit_angle) * (glow_radius + 2))

        cv2.circle(glow_layer, (x, y), glow_radius, (50, 80, 255), -1)
        cv2.circle(glow_layer, (orbit_x, orbit_y), 2, (180, 220, 255), -1)
        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
        cv2.circle(frame, (x, y), 7, (255, 255, 255), 1)

    cv2.addWeighted(glow_layer, 0.22, frame, 0.78, 0, frame)

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
    template_cx = template_x + template_size[0] // 2
    template_cy = template_y + template_size[1] // 2
    
    # Tolerance for alignment
    tolerance = 100
    return abs(center_x - template_cx) < tolerance and abs(center_y - template_cy) < tolerance

def draw_verification_circle(frame, center_x, center_y, progress, color=(0, 255, 0)):
    """Draw circular progress indicator"""
    radius = 150
    cv2.circle(frame, (center_x, center_y), radius, (50, 50, 50), 8)
    
    if progress > 0:
        end_angle = -90 + (360 * progress)
        cv2.ellipse(frame, (center_x, center_y), (radius, radius), 0, -90, end_angle, color, 8)
    
    cv2.circle(frame, (center_x, center_y), 12, (255, 255, 255), -1)
    cv2.circle(frame, (center_x, center_y), 8, (50, 50, 50), -1)

def overlay_image_alpha(img, img_overlay, pos):
    """Overlay image with alpha channel"""
    x, y = pos
    h, w = img.shape[:2]
    overlay_h, overlay_w = img_overlay.shape[:2]
    
    if x < 0 or y < 0 or x + overlay_w > w or y + overlay_h > h:
        return
    
    if img_overlay.shape[2] == 4:
        alpha = img_overlay[:, :, 3] / 255.0
        for c in range(3):
            img[y:y+overlay_h, x:x+overlay_w, c] = (
                alpha * img_overlay[:, :, c] +
                (1 - alpha) * img[y:y+overlay_h, x:x+overlay_w, c]
            )

def parse_args():
    parser = argparse.ArgumentParser(description="Hand scanner camera app")
    parser.add_argument(
        "--camera-index",
        type=int,
        default=None,
        help="Use specific camera index (e.g. 1 for iPhone Continuity Camera).",
    )
    parser.add_argument(
        "--swap-handedness",
        action="store_true",
        help="Swap detected left/right labels (useful for mirrored camera setups).",
    )
    parser.add_argument(
        "--no-swap-handedness",
        action="store_true",
        help="Do not swap detected left/right labels.",
    )
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="List available camera indices and exit.",
    )
    return parser.parse_args()

def normalize_handedness(label, swap_handedness):
    if not swap_handedness:
        return label
    if label == "Left":
        return "Right"
    if label == "Right":
        return "Left"
    return label

def open_camera(preferred_index=None):
    """Open preferred camera or use system/default camera."""
    if preferred_index is not None:
        candidate_indices = [preferred_index]
    else:
        default_cap = cv2.VideoCapture(0)
        if default_cap.isOpened():
            ok, _ = default_cap.read()
            if ok:
                print("📷 Camera selected: system default")
                return default_cap, 0
        default_cap.release()
        candidate_indices = [0, 1, 2, 3, 4, 5]

    for cam_idx in candidate_indices:
        cap = cv2.VideoCapture(cam_idx, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            cap.release()
            continue

        ok, _ = cap.read()
        if ok:
            print(f"📷 Camera selected: index {cam_idx}")
            return cap, cam_idx

        cap.release()

    return None, None

def list_cameras(max_index=8):
    print("📷 Available camera indices:")
    found = False
    for cam_idx in range(max_index + 1):
        cap = cv2.VideoCapture(cam_idx, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            cap.release()
            continue
        ok, frame = cap.read()
        if ok and frame is not None:
            h, w = frame.shape[:2]
            print(f"  - index {cam_idx}: {w}x{h}")
            found = True
        cap.release()
    if not found:
        print("  (none)")

# Setup camera
print("📷 Starting camera...")
args = parse_args()
if args.list_cameras:
    list_cameras()
    raise SystemExit(0)

if args.swap_handedness:
    SWAP_HANDEDNESS = True
elif args.no_swap_handedness:
    SWAP_HANDEDNESS = False

cap, selected_camera_index = open_camera(args.camera_index)
if cap is None:
    raise RuntimeError("❌ No working camera found. Try --camera-index 0 or --camera-index 1.")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, WINDOW_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_HEIGHT)

cv2.namedWindow('Hand Scanner', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Hand Scanner', WINDOW_WIDTH, WINDOW_HEIGHT)

print("✅ System ready! Place your LEFT hand on the template to verify.")
print("Press ESC to exit.\n")

# Main loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    
    # Calculate template position (center)
    template_x = w // 2 - template_w // 2
    template_y = h // 2 - template_h // 2
    
    # Draw UI
    cv2.rectangle(frame, (0, 0), (w, 80), (0, 0, 0), -1)
    cv2.rectangle(frame, (0, 0), (w, 80), (100, 100, 100), 3)
    cv2.putText(frame, "BIOMETRIC HAND SCANNER", (30, 45),
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    cv2.putText(frame, "Place your hand on the template", (30, 70),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    
    # Process hands
    if not video_playing:
        hand_detected = False
        is_correct_hand = False
        hand_center = None
        
        # New API (MediaPipe 0.10+)
        mp_image = mp.Image(image_format=ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        results = hands.detect(mp_image)
        
        if results.hand_landmarks:
            hand_landmarks = results.hand_landmarks[0]
            raw_handedness = results.handedness[0][0].category_name if results.handedness else "Unknown"
            handedness_label = normalize_handedness(raw_handedness, SWAP_HANDEDNESS)
            
            # Convert New API landmarks to old format for compatibility
            class Landmark:
                def __init__(self, x, y, z=0):
                    self.x = x
                    self.y = y
                    self.z = z
            
            class LandmarkList:
                def __init__(self, landmarks):
                    self.landmark = [Landmark(lm.x, lm.y, lm.z) for lm in landmarks]
            
            hand_landmarks = LandmarkList(hand_landmarks)
            
            # Draw skeleton
            draw_hand_skeleton(frame, hand_landmarks, w, h)
            hand_center = get_hand_center(hand_landmarks, w, h)
            
            # Check hand type
            is_correct_hand = handedness_label == ACCEPTED_HAND
            hand_detected = is_hand_aligned(hand_landmarks, (template_x, template_y), (template_w, template_h), w, h)
            
            # Show hand type
            color = (0, 255, 0) if is_correct_hand else (0, 0, 255)
            cv2.putText(frame, f"Hand: {handedness_label}", (w - 150, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Handle verification
        if hand_detected and is_correct_hand:
            if not verifying:
                verifying = True
                verification_start_time = time.time()
                print(f"✅ {ACCEPTED_HAND} hand detected! Verifying...")
            
            elapsed = time.time() - verification_start_time
            verification_progress = min(elapsed / VERIFICATION_DURATION, 1.0)
            
            if hand_center:
                draw_verification_circle(frame, hand_center[0], hand_center[1], verification_progress)
                cv2.putText(frame, "Verifying...", (hand_center[0] - 60, hand_center[1] + 200),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            
            # Verification complete
            if verification_progress >= 1.0:
                print("🔓 Verification complete! Opening video...")
                play_sound_effect("success")
                verifying = False
                verification_progress = 0.0
                video_playing = True
                play_video()
                break
        else:
            verifying = False
            verification_progress = 0.0
            
            # Wrong-hand warning: fill red bar first, then play hard warning.
            if hand_detected and not is_correct_hand and not wrong_hand_shown:
                wrong_hand_shown = True
                wrong_hand_time = time.time()
                wrong_hand_progress = 0.0
                wrong_hand_tick_time = 0.0
                print(f"⚠️ Wrong hand detected. Waiting warning bar to complete...")
            
            if wrong_hand_shown and (time.time() - wrong_hand_time) < 2.0:
                elapsed_wrong = time.time() - wrong_hand_time
                wrong_hand_progress = min(elapsed_wrong / WRONG_HAND_WARNING_DURATION, 1.0)

                # While red bar is filling, play short ticks.
                if wrong_hand_progress < 1.0 and (time.time() - wrong_hand_tick_time) > 0.22:
                    play_sound_effect("neutral")
                    wrong_hand_tick_time = time.time()

                # After bar completes, trigger hard "dat dat" alert once.
                if wrong_hand_progress >= 1.0 and (time.time() - wrong_hand_sound_time) > 0.8:
                    print(f"❌ Wrong hand confirmed! Please use {ACCEPTED_HAND} hand.")
                    play_sound_effect("wrong_hand")
                    wrong_hand_sound_time = time.time()

                if hand_center:
                    draw_verification_circle(
                        frame,
                        hand_center[0],
                        hand_center[1],
                        wrong_hand_progress,
                        color=(0, 0, 255),
                    )
                warning_text = "Scanning hand..."
                cv2.putText(frame, warning_text, (w // 2 - 170, h // 2),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            elif wrong_hand_shown:
                wrong_hand_shown = False
                wrong_hand_progress = 0.0
        
        # Overlay template
        overlay_image_alpha(frame, template, (template_x, template_y))
    
    cv2.imshow('Hand Scanner', frame)
    
    # Keyboard input
    key = cv2.waitKey(5) & 0xFF
    if key == 27:  # ESC
        break

# Cleanup
print("\n🛑 Shutting down...")
video_playing = False
cap.release()
cv2.destroyAllWindows()
print("✅ Closed successfully!")
