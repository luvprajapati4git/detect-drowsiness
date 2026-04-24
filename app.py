import streamlit as st
import cv2
from ultralytics import YOLO
import mediapipe as mp
import numpy as np
import time
from collections import deque
import threading
import gc
try:
    import winsound
except ImportError:
    winsound = None
from scipy.spatial import distance as dist
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode, RTCConfiguration

# --- Page Config ---
st.set_page_config(page_title="Guardian AI - Pro Local", layout="wide", page_icon="🛡️")

# --- Optimized CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Inter:wght@300;400;600&display=swap');
    header[data-testid="stHeader"] { background: transparent !important; color: transparent !important; }
    [data-testid="stFooter"] { display: none; }
    .block-container { padding-top: 1rem !important; padding-bottom: 0rem !important; max-width: 98% !important; }
    .stApp { background: radial-gradient(circle at top left, #0d1117, #010409); color: #e6edf3; font-family: 'Inter', sans-serif; }
    .glass-card { background: rgba(23, 27, 33, 0.7); backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 20px; padding: 15px; margin-bottom: 12px; }
    iframe { border-radius: 24px; border: 1px solid rgba(255, 255, 255, 0.1); }
    .status-card { padding: 18px; border-radius: 14px; font-family: 'Orbitron', sans-serif; font-weight: 700; text-align: center; font-size: 20px; transition: all 0.4s ease; }
    .safe { border: 1px solid #00ff88; color: #00ff88; background: rgba(0, 255, 136, 0.05); }
    .warning-ui { border: 1px solid #ffcc00; color: #ffcc00; background: rgba(255, 204, 0, 0.05); }
    .danger { border: 2px solid #ff3366; color: #ff3366; background: rgba(255, 51, 102, 0.1); animation: pulse-danger 1.5s infinite; }
    @keyframes pulse-danger { 0% { transform: scale(1); } 50% { transform: scale(1.02); } 100% { transform: scale(1); } }
    [data-testid="stMetricValue"] { font-family: 'Orbitron', sans-serif; font-size: 1.6rem !important; color: #58a6ff !important; }
    .calibration-alert { color: #58a6ff; font-weight: bold; animation: blinker 1.5s linear infinite; }
    @keyframes blinker { 50% { opacity: 0; } }
</style>
""", unsafe_allow_html=True)

# --- MediaPipe Setup ---
import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

LEFT_EYE = [362, 385, 386, 263, 374, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [13, 14, 78, 308]

def get_ear(landmarks, eye_points):
    p2_p6 = dist.euclidean([landmarks[eye_points[1]].x, landmarks[eye_points[1]].y], [landmarks[eye_points[5]].x, landmarks[eye_points[5]].y])
    p3_p5 = dist.euclidean([landmarks[eye_points[2]].x, landmarks[eye_points[2]].y], [landmarks[eye_points[4]].x, landmarks[eye_points[4]].y])
    p1_p4 = dist.euclidean([landmarks[eye_points[0]].x, landmarks[eye_points[0]].y], [landmarks[eye_points[3]].x, landmarks[eye_points[3]].y])
    return (p2_p6 + p3_p5) / (2.0 * p1_p4 + 1e-6)

def get_mar(landmarks, mouth_points):
    v = dist.euclidean([landmarks[mouth_points[0]].x, landmarks[mouth_points[0]].y], [landmarks[mouth_points[1]].x, landmarks[mouth_points[1]].y])
    h = dist.euclidean([landmarks[mouth_points[2]].x, landmarks[mouth_points[2]].y], [landmarks[mouth_points[3]].x, landmarks[mouth_points[3]].y])
    return v / (h + 1e-6)

def play_local_beep():
    if winsound:
        try: winsound.Beep(1200, 400)
        except: pass

def play_browser_audio():
    st.components.v1.html("""<script>(function(){var c=new(window.AudioContext||window.webkitAudioContext)();var o=c.createOscillator();var g=c.createGain();o.type='sawtooth';o.frequency.setValueAtTime(800,c.currentTime);g.gain.setValueAtTime(0.3,c.currentTime);g.gain.exponentialRampToValueAtTime(0.01,c.currentTime+0.4);o.connect(g);g.connect(c.destination);o.start();o.stop(c.currentTime+0.4);})();</script>""", height=0)

@st.cache_resource
def load_yolo(): return YOLO("best.pt")
yolo_model = load_yolo()

# --- Session State ---
if 'last_sound_time' not in st.session_state: st.session_state.last_sound_time = 0
if 'last_gc_time' not in st.session_state: st.session_state.last_gc_time = time.time()
if 'ear_threshold' not in st.session_state: st.session_state.ear_threshold = 0.22
if 'mar_threshold' not in st.session_state: st.session_state.mar_threshold = 0.4
if 'calibration_trigger' not in st.session_state: st.session_state.calibration_trigger = False

with st.sidebar:
    st.markdown("### 🛡️ GUARDIAN PRO")
    
    # Manual Override Sliders (linked to session state)
    st.session_state.ear_threshold = st.slider("Eye Closure (EAR)", 0.15, 0.35, st.session_state.ear_threshold, format="%.3f")
    st.session_state.mar_threshold = st.slider("Yawn Detection (MAR)", 0.1, 0.8, st.session_state.mar_threshold, format="%.3f")
    
    if st.button("🔄 AUTO CALIBRATE (5s)", use_container_width=True):
        st.session_state.calibration_trigger = True
        st.info("Calibration started! Keep eyes open & mouth closed.")

    st.divider()
    alert_delay = st.slider("Trigger Delay (s)", 0.2, 3.0, 1.6)
    enable_browser_sound = st.checkbox("Browser Audio Alert", value=True)
    enable_local_sound = st.checkbox("Local System Beep", value=True)
    run_system = st.checkbox("Activate Monitoring", value=True)

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.d_start = None
        self.ear = 0.3
        self.mar = 0.1
        self.elapsed = 0.0
        self.status = "SECURE"
        self.frame_count = 0
        self.last_yolo_drowsy = False
        
        # Calibration State
        self.calibrating = False
        self.calib_start = None
        self.calib_ears = []
        self.calib_mars = []
        self.calib_finished = False
        
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1
        
        # Pull latest thresholds from session state
        curr_ear_thresh = st.session_state.ear_threshold
        curr_mar_thresh = st.session_state.mar_threshold
        
        # Check if calibration was triggered externally
        if st.session_state.calibration_trigger and not self.calibrating:
            self.calibrating = True
            self.calib_start = time.time()
            self.calib_ears = []
            self.calib_mars = []
            st.session_state.calibration_trigger = False # Reset trigger

        # 1. MediaPipe (ALWAYS for calibration and detection)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_res = face_mesh.process(rgb_img)
        is_drowsy_mp = False
        
        if mp_res.multi_face_landmarks:
            landmarks = mp_res.multi_face_landmarks[0].landmark
            self.ear = (get_ear(landmarks, LEFT_EYE) + get_ear(landmarks, RIGHT_EYE)) / 2.0
            self.mar = get_mar(landmarks, MOUTH)
            
            if self.calibrating:
                self.calib_ears.append(self.ear)
                self.calib_mars.append(self.mar)
                # 5-second calibration window
                if time.time() - self.calib_start > 5.0:
                    avg_ear = np.mean(self.calib_ears)
                    avg_mar = np.mean(self.calib_mars)
                    # Set thresholds: 75% of open EAR, 150% of closed MAR
                    st.session_state.ear_threshold = round(avg_ear * 0.75, 3)
                    st.session_state.mar_threshold = round(avg_mar * 1.60, 3)
                    self.calibrating = False
                    self.calib_finished = True
            
            if not self.calibrating and (self.ear < curr_ear_thresh or self.mar > curr_mar_thresh):
                is_drowsy_mp = True
        
        # 2. YOLO (Only if not calibrating)
        if not self.calibrating and self.frame_count % 2 == 0:
            yolo_res = yolo_model.predict(source=img, conf=0.15, imgsz=416, verbose=False)[0]
            self.last_yolo_drowsy = any(yolo_res.names[int(box.cls[0])].lower() == 'drowsy' for box in yolo_res.boxes)
            del yolo_res

        # 3. Logic
        if self.calibrating:
            self.status = "CALIBRATING"
        else:
            final_drowsy = is_drowsy_mp or self.last_yolo_drowsy
            curr = time.time()
            if final_drowsy:
                if self.d_start is None: self.d_start = curr
                self.elapsed = curr - self.d_start
                self.status = "EMERGENCY" if self.elapsed >= alert_delay else "WARNING"
            else:
                self.d_start = None
                self.elapsed = 0.0
                self.status = "SECURE"

        # Overlays
        if self.calibrating:
            cv2.putText(img, f"CALIBRATING: {5.0 - (time.time()-self.calib_start):.1f}s", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.rectangle(img, (0,0), (img.shape[1], img.shape[0]), (255,255,0), 10)
        elif self.status == "EMERGENCY":
            cv2.rectangle(img, (0,0), (img.shape[1], img.shape[0]), (0,0,255), 15)
        elif self.status == "WARNING":
            cv2.rectangle(img, (0,0), (img.shape[1], img.shape[0]), (0,255,255), 10)
            
        return img

st.title("🛡️ Guardian AI Monitoring")
col_v, col_s = st.columns([1.6, 1])

with col_v:
    if run_system:
        ctx = webrtc_streamer(
            key="guardian-ai",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": {"width": 640, "height": 480, "frameRate": 20}, "audio": False},
            video_transformer_factory=VideoTransformer,
            async_transform=True,
        )

with col_s:
    status_ui = st.empty()
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    ear_m, mar_m, timer_m = m1.empty(), m2.empty(), m3.empty()
    st.markdown('</div>', unsafe_allow_html=True)

if run_system and ctx:
    while ctx.state.playing:
        if ctx.video_transformer:
            curr_t = time.time()
            ear, mar, elapsed, status = ctx.video_transformer.ear, ctx.video_transformer.mar, ctx.video_transformer.elapsed, ctx.video_transformer.status
            
            if status == "CALIBRATING":
                status_ui.markdown('<div class="status-card" style="border:1px solid #58a6ff; color:#58a6ff;">⚖️ CALIBRATING...</div>', unsafe_allow_html=True)
            elif status == "EMERGENCY":
                status_ui.markdown('<div class="status-card danger">🚨 EMERGENCY!</div>', unsafe_allow_html=True)
                if (curr_t - st.session_state.last_sound_time) > 1.0:
                    if enable_browser_sound: play_browser_audio()
                    if enable_local_sound: threading.Thread(target=play_local_beep, daemon=True).start()
                    st.session_state.last_sound_time = curr_t
            elif status == "WARNING":
                status_ui.markdown(f'<div class="status-card warning-ui">⚠️ FATIGUE: {elapsed:.1f}s</div>', unsafe_allow_html=True)
            else:
                status_ui.markdown('<div class="status-card safe">✅ SYSTEM SECURE</div>', unsafe_allow_html=True)

            ear_m.metric("EAR", f"{ear:.3f}")
            mar_m.metric("MAR", f"{mar:.3f}")
            timer_m.metric("TIMER", f"{elapsed:.1f}s")
            
            if ctx.video_transformer.calib_finished:
                st.success(f"Calibration Done! EAR: {st.session_state.ear_threshold} | MAR: {st.session_state.mar_threshold}")
                ctx.video_transformer.calib_finished = False # Reset local flag
                st.rerun() # Force UI refresh to update sliders
                
            if (curr_t - st.session_state.last_gc_time) > 60:
                gc.collect()
                st.session_state.last_gc_time = curr_t
            
        time.sleep(0.5) 
else:
    status_ui.markdown('<div class="status-card safe">💤 SYSTEM INACTIVE</div>', unsafe_allow_html=True)
