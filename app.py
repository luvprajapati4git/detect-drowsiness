import streamlit as st
import cv2
from ultralytics import YOLO
import mediapipe as mp
import numpy as np
import time
from collections import deque
import threading
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
    
    /* Hide Streamlit header but KEEP the sidebar toggle button */
    header[data-testid="stHeader"] {
        background: transparent !important;
        color: transparent !important;
    }
    
    header[data-testid="stHeader"] button {
        color: #58a6ff !important;
    }
    
    [data-testid="stFooter"] { display: none; }
    
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 0rem !important;
        max-width: 98% !important;
    }
    
    .stApp { 
        background: radial-gradient(circle at top left, #0d1117, #010409);
        color: #e6edf3;
        font-family: 'Inter', sans-serif;
    }
    
    .glass-card {
        background: rgba(23, 27, 33, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 15px;
        margin-bottom: 12px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    
    /* Curve the WebRTC Video Window */
    iframe {
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
    }
    
    .status-card { 
        padding: 18px; 
        border-radius: 14px; 
        font-family: 'Orbitron', sans-serif;
        font-weight: 700; 
        text-align: center; 
        font-size: 20px; 
        letter-spacing: 2px;
        transition: all 0.4s ease;
    }
    
    .safe { border: 1px solid #00ff88; color: #00ff88; background: rgba(0, 255, 136, 0.05); box-shadow: 0 0 20px rgba(0, 255, 136, 0.1); }
    .warning-ui { border: 1px solid #ffcc00; color: #ffcc00; background: rgba(255, 204, 0, 0.05); box-shadow: 0 0 20px rgba(255, 204, 0, 0.1); }
    .danger { border: 2px solid #ff3366; color: #ff3366; background: rgba(255, 51, 102, 0.1); box-shadow: 0 0 30px rgba(255, 51, 102, 0.3); animation: pulse-danger 1.5s infinite; }
    
    @keyframes pulse-danger { 0% { transform: scale(1); } 50% { transform: scale(1.02); } 100% { transform: scale(1); } }
    
    [data-testid="stMetricValue"] { font-family: 'Orbitron', sans-serif; font-size: 1.6rem !important; color: #58a6ff !important; }
    .sidebar-title { font-family: 'Orbitron', sans-serif; font-size: 1.4rem; font-weight: 700; color: #58a6ff; margin-bottom: 15px; }
</style>
""", unsafe_allow_html=True)

# --- MediaPipe Setup ---
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

# --- Audio Logic ---
def play_local_beep():
    if winsound:
        try: winsound.Beep(1200, 400)
        except: pass

def play_browser_audio():
    st.components.v1.html("""<script>(function(){var c=new(window.AudioContext||window.webkitAudioContext)();var o=c.createOscillator();var g=c.createGain();o.type='sawtooth';o.frequency.setValueAtTime(800,c.currentTime);g.gain.setValueAtTime(0.3,c.currentTime);g.gain.exponentialRampToValueAtTime(0.01,c.currentTime+0.4);o.connect(g);g.connect(c.destination);o.start();o.stop(c.currentTime+0.4);})();</script>""", height=0)

# --- Logic & Model ---
@st.cache_resource
def load_yolo(): return YOLO("best.pt")
yolo_model = load_yolo()

# --- State ---
if 'history' not in st.session_state: st.session_state.history = deque([0.3]*50, maxlen=50)
if 'last_sound_time' not in st.session_state: st.session_state.last_sound_time = 0

with st.sidebar:
    st.markdown('<p class="sidebar-title">🛡️ GUARDIAN PRO</p>', unsafe_allow_html=True)
    st.markdown("Advanced Driver Monitoring System")
    st.divider()
    
    with st.expander("Sensitivity Settings", expanded=True):
        ear_thresh = st.slider("Eye Closure (EAR)", 0.15, 0.35, 0.22)
        mar_thresh = st.slider("Yawn Detection (MAR)", 0.1, 0.8, 0.4)
        alert_delay = st.slider("Trigger Delay (s)", 0.2, 3.0, 0.8)
    
    with st.expander("Alert Preferences", expanded=True):
        enable_browser_sound = st.checkbox("Browser Audio Alert", value=True)
        enable_local_sound = st.checkbox("Local System Beep", value=True)
        run_system = st.checkbox("Activate Monitoring", value=True)
    
    if st.button("🔊 Test Alert System", use_container_width=True): 
        play_browser_audio()
        if enable_local_sound: threading.Thread(target=play_local_beep, daemon=True).start()

# --- Video Processing Class ---
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.d_start = None
        self.ear = 0.3
        self.mar = 0.1
        self.elapsed = 0.0
        self.status = "SECURE"
        
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 1. MediaPipe
        mp_res = face_mesh.process(rgb_img)
        is_drowsy_mp = False
        if mp_res.multi_face_landmarks:
            landmarks = mp_res.multi_face_landmarks[0].landmark
            self.ear = (get_ear(landmarks, LEFT_EYE) + get_ear(landmarks, RIGHT_EYE)) / 2.0
            self.mar = get_mar(landmarks, MOUTH)
            if self.ear < ear_thresh or self.mar > mar_thresh:
                is_drowsy_mp = True

        # 2. YOLO
        yolo_res = yolo_model.predict(source=img, conf=0.15, imgsz=416, verbose=False)[0]
        is_drowsy_yolo = any(yolo_res.names[int(box.cls[0])].lower() == 'drowsy' for box in yolo_res.boxes)
        
        # 3. Decision Logic
        final_drowsy = is_drowsy_mp or is_drowsy_yolo
        curr = time.time()
        
        if final_drowsy:
            if self.d_start is None: self.d_start = curr
            self.elapsed = curr - self.d_start
            if self.elapsed >= alert_delay:
                self.status = "EMERGENCY"
                # Visual Alert
                cv2.rectangle(img, (0,0), (img.shape[1], img.shape[0]), (0,0,255), 20)
            else:
                self.status = "WARNING"
        else:
            self.d_start = None
            self.elapsed = 0.0
            self.status = "SECURE"

        ann_img = yolo_res.plot()
        if self.status == "EMERGENCY":
            cv2.rectangle(ann_img, (0,0), (img.shape[1], img.shape[0]), (0,0,255), 20)
            
        return ann_img

# --- Layout ---
st.title("🛡️ Guardian AI Monitoring")
col_v, col_s = st.columns([1.6, 1])

with col_v:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    if run_system:
        ctx = webrtc_streamer(
            key="guardian-ai",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
            video_transformer_factory=VideoTransformer,
            async_transform=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)

with col_s:
    status_ui = st.empty()
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    ear_m = m1.empty()
    mar_m = m2.empty()
    timer_m = m3.empty()
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### Eye Aspect Ratio Trend")
    chart_place = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

# --- Dynamic Updates ---
if run_system and ctx and ctx.video_transformer:
    # Update UI from transformer state
    ear = ctx.video_transformer.ear
    mar = ctx.video_transformer.mar
    elapsed = ctx.video_transformer.elapsed
    status = ctx.video_transformer.status
    
    # Status Card
    if status == "EMERGENCY":
        status_ui.markdown('<div class="status-card danger">🚨 EMERGENCY!</div>', unsafe_allow_html=True)
        # Handle Sound (Streamlit main thread)
        curr = time.time()
        if (curr - st.session_state.last_sound_time) > 0.6:
            if enable_browser_sound: play_browser_audio()
            if enable_local_sound: threading.Thread(target=play_local_beep, daemon=True).start()
            st.session_state.last_sound_time = curr
    elif status == "WARNING":
        status_ui.markdown(f'<div class="status-card warning-ui">⚠️ FATIGUE: {elapsed:.1f}s</div>', unsafe_allow_html=True)
    else:
        status_ui.markdown('<div class="status-card safe">✅ SYSTEM SECURE</div>', unsafe_allow_html=True)

    # Metrics & Chart
    ear_m.metric("EAR", f"{ear:.2f}")
    mar_m.metric("MAR", f"{mar:.2f}")
    timer_m.metric("TIMER", f"{elapsed:.1f}s")
    st.session_state.history.append(ear)
    chart_place.line_chart(list(st.session_state.history), height=150)
else:
    status_ui.markdown('<div class="status-card safe">💤 SYSTEM INACTIVE</div>', unsafe_allow_html=True)
