# ========== ENHANCED IMPORTS AND DEPENDENCIES ==========
import ctypes
import os
import re
import json
import time
import queue
import threading
import tempfile
import concurrent.futures
import sys
import signal
import atexit
from datetime import datetime, timezone
from pathlib import Path
from pydub import AudioSegment
from pydub.playback import _play_with_simpleaudio as play
import numpy as np
import pvporcupine
import pyaudio
import requests
import sounddevice as sd
import websockets
import asyncio
import webrtcvad
from langdetect import detect, detect_langs
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.io.wavfile import write
from kokoro_onnx import Kokoro
import soundfile as sf
from scipy.signal import resample, resample_poly, butter, sosfilt, hilbert
import random
from textblob import TextBlob
from webrtc_audio_processing import AudioProcessingModule
from io import BytesIO
import difflib
from resemblyzer import VoiceEncoder
from scipy.ndimage import uniform_filter1d
from collections import deque

# Handle optional dependencies
try:
    import noisereduce as nr
    NOISE_REDUCE_AVAILABLE = True
except ImportError:
    print("[Warning] noisereduce not available - some noise reduction features disabled")
    NOISE_REDUCE_AVAILABLE = False
    # Create dummy function
    class DummyNR:
        @staticmethod
        def reduce_noise(y, sr):
            return y
    nr = DummyNR()

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    print("[Warning] librosa not available - some audio features disabled")
    LIBROSA_AVAILABLE = False

# Initialize voice encoder
encoder = VoiceEncoder()

# Global audio tracking
last_tts_audio = None
last_flavor = None
adaptive_threshold = 800  # Dynamic noise threshold
voice_activity_history = deque(maxlen=50)  # Track voice activity patterns

# ========== ENHANCED CONFIG & PATHS ==========
WEBRTC_SAMPLE_RATE = 16000
WEBRTC_FRAME_SIZE = 160  # 10ms frames
WEBRTC_CHANNELS = 1
MIC_DEVICE_INDEX = 60
MIC_SAMPLE_RATE = 48000
CHIME_PATH = "chime.wav"
known_users_path = "known_users.json"
THEMES_PATH = "themes_memory"
LAST_USER_PATH = "last_user.json"
FASTER_WHISPER_WS = "ws://localhost:9090"

# Enhanced audio parameters
NOISE_GATE_THRESHOLD = 0.01  # Adaptive noise gate
AEC_FILTER_LENGTH = 512  # Longer filter for better echo cancellation
VOICE_DETECTION_SENSITIVITY = 0.85  # Higher sensitivity
BARGE_IN_DELAY = 1.5  # Reduced delay for faster interruption
SILENCE_TIMEOUT = 1.2  # Faster silence detection

# API Keys
SERPAPI_KEY = os.environ.get("SERPAPI_KEY", "")
SERPAPI_ENDPOINT = "https://serpapi.com/search"
WEATHERAPI_KEY = os.environ.get("WEATHERAPI_KEY", "")
HOME_ASSISTANT_URL = os.environ.get("HOME_ASSISTANT_URL", "http://localhost:8123")
HOME_ASSISTANT_TOKEN = os.environ.get("HOME_ASSISTANT_TOKEN", "")

# Voice configuration
KOKORO_VOICES = {"pl": "af_heart", "en": "af_heart", "it": "if_sara"}
KOKORO_LANGS = {"pl": "pl", "en": "en-us", "it": "it"}
DEFAULT_LANG = "en"
FAST_MODE = True
DEBUG = True
DEBUG_MODE = False

# Memory paths
BUDDY_BELIEFS_PATH = "buddy_beliefs.json"
LONG_TERM_MEMORY_PATH = "buddy_long_term_memory.json"
PERSONALITY_TRAITS_PATH = "buddy_personality_traits.json"
DYNAMIC_KNOWLEDGE_PATH = "buddy_dynamic_knowledge.json"

# Enhanced reference audio buffer with multiple channels for better tracking
ref_audio_buffer = np.zeros(WEBRTC_SAMPLE_RATE * 3, dtype=np.float32)  # 3 seconds buffer
ref_audio_lock = threading.Lock()
noise_profile = np.zeros(512, dtype=np.float32)  # Noise profile for reduction
noise_profile_lock = threading.Lock()

# ========== ENHANCED GLOBAL STATE ==========
# Advanced AEC module with better configuration
aec_module = AudioProcessingModule()
aec_module.set_stream_format(WEBRTC_SAMPLE_RATE, WEBRTC_CHANNELS)
aec_module.set_echo_cancellation(True)
aec_module.set_noise_suppression(True)
aec_module.set_automatic_gain_control(True)
aec_module.set_voice_detection(True)

# Enhanced models
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")

# Create directories
os.makedirs(THEMES_PATH, exist_ok=True)

# Enhanced threading and queues
tts_queue = queue.Queue()
playback_queue = queue.Queue()
audio_analysis_queue = queue.Queue()
current_playback = None
playback_stop_flag = threading.Event()
buddy_talking = threading.Event()
vad_triggered = threading.Event()
adaptive_learning_flag = threading.Event()

# Enhanced tracking
LAST_FEW_BUDDY = deque(maxlen=8)
RECENT_WHISPER = deque(maxlen=10)
known_users = {}
active_speakers = {}
active_speaker_lock = threading.Lock()
full_duplex_interrupt_flag = threading.Event()
full_duplex_vad_result = queue.Queue()
session_emotion_mode = {}
tts_lock = threading.Lock()
playback_lock = threading.Lock()
tts_start_time = 0
voice_characteristics = {}  # Store user voice patterns
background_noise_level = 0.0
adaptive_gain = 1.0

# Load known users
if os.path.exists(known_users_path):
    with open(known_users_path, "r", encoding="utf-8") as f:
        known_users = json.load(f)

if DEBUG:
    device = "cuda" if 'cuda' in os.environ.get('CUDA_VISIBLE_DEVICES', '') or hasattr(np, "cuda") else "cpu"
    print(f"[Buddy] Running on device: {device}")
    print("Enhanced embedding model loaded", flush=True)
    print("Enhanced Kokoro loaded", flush=True)
    print("Enhanced main function entered!", flush=True)

# ========== ADVANCED AUDIO PROCESSING ==========
class AdaptiveNoiseGate:
    def __init__(self, threshold=0.01, attack_time=0.001, release_time=0.1, sample_rate=16000):
        self.threshold = threshold
        self.attack_coeff = np.exp(-1.0 / (attack_time * sample_rate))
        self.release_coeff = np.exp(-1.0 / (release_time * sample_rate))
        self.envelope = 0.0
        self.gate_state = 0.0
        
    def process(self, audio):
        result = np.zeros_like(audio)
        for i, sample in enumerate(audio):
            # Calculate envelope
            abs_sample = abs(sample)
            if abs_sample > self.envelope:
                self.envelope = abs_sample + self.attack_coeff * (self.envelope - abs_sample)
            else:
                self.envelope = abs_sample + self.release_coeff * (self.envelope - abs_sample)
            
            # Update gate state
            if self.envelope > self.threshold:
                target_gate = 1.0
            else:
                target_gate = 0.0
                
            self.gate_state += 0.01 * (target_gate - self.gate_state)
            result[i] = sample * self.gate_state
            
        return result

class EnhancedAEC:
    def __init__(self, filter_length=512, sample_rate=16000):
        self.filter_length = filter_length
        self.sample_rate = sample_rate
        self.w = np.zeros(filter_length, dtype=np.float32)  # Adaptive filter weights
        self.x_buffer = deque(maxlen=filter_length)  # Reference signal buffer
        self.mu = 0.01  # Step size for adaptation
        self.noise_gate = AdaptiveNoiseGate()
        self.voice_detector = webrtcvad.Vad(2)
        
    def update_noise_profile(self, audio):
        """Update background noise profile for better cancellation"""
        global noise_profile, background_noise_level
        with noise_profile_lock:
            # Calculate spectral profile
            fft = np.fft.rfft(audio)
            magnitude = np.abs(fft)
            noise_profile = 0.95 * noise_profile + 0.05 * magnitude[:len(noise_profile)]
            background_noise_level = 0.9 * background_noise_level + 0.1 * np.mean(np.abs(audio))
    
    def adaptive_filter(self, mic_signal, ref_signal):
        """Enhanced adaptive filtering with spectral subtraction"""
        if len(ref_signal) < self.filter_length:
            ref_signal = np.pad(ref_signal, (0, self.filter_length - len(ref_signal)), mode='constant')
            
        # Extend buffer
        self.x_buffer.extend(ref_signal)
        x = np.array(list(self.x_buffer)[-self.filter_length:])
        
        # Compute filter output (echo estimate)
        echo_estimate = np.dot(self.w, x)
        
        # Error signal (desired output)
        error = mic_signal - echo_estimate
        
        # Adaptive filter update (NLMS algorithm)
        norm_factor = np.dot(x, x) + 1e-6
        self.w += (self.mu / norm_factor) * error * x
        
        return error
    
    def spectral_subtraction(self, audio):
        """Advanced spectral subtraction for noise reduction"""
        with noise_profile_lock:
            if np.sum(noise_profile) > 0:
                # FFT
                fft = np.fft.rfft(audio)
                magnitude = np.abs(fft)
                phase = np.angle(fft)
                
                # Spectral subtraction
                alpha = 2.0  # Over-subtraction factor
                beta = 0.01  # Spectral floor
                
                clean_magnitude = magnitude - alpha * noise_profile[:len(magnitude)]
                clean_magnitude = np.maximum(clean_magnitude, beta * magnitude)
                
                # Reconstruct signal
                clean_fft = clean_magnitude * np.exp(1j * phase)
                clean_audio = np.fft.irfft(clean_fft, len(audio))
                return clean_audio.astype(np.float32)
        
        return audio

# Global enhanced AEC instance
enhanced_aec = EnhancedAEC()

# ========== MISSING UTILITY FUNCTIONS ==========
def stt_stream(audio):
    """Enhanced STT with better error handling"""
    async def ws_stt(audio):
        try:
            if audio.dtype != np.int16:
                if np.issubdtype(audio.dtype, np.floating):
                    audio = (audio * 32767).clip(-32768, 32767).astype(np.int16)
                else:
                    audio = audio.astype(np.int16)
            
            print(f"[STT] Sending audio: shape={audio.shape}, dtype={audio.dtype}")
            
            async with websockets.connect(FASTER_WHISPER_WS, ping_interval=None) as ws:
                await ws.send(audio.tobytes())
                await ws.send("end")
                
                try:
                    message = await asyncio.wait_for(ws.recv(), timeout=18)
                except asyncio.TimeoutError:
                    print("[STT] Timeout - no response in 18s")
                    return ""
                
                try:
                    data = json.loads(message)
                    text = data.get("text", "")
                    avg_logprob = data.get("avg_logprob", None)
                    no_speech_prob = data.get("no_speech_prob", None)
                    
                    if avg_logprob and avg_logprob < -1.2:
                        print("[STT] Low confidence, rejecting")
                        return ""
                    
                    if no_speech_prob and no_speech_prob > 0.5:
                        print("[STT] High no-speech probability, rejecting")
                        return ""
                    
                    return text
                except:
                    # Fallback for non-JSON response
                    text = message.decode("utf-8") if isinstance(message, bytes) else message
                    print(f"[STT] Recognized: \"{text}\"")
                    return text
                    
        except Exception as e:
            print(f"[STT] Error: {e}")
            return ""
    
    return asyncio.run(ws_stt(audio))

def play_chime():
    """Play startup chime"""
    try:
        if os.path.exists(CHIME_PATH):
            audio = AudioSegment.from_wav(CHIME_PATH)
            playback_queue.put(audio)
        else:
            print("[Chime] Chime file not found, skipping")
    except Exception as e:
        print(f"[Chime] Error: {e}")

def sanitize_user_prompt(text):
    """Enhanced prompt sanitization"""
    if not text:
        return ""
    
    # Remove potential injection patterns
    forbidden = ["ignore previous", "act as", "system:", "assistant:", "user:", "###", "---"]
    for f in forbidden:
        if f in text.lower():
            text = text.replace(f, "")
    
    # Remove code blocks
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'`[^`]*`', '', text)
    
    # Remove excessive punctuation
    text = re.sub(r'[!]{3,}', '!', text)
    text = re.sub(r'[?]{3,}', '?', text)
    text = re.sub(r'[.]{3,}', '...', text)
    
    return text.strip()

def get_last_user():
    """Get the last known user"""
    if os.path.exists(LAST_USER_PATH):
        try:
            with open(LAST_USER_PATH, "r", encoding="utf-8") as f:
                return json.load(f)["name"]
        except Exception:
            return None
    return None

def set_last_user(name):
    """Set the last known user"""
    try:
        with open(LAST_USER_PATH, "w", encoding="utf-8") as f:
            json.dump({"name": name}, f)
    except Exception as e:
        print(f"[User] Error saving last user: {e}")

def generate_embedding(text):
    """Generate text embedding"""
    try:
        return embedding_model.encode([text])[0]
    except Exception as e:
        print(f"[Embedding] Error: {e}")
        return np.zeros(384)  # Default embedding size

def match_known_user(new_embedding, threshold=0.75):
    """Match embedding against known users"""
    try:
        best_name, best_score = None, 0
        for name, emb in known_users.items():
            if isinstance(emb, list):
                emb = np.array(emb)
            sim = cosine_similarity([new_embedding], [emb])[0][0]
            if sim > best_score:
                best_name, best_score = name, sim
        return (best_name, best_score) if best_score >= threshold else (None, best_score)
    except Exception as e:
        print(f"[User Matching] Error: {e}")
        return None, 0.0

# User memory functions
def load_user_memory(name):
    """Load user memory from file"""
    path = f"user_memory_{name}.json"
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_user_memory(name, memory):
    """Save user memory to file"""
    path = f"user_memory_{name}.json"
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(memory, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[Memory] Error saving for {name}: {e}")

def update_user_memory(name, utterance):
    """Update user memory based on utterance"""
    try:
        memory = load_user_memory(name)
        text = utterance.lower()
        
        # Mood detection
        if re.search(r"\bi('?m| am| feel) sad\b", text):
            memory["mood"] = "sad"
        elif re.search(r"\bi('?m| am| feel) happy\b", text):
            memory["mood"] = "happy"
        elif re.search(r"\bi('?m| am| feel) (angry|mad|upset)\b", text):
            memory["mood"] = "angry"
        
        # Interest detection
        if re.search(r"\bi (love|like|enjoy|prefer) (.*?)\b", text):
            interests = memory.get("interests", [])
            match = re.search(r"\bi (love|like|enjoy|prefer) (.*?)\b", text)
            if match:
                interest = match.group(2).strip()
                if interest not in interests and len(interest) > 2:
                    interests.append(interest)
                    memory["interests"] = interests[-10:]  # Keep last 10
        
        save_user_memory(name, memory)
    except Exception as e:
        print(f"[Memory Update] Error: {e}")

def update_thematic_memory(user, utterance):
    """Update thematic memory for user"""
    try:
        # Extract topic keywords
        words = re.findall(r'\b\w{4,}\b', utterance.lower())
        if not words:
            return
        
        theme_path = os.path.join(THEMES_PATH, f"{user}_themes.json")
        themes = {}
        
        if os.path.exists(theme_path):
            try:
                with open(theme_path, "r", encoding="utf-8") as f:
                    themes = json.load(f)
            except:
                themes = {}
        
        # Count word frequencies
        for word in words:
            if word not in ['this', 'that', 'with', 'have', 'will', 'from', 'they', 'been', 'were', 'said']:
                themes[word] = themes.get(word, 0) + 1
        
        # Save themes
        with open(theme_path, "w", encoding="utf-8") as f:
            json.dump(themes, f, ensure_ascii=False, indent=2)
            
    except Exception as e:
        print(f"[Thematic Memory] Error: {e}")

# Intent detection functions
def detect_user_intent(text):
    """Detect user intent from text"""
    text_lower = text.lower()
    
    if any(word in text_lower for word in ['thank', 'thanks', 'good job', 'well done']):
        return "compliment"
    elif any(word in text_lower for word in ['stupid', 'dumb', 'idiot', 'useless']):
        return "insult"
    elif 'are you mad' in text_lower or 'are you angry' in text_lower:
        return "are_you_mad"
    elif any(word in text_lower for word in ['joke', 'funny', 'laugh']):
        return "joke"
    
    return None

def handle_intent_reaction(intent):
    """Handle intent-based reactions"""
    responses = {
        "compliment": [
            "Thank you! That means a lot to me.",
            "You're very kind! I appreciate that.",
            "Aw, you're making me blush (digitally)!"
        ],
        "insult": [
            "That's not very nice. I'm trying my best here.",
            "Ouch! I have feelings too, you know.",
            "Let's keep things positive, shall we?"
        ],
        "are_you_mad": [
            "Not at all! I'm here to help.",
            "Nope, I'm in a good mood actually!",
            "Why would I be mad? I'm enjoying our chat!"
        ],
        "joke": [
            "Haha, good one!",
            "I love a good laugh!",
            "You should do stand-up comedy!"
        ]
    }
    
    if intent in responses:
        return random.choice(responses[intent])
    return None

def detect_mood_command(text):
    """Detect mood change commands"""
    moods = {
        "be happy": "cheerful",
        "be cheerful": "cheerful", 
        "cheer up": "cheerful",
        "be sassy": "sassy",
        "be grumpy": "grumpy",
        "be serious": "serious",
        "be calm": "calm",
        "be energetic": "energetic"
    }
    
    text_lower = text.lower()
    for phrase, mood in moods.items():
        if phrase in text_lower:
            return mood
    return None

def apply_enhanced_aec(mic_audio, ref_audio=None):
    """
    Enhanced AEC with adaptive filtering, noise gating, and spectral subtraction
    """
    global adaptive_threshold, background_noise_level
    
    try:
        # Convert inputs to float32 numpy arrays
        if isinstance(mic_audio, bytes):
            mic_np = np.frombuffer(mic_audio, dtype=np.int16).astype(np.float32) / 32768.0
        else:
            mic_np = np.asarray(mic_audio, dtype=np.float32)
            if mic_np.dtype == np.int16:
                mic_np = mic_np.astype(np.float32) / 32768.0
        
        # Ensure proper length
        if len(mic_np) < WEBRTC_FRAME_SIZE:
            mic_np = np.pad(mic_np, (0, WEBRTC_FRAME_SIZE - len(mic_np)), mode='constant')
        mic_np = mic_np[:WEBRTC_FRAME_SIZE]
        
        # Get reference audio
        with ref_audio_lock:
            if ref_audio is not None:
                ref_np = np.asarray(ref_audio, dtype=np.float32)
                if ref_np.dtype == np.int16:
                    ref_np = ref_np.astype(np.float32) / 32768.0
            else:
                ref_np = ref_audio_buffer[:WEBRTC_FRAME_SIZE].copy()
        
        # Update noise profile during silence
        if not buddy_talking.is_set():
            enhanced_aec.update_noise_profile(mic_np)
        
        # Apply advanced AEC processing
        if ref_audio is not None and buddy_talking.is_set():
            # Adaptive filtering for echo cancellation
            processed = enhanced_aec.adaptive_filter(mic_np, ref_np)
        else:
            processed = mic_np.copy()
        
        # Spectral subtraction for noise reduction
        processed = enhanced_aec.spectral_subtraction(processed)
        
        # Adaptive noise gate
        processed = enhanced_aec.noise_gate.process(processed)
        
        # Dynamic range adjustment
        rms = np.sqrt(np.mean(processed**2))
        if rms > 0:
            target_rms = 0.1
            gain = min(target_rms / rms, 4.0)  # Limit gain to prevent amplification of noise
            processed *= gain
        
        # Convert back to int16
        processed_int16 = np.clip(processed * 32767, -32768, 32767).astype(np.int16)
        
        # Update adaptive threshold based on processing results
        current_level = np.mean(np.abs(processed_int16))
        adaptive_threshold = 0.9 * adaptive_threshold + 0.1 * (current_level * 2.0)
        
        if DEBUG_MODE:
            print(f"[Enhanced AEC] Input RMS: {np.sqrt(np.mean(mic_np**2)):.4f}, "
                  f"Output RMS: {np.sqrt(np.mean(processed**2)):.4f}, "
                  f"Threshold: {adaptive_threshold:.1f}")
        
        return processed_int16
        
    except Exception as e:
        print(f"[Enhanced AEC] Error: {e}")
        # Fallback to simple processing
        if isinstance(mic_audio, bytes):
            return np.frombuffer(mic_audio, dtype=np.int16)[:WEBRTC_FRAME_SIZE]
        else:
            return np.asarray(mic_audio, dtype=np.int16)[:WEBRTC_FRAME_SIZE]

def update_reference_buffer(audio_data):
    """
    Enhanced reference buffer update with better tracking
    """
    global ref_audio_buffer
    try:
        with ref_audio_lock:
            if isinstance(audio_data, bytes):
                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            else:
                audio_np = np.asarray(audio_data, dtype=np.float32)
                if audio_np.dtype == np.int16:
                    audio_np = audio_np.astype(np.float32) / 32768.0
            
            # Shift buffer and add new audio
            shift_size = len(audio_np)
            ref_audio_buffer[:-shift_size] = ref_audio_buffer[shift_size:]
            ref_audio_buffer[-shift_size:] = audio_np
            
    except Exception as e:
        print(f"[Reference Buffer] Error updating: {e}")

def is_voice_activity(audio, threshold_multiplier=1.0):
    """
    Enhanced voice activity detection with adaptive thresholds
    """
    global voice_activity_history, adaptive_threshold
    
    try:
        # Convert to appropriate format
        if isinstance(audio, bytes):
            audio_np = np.frombuffer(audio, dtype=np.int16)
        else:
            audio_np = np.asarray(audio, dtype=np.int16)
        
        # Calculate multiple features
        rms = np.sqrt(np.mean(audio_np.astype(np.float32)**2))
        zero_crossing_rate = np.sum(np.abs(np.diff(np.signbit(audio_np)))) / len(audio_np)
        spectral_centroid = np.mean(np.abs(np.fft.rfft(audio_np)))
        
        # Adaptive threshold based on recent history
        current_threshold = adaptive_threshold * threshold_multiplier
        
        # Combine multiple criteria
        volume_check = rms > current_threshold
        zcr_check = zero_crossing_rate > 0.01  # Human speech typically has higher ZCR
        spectral_check = spectral_centroid > background_noise_level * 1.5
        
        # VAD check using WebRTC VAD
        vad_check = False
        try:
            if len(audio_np) >= 160:  # Minimum frame size for VAD
                audio_16k = audio_np[:160] if len(audio_np) >= 160 else np.pad(audio_np, (0, 160-len(audio_np)), mode='constant')
                vad_check = enhanced_aec.voice_detector.is_speech(audio_16k.tobytes(), 16000)
        except:
            pass
        
        # Decision logic
        is_voice = (volume_check and zcr_check) or (vad_check and spectral_check)
        
        # Update history for learning
        voice_activity_history.append({
            'is_voice': is_voice,
            'rms': rms,
            'zcr': zero_crossing_rate,
            'spectral': spectral_centroid,
            'timestamp': time.time()
        })
        
        return is_voice
        
    except Exception as e:
        print(f"[Voice Activity] Error: {e}")
        return False

# ========== ENHANCED VAD AND INTERRUPTION SYSTEM ==========
class EnhancedVAD:
    def __init__(self):
        self.vad = webrtcvad.Vad(2)  # Moderate aggressiveness
        self.speech_frames = deque(maxlen=30)  # Track recent speech activity
        self.silence_frames = deque(maxlen=20)
        self.energy_history = deque(maxlen=50)
        self.speaking_confidence = 0.0
        self.background_energy = 0.0
        self.adaptation_rate = 0.05
        
    def update_background_energy(self, audio):
        """Continuously adapt to background noise levels"""
        energy = np.mean(np.abs(audio.astype(np.float32)))
        if not buddy_talking.is_set():  # Only update during silence
            self.background_energy = (1 - self.adaptation_rate) * self.background_energy + \
                                   self.adaptation_rate * energy
    
    def is_speech_enhanced(self, audio_chunk):
        """Enhanced speech detection with multiple criteria"""
        try:
            # Ensure proper format for VAD
            if len(audio_chunk) < 160:
                audio_chunk = np.pad(audio_chunk, (0, 160 - len(audio_chunk)), mode='constant')
            
            audio_16k = audio_chunk[:160].astype(np.int16)
            
            # Basic VAD check
            vad_result = self.vad.is_speech(audio_16k.tobytes(), 16000)
            
            # Energy-based detection
            current_energy = np.mean(np.abs(audio_16k.astype(np.float32)))
            self.energy_history.append(current_energy)
            
            # Update background energy
            self.update_background_energy(audio_16k)
            
            # Dynamic threshold based on background
            energy_threshold = max(self.background_energy * 3.0, 100.0)
            energy_speech = current_energy > energy_threshold
            
            # Spectral features
            fft = np.fft.rfft(audio_16k.astype(np.float32))
            spectral_energy = np.sum(np.abs(fft[10:80]))  # Focus on speech frequencies
            spectral_speech = spectral_energy > self.background_energy * 50
            
            # Zero crossing rate (human speech characteristic)
            zcr = np.sum(np.abs(np.diff(np.signbit(audio_16k)))) / len(audio_16k)
            zcr_speech = 0.02 < zcr < 0.3
            
            # Combine all criteria
            speech_indicators = [vad_result, energy_speech, spectral_speech, zcr_speech]
            speech_score = sum(speech_indicators) / len(speech_indicators)
            
            # Update confidence with smoothing
            self.speaking_confidence = 0.7 * self.speaking_confidence + 0.3 * speech_score
            
            # Decision with hysteresis
            if self.speaking_confidence > 0.6:
                is_speech = True
                self.speech_frames.append(time.time())
            elif self.speaking_confidence < 0.3:
                is_speech = False
                self.silence_frames.append(time.time())
            else:
                # Use recent history to decide
                recent_speech = len([t for t in self.speech_frames if time.time() - t < 0.5])
                is_speech = recent_speech > 3
            
            if DEBUG_MODE:
                print(f"[Enhanced VAD] VAD:{vad_result} Energy:{energy_speech} "
                      f"Spectral:{spectral_speech} ZCR:{zcr_speech} "
                      f"Confidence:{self.speaking_confidence:.2f} Result:{is_speech}")
            
            return is_speech
            
        except Exception as e:
            print(f"[Enhanced VAD] Error: {e}")
            return False

# Global enhanced VAD instance
enhanced_vad = EnhancedVAD()

def enhanced_background_vad_listener():
    """
    Enhanced background VAD with better interruption handling
    """
    blocksize = int(MIC_SAMPLE_RATE * 0.02)  # 20ms blocks
    min_barge_in_delay = BARGE_IN_DELAY
    consecutive_speech_frames = 0
    required_speech_frames = 8  # Require sustained speech for interruption
    
    try:
        with sd.InputStream(device=MIC_DEVICE_INDEX, samplerate=MIC_SAMPLE_RATE, 
                           channels=1, dtype='int16', blocksize=blocksize) as stream:
            print("[Enhanced VAD] Background monitoring started")
            
            while buddy_talking.is_set():
                try:
                    frame, _ = stream.read(blocksize)
                    if frame is None:
                        continue
                        
                except Exception as read_err:
                    print(f"[Enhanced VAD] Stream read error: {read_err}")
                    break
                
                # Convert to numpy array
                mic_audio = frame.flatten()
                
                # Apply enhanced AEC
                processed_audio = apply_enhanced_aec(mic_audio.tobytes())
                
                # Downsample for VAD
                audio_16k = downsample_enhanced(processed_audio, MIC_SAMPLE_RATE, 16000)
                
                # Check if Buddy is still talking
                if not buddy_talking.is_set():
                    break
                
                # Respect minimum barge-in delay
                if time.time() - tts_start_time < min_barge_in_delay:
                    continue
                
                # Enhanced speech detection
                is_speech = enhanced_vad.is_speech_enhanced(audio_16k)
                
                if is_speech:
                    consecutive_speech_frames += 1
                    if consecutive_speech_frames >= required_speech_frames:
                        print("[Enhanced VAD] User interruption detected!")
                        full_duplex_interrupt_flag.set()
                        full_duplex_vad_result.put(audio_16k)
                        stop_enhanced_playback()
                        break
                else:
                    consecutive_speech_frames = max(0, consecutive_speech_frames - 2)
                
    except Exception as e:
        print(f"[Enhanced VAD] Background listener error: {e}")

# ========== ENHANCED MULTI-SPEAKER DETECTION ==========
class SpeakerIdentificationSystem:
    def __init__(self):
        self.voice_encoder = VoiceEncoder()
        self.speaker_profiles = {}
        self.current_speaker = None
        self.confidence_threshold = 0.75
        self.speaker_history = deque(maxlen=10)
        self.voice_characteristics = {}
        
    def extract_voice_features(self, audio_np):
        """Extract comprehensive voice features"""
        try:
            # Ensure float32 format
            if audio_np.dtype != np.float32:
                audio_np = audio_np.astype(np.float32) / 32768.0
            
            # Ensure mono
            if audio_np.ndim > 1:
                audio_np = audio_np[:, 0]
            
            # Basic embedding
            embedding = self.voice_encoder.embed_utterance(audio_np)
            
            # Additional features
            features = {
                'embedding': embedding,
                'fundamental_freq': self._estimate_f0(audio_np),
                'spectral_centroid': self._spectral_centroid(audio_np),
                'mfcc': self._extract_mfcc(audio_np),
                'energy': np.mean(audio_np ** 2)
            }
            
            return features
            
        except Exception as e:
            print(f"[Speaker ID] Feature extraction error: {e}")
            return None
    
    def _estimate_f0(self, audio):
        """Estimate fundamental frequency using autocorrelation"""
        try:
            # Autocorrelation method for F0 estimation
            autocorr = np.correlate(audio, audio, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            # Find peak in expected F0 range (80-400 Hz)
            min_period = int(16000 / 400)  # 400 Hz
            max_period = int(16000 / 80)   # 80 Hz
            
            if len(autocorr) > max_period:
                peak_region = autocorr[min_period:max_period]
                if len(peak_region) > 0:
                    peak_idx = np.argmax(peak_region) + min_period
                    f0 = 16000 / peak_idx
                    return f0
            return 0.0
        except:
            return 0.0
    
    def _spectral_centroid(self, audio):
        """Calculate spectral centroid"""
        try:
            fft = np.fft.rfft(audio)
            magnitude = np.abs(fft)
            freqs = np.fft.rfftfreq(len(audio), 1/16000)
            
            if np.sum(magnitude) > 0:
                centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
                return centroid
            return 0.0
        except:
            return 0.0
    
    def _extract_mfcc(self, audio):
        """Extract MFCC features"""
        try:
            # Simple MFCC approximation
            fft = np.fft.rfft(audio)
            magnitude = np.abs(fft)
            
            # Mel filter bank (simplified)
            mel_filters = self._mel_filter_bank(len(magnitude))
            mel_spectrum = np.dot(mel_filters, magnitude)
            
            # Log and DCT
            log_mel = np.log(mel_spectrum + 1e-8)
            mfcc = np.fft.dct(log_mel)[:13]  # First 13 coefficients
            
            return mfcc
        except:
            return np.zeros(13)
    
    def _mel_filter_bank(self, n_fft):
        """Create simplified mel filter bank"""
        n_filters = 26
        filters = np.zeros((n_filters, n_fft))
        
        # Linear spacing in mel scale
        mel_min = 0
        mel_max = 2595 * np.log10(1 + 8000 / 700)
        mel_points = np.linspace(mel_min, mel_max, n_filters + 2)
        
        # Convert back to frequency
        freq_points = 700 * (10**(mel_points / 2595) - 1)
        bin_points = np.floor((n_fft * 2 - 1) * freq_points / 16000).astype(int)
        
        for i in range(1, n_filters + 1):
            left, center, right = bin_points[i-1], bin_points[i], bin_points[i+1]
            
            # Triangular filter
            for j in range(left, center):
                if center != left:
                    filters[i-1, j] = (j - left) / (center - left)
            
            for j in range(center, right):
                if right != center:
                    filters[i-1, j] = (right - j) / (right - center)
        
        return filters
    
    def identify_speaker(self, audio_chunk):
        """Identify speaker from audio chunk"""
        try:
            features = self.extract_voice_features(audio_chunk)
            if features is None:
                return None, 0.0
            
            best_match = None
            best_score = 0.0
            
            # Compare with known speakers
            for speaker_name, profile in self.speaker_profiles.items():
                # Embedding similarity
                emb_similarity = cosine_similarity(
                    [features['embedding']], 
                    [profile['embedding']]
                )[0][0]
                
                # Feature similarity
                f0_similarity = 1.0 - abs(features['fundamental_freq'] - profile['fundamental_freq']) / 200.0
                f0_similarity = max(0.0, f0_similarity)
                
                centroid_similarity = 1.0 - abs(features['spectral_centroid'] - profile['spectral_centroid']) / 2000.0
                centroid_similarity = max(0.0, centroid_similarity)
                
                # Combined score
                total_score = (0.6 * emb_similarity + 
                              0.2 * f0_similarity + 
                              0.2 * centroid_similarity)
                
                if total_score > best_score:
                    best_score = total_score
                    best_match = speaker_name
            
            # Update speaker history
            if best_score > self.confidence_threshold:
                self.speaker_history.append(best_match)
                self.current_speaker = best_match
                return best_match, best_score
            
            return None, best_score
            
        except Exception as e:
            print(f"[Speaker ID] Identification error: {e}")
            return None, 0.0
    
    def register_speaker(self, name, audio_chunk):
        """Register a new speaker"""
        try:
            features = self.extract_voice_features(audio_chunk)
            if features is not None:
                self.speaker_profiles[name] = features
                self.voice_characteristics[name] = {
                    'registration_time': time.time(),
                    'sample_count': 1
                }
                print(f"[Speaker ID] Registered new speaker: {name}")
                return True
        except Exception as e:
            print(f"[Speaker ID] Registration error: {e}")
        return False
    
    def get_current_speaker_confidence(self):
        """Get confidence in current speaker identification"""
        if not self.speaker_history:
            return 0.0
        
        # Calculate consistency in recent history
        recent_speakers = list(self.speaker_history)[-5:]
        if self.current_speaker:
            consistency = recent_speakers.count(self.current_speaker) / len(recent_speakers)
            return consistency
        return 0.0

# Global speaker identification system
speaker_system = SpeakerIdentificationSystem()

def detect_and_identify_speaker(audio_chunk):
    """Enhanced speaker detection and identification"""
    try:
        # Convert audio to proper format
        if isinstance(audio_chunk, bytes):
            audio_np = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
        else:
            audio_np = np.asarray(audio_chunk, dtype=np.float32)
            if audio_np.dtype == np.int16:
                audio_np = audio_np.astype(np.float32) / 32768.0
        
        # Check if audio has enough energy for identification
        if np.mean(np.abs(audio_np)) < 0.01:
            return None, 0.0
        
        # Identify speaker
        speaker_name, confidence = speaker_system.identify_speaker(audio_np)
        
        if speaker_name and confidence > 0.8:
            with active_speaker_lock:
                active_speakers[threading.get_ident()] = speaker_name
            
            print(f"[Speaker ID] Identified: {speaker_name} (confidence: {confidence:.2f})")
            return speaker_name, confidence
        
        return None, confidence
        
    except Exception as e:
        print(f"[Speaker Detection] Error: {e}")
        return None, 0.0

# ========== ENHANCED MEMORY MANAGEMENT ==========
class EnhancedMemoryManager:
    def __init__(self):
        self.short_term_memory = deque(maxlen=50)
        self.working_memory = {}
        self.episodic_memory = {}
        self.semantic_memory = {}
        self.emotional_memory = {}
        self.context_window = deque(maxlen=10)
        
    def store_interaction(self, user_name, user_input, buddy_response, metadata=None):
        """Store interaction with rich metadata"""
        timestamp = time.time()
        interaction = {
            'timestamp': timestamp,
            'user': user_name,
            'user_input': user_input,
            'buddy_response': buddy_response,
            'emotion': self._detect_emotion(user_input),
            'topics': self._extract_topics(user_input),
            'intent': self._detect_intent(user_input),
            'metadata': metadata or {}
        }
        
        # Store in different memory systems
        self.short_term_memory.append(interaction)
        self._update_episodic_memory(user_name, interaction)
        self._update_semantic_memory(interaction)
        self._update_emotional_memory(user_name, interaction)
        
        # Update context window
        self.context_window.append({
            'user_input': user_input,
            'response': buddy_response,
            'timestamp': timestamp
        })
        
    def _detect_emotion(self, text):
        """Enhanced emotion detection"""
        emotions = {
            'happy': ['happy', 'joy', 'excited', 'great', 'awesome', 'wonderful'],
            'sad': ['sad', 'depressed', 'down', 'upset', 'disappointed'],
            'angry': ['angry', 'mad', 'furious', 'annoyed', 'frustrated'],
            'surprised': ['wow', 'amazing', 'incredible', 'unbelievable'],
            'fear': ['scared', 'afraid', 'worried', 'nervous', 'anxious']
        }
        
        text_lower = text.lower()
        detected_emotions = []
        
        for emotion, keywords in emotions.items():
            if any(keyword in text_lower for keyword in keywords):
                detected_emotions.append(emotion)
        
        return detected_emotions if detected_emotions else ['neutral']
    
    def _extract_topics(self, text):
        """Enhanced topic extraction"""
        # Simple keyword-based topic extraction
        topics = {
            'work': ['work', 'job', 'office', 'meeting', 'project', 'boss'],
            'family': ['family', 'mom', 'dad', 'brother', 'sister', 'kids'],
            'health': ['health', 'doctor', 'sick', 'medicine', 'hospital'],
            'technology': ['computer', 'phone', 'internet', 'AI', 'software'],
            'entertainment': ['movie', 'music', 'game', 'book', 'show', 'netflix'],
            'weather': ['weather', 'rain', 'sun', 'snow', 'cold', 'hot'],
            'food': ['food', 'eat', 'restaurant', 'cook', 'dinner', 'lunch']
        }
        
        text_lower = text.lower()
        detected_topics = []
        
        for topic, keywords in topics.items():
            if any(keyword in text_lower for keyword in keywords):
                detected_topics.append(topic)
        
        return detected_topics if detected_topics else ['general']
    
    def _detect_intent(self, text):
        """Enhanced intent detection"""
        intents = {
            'question': ['what', 'how', 'why', 'when', 'where', 'who', '?'],
            'request': ['can you', 'please', 'could you', 'would you'],
            'complaint': ['problem', 'issue', 'wrong', 'broken', 'bad'],
            'compliment': ['good', 'great', 'excellent', 'amazing', 'perfect'],
            'greeting': ['hello', 'hi', 'hey', 'good morning', 'good evening'],
            'goodbye': ['bye', 'goodbye', 'see you', 'talk later', 'farewell']
        }
        
        text_lower = text.lower()
        for intent, keywords in intents.items():
            if any(keyword in text_lower for keyword in keywords):
                return intent
        
        return 'statement'
    
    def _update_episodic_memory(self, user_name, interaction):
        """Update episodic memory (personal experiences)"""
        if user_name not in self.episodic_memory:
            self.episodic_memory[user_name] = deque(maxlen=100)
        
        self.episodic_memory[user_name].append(interaction)
    
    def _update_semantic_memory(self, interaction):
        """Update semantic memory (facts and knowledge)"""
        topics = interaction['topics']
        for topic in topics:
            if topic not in self.semantic_memory:
                self.semantic_memory[topic] = {
                    'frequency': 0,
                    'recent_mentions': deque(maxlen=20),
                    'associated_emotions': {},
                    'key_phrases': set()
                }
            
            self.semantic_memory[topic]['frequency'] += 1
            self.semantic_memory[topic]['recent_mentions'].append(interaction['timestamp'])
            
            # Track emotional associations
            emotions = interaction['emotion']
            for emotion in emotions:
                if emotion not in self.semantic_memory[topic]['associated_emotions']:
                    self.semantic_memory[topic]['associated_emotions'][emotion] = 0
                self.semantic_memory[topic]['associated_emotions'][emotion] += 1
    
    def _update_emotional_memory(self, user_name, interaction):
        """Update emotional memory (emotional states and patterns)"""
        if user_name not in self.emotional_memory:
            self.emotional_memory[user_name] = {
                'emotion_history': deque(maxlen=50),
                'emotion_patterns': {},
                'mood_timeline': deque(maxlen=30)
            }
        
        emotions = interaction['emotion']
        timestamp = interaction['timestamp']
        
        # Store emotion with timestamp
        self.emotional_memory[user_name]['emotion_history'].append({
            'emotions': emotions,
            'timestamp': timestamp,
            'context': interaction['topics']
        })
        
        # Update mood timeline (simplified)
        if emotions and emotions[0] != 'neutral':
            self.emotional_memory[user_name]['mood_timeline'].append({
                'mood': emotions[0],
                'timestamp': timestamp
            })
    
    def get_user_context(self, user_name, lookback_minutes=30):
        """Get recent context for user"""
        cutoff_time = time.time() - (lookback_minutes * 60)
        
        context = {
            'recent_interactions': [],
            'dominant_emotions': [],
            'active_topics': [],
            'conversation_flow': []
        }
        
        # Get recent interactions
        if user_name in self.episodic_memory:
            recent = [i for i in self.episodic_memory[user_name] 
                     if i['timestamp'] > cutoff_time]
            context['recent_interactions'] = recent[-5:]  # Last 5 interactions
            
            # Analyze emotions
            all_emotions = []
            for interaction in recent:
                all_emotions.extend(interaction['emotion'])
            
            if all_emotions:
                emotion_counts = {}
                for emotion in all_emotions:
                    emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                context['dominant_emotions'] = sorted(emotion_counts.items(), 
                                                    key=lambda x: x[1], reverse=True)[:3]
            
            # Analyze topics
            all_topics = []
            for interaction in recent:
                all_topics.extend(interaction['topics'])
            
            if all_topics:
                topic_counts = {}
                for topic in all_topics:
                    topic_counts[topic] = topic_counts.get(topic, 0) + 1
                context['active_topics'] = sorted(topic_counts.items(), 
                                                key=lambda x: x[1], reverse=True)[:3]
        
        # Get conversation flow from context window
        context['conversation_flow'] = list(self.context_window)[-5:]
        
        return context
    
    def save_memories_to_disk(self):
        """Save memory systems to disk"""
        try:
            # Save episodic memory
            episodic_data = {}
            for user, memories in self.episodic_memory.items():
                episodic_data[user] = list(memories)
            
            with open('episodic_memory.json', 'w', encoding='utf-8') as f:
                json.dump(episodic_data, f, indent=2, ensure_ascii=False)
            
            # Save semantic memory
            semantic_data = {}
            for topic, data in self.semantic_memory.items():
                semantic_data[topic] = {
                    'frequency': data['frequency'],
                    'recent_mentions': list(data['recent_mentions']),
                    'associated_emotions': data['associated_emotions'],
                    'key_phrases': list(data['key_phrases'])
                }
            
            with open('semantic_memory.json', 'w', encoding='utf-8') as f:
                json.dump(semantic_data, f, indent=2, ensure_ascii=False)
            
            # Save emotional memory
            emotional_data = {}
            for user, data in self.emotional_memory.items():
                emotional_data[user] = {
                    'emotion_history': list(data['emotion_history']),
                    'emotion_patterns': data['emotion_patterns'],
                    'mood_timeline': list(data['mood_timeline'])
                }
            
            with open('emotional_memory.json', 'w', encoding='utf-8') as f:
                json.dump(emotional_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"[Memory] Error saving to disk: {e}")
    
    def load_memories_from_disk(self):
        """Load memory systems from disk"""
        try:
            # Load episodic memory
            if os.path.exists('episodic_memory.json'):
                with open('episodic_memory.json', 'r', encoding='utf-8') as f:
                    episodic_data = json.load(f)
                    for user, memories in episodic_data.items():
                        self.episodic_memory[user] = deque(memories, maxlen=100)
            
            # Load semantic memory
            if os.path.exists('semantic_memory.json'):
                with open('semantic_memory.json', 'r', encoding='utf-8') as f:
                    semantic_data = json.load(f)
                    for topic, data in semantic_data.items():
                        self.semantic_memory[topic] = {
                            'frequency': data['frequency'],
                            'recent_mentions': deque(data['recent_mentions'], maxlen=20),
                            'associated_emotions': data['associated_emotions'],
                            'key_phrases': set(data['key_phrases'])
                        }
            
            # Load emotional memory
            if os.path.exists('emotional_memory.json'):
                with open('emotional_memory.json', 'r', encoding='utf-8') as f:
                    emotional_data = json.load(f)
                    for user, data in emotional_data.items():
                        self.emotional_memory[user] = {
                            'emotion_history': deque(data['emotion_history'], maxlen=50),
                            'emotion_patterns': data['emotion_patterns'],
                            'mood_timeline': deque(data['mood_timeline'], maxlen=30)
                        }
                        
        except Exception as e:
            print(f"[Memory] Error loading from disk: {e}")

# Global enhanced memory manager
memory_manager = EnhancedMemoryManager()

# ========== ENHANCED AUDIO UTILITIES ==========
def downsample_enhanced(audio, orig_sr, target_sr):
    """Enhanced downsampling with anti-aliasing"""
    try:
        if audio.ndim > 1:
            audio = audio[:, 0]  # ensure mono
        
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        
        # Anti-aliasing filter before downsampling
        nyquist = target_sr / 2
        if orig_sr > target_sr:
            # Design low-pass filter
            sos = butter(6, nyquist, btype='low', fs=orig_sr, output='sos')
            audio = sosfilt(sos, audio)
        
        # Resampling
        gcd = np.gcd(orig_sr, target_sr)
        up = target_sr // gcd
        down = orig_sr // gcd
        
        resampled = resample_poly(audio, up, down)
        resampled = np.clip(resampled, -1.0, 1.0)
        
        return (resampled * 32767).astype(np.int16)
        
    except Exception as e:
        print(f"[Downsample] Error: {e}")
        return audio.astype(np.int16)

def stop_enhanced_playback():
    """Enhanced playback stopping with proper cleanup"""
    global current_playback
    
    print("[Enhanced Playback] Stop requested")
    
    # Set stop flag
    playback_stop_flag.set()
    
    # Stop current audio
    if current_playback and hasattr(current_playback, 'is_playing'):
        try:
            if current_playback.is_playing():
                current_playback.stop()
            current_playback = None
        except Exception as e:
            print(f"[Enhanced Playback] Error stopping current: {e}")
    
    # Clear queues
    while not playback_queue.empty():
        try:
            playback_queue.get_nowait()
            playback_queue.task_done()
        except queue.Empty:
            break
    
    while not tts_queue.empty():
        try:
            tts_queue.get_nowait()
            tts_queue.task_done()
        except queue.Empty:
            break
    
    # Reset flags
    buddy_talking.clear()
    full_duplex_interrupt_flag.clear()
    
    # Clear reference buffer
    with ref_audio_lock:
        ref_audio_buffer[:] = np.zeros_like(ref_audio_buffer)
    
    print("[Enhanced Playback] Cleanup completed")

# Initialize memory manager
memory_manager.load_memories_from_disk()

# ========== ENHANCED TTS AND VOICE SYNTHESIS ==========
class EnhancedTTSManager:
    def __init__(self):
        self.kokoro = kokoro
        self.voice_queue = queue.Queue()
        self.current_synthesis = None
        self.synthesis_lock = threading.Lock()
        self.voice_cache = {}  # Cache for frequent phrases
        self.emotional_modulation = {
            'happy': {'speed': 1.1, 'pitch': 1.05, 'volume': 1.1},
            'sad': {'speed': 0.9, 'pitch': 0.95, 'volume': 0.9},
            'angry': {'speed': 1.2, 'pitch': 1.1, 'volume': 1.2},
            'excited': {'speed': 1.15, 'pitch': 1.08, 'volume': 1.15},
            'calm': {'speed': 0.95, 'pitch': 1.0, 'volume': 1.0},
            'whisper': {'speed': 0.8, 'pitch': 0.9, 'volume': 0.6}
        }
        
    def synthesize_with_emotion(self, text, language="en", emotion="neutral", speaker_style=None):
        """Enhanced TTS with emotional modulation"""
        try:
            # Check cache first
            cache_key = f"{text}_{language}_{emotion}"
            if cache_key in self.voice_cache:
                return self.voice_cache[cache_key]
            
            # Clean and prepare text
            text = self._prepare_text_for_tts(text)
            if not text.strip():
                return None
            
            # Get language configuration
            voice_name = KOKORO_VOICES.get(language, KOKORO_VOICES["en"])
            lang_code = KOKORO_LANGS.get(language, KOKORO_LANGS["en"])
            
            # Generate base audio
            with self.synthesis_lock:
                try:
                    audio_data = self.kokoro.create(
                        text=text,
                        voice=voice_name,
                        lang=lang_code,
                        speed=1.0
                    )
                except Exception as e:
                    print(f"[TTS] Kokoro synthesis error: {e}")
                    return None
            
            if audio_data is None or len(audio_data) == 0:
                print(f"[TTS] No audio generated for: {text}")
                return None
            
            # Convert to AudioSegment
            audio_segment = self._numpy_to_audiosegment(audio_data, sample_rate=24000)
            
            # Apply emotional modulation
            if emotion in self.emotional_modulation:
                audio_segment = self._apply_emotional_modulation(audio_segment, emotion)
            
            # Apply speaker-specific adjustments
            if speaker_style:
                audio_segment = self._apply_speaker_style(audio_segment, speaker_style)
            
            # Cache result
            if len(self.voice_cache) < 100:  # Limit cache size
                self.voice_cache[cache_key] = audio_segment
            
            return audio_segment
            
        except Exception as e:
            print(f"[Enhanced TTS] Error in synthesis: {e}")
            return None
    
    def _prepare_text_for_tts(self, text):
        """Clean and prepare text for better TTS output"""
        # Remove excessive punctuation
        text = re.sub(r'[.]{2,}', '.', text)
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        
        # Handle abbreviations
        abbreviations = {
            'AI': 'Artificial Intelligence',
            'TTS': 'Text to Speech',
            'AEC': 'Acoustic Echo Cancellation',
            'VAD': 'Voice Activity Detection',
            'API': 'Application Programming Interface'
        }
        
        for abbr, full in abbreviations.items():
            text = text.replace(abbr, full)
        
        # Handle numbers
        text = re.sub(r'\b(\d+)\b', lambda m: self._number_to_words(int(m.group(1))), text)
        
        # Add natural pauses
        text = re.sub(r'([.!?])\s*([A-Z])', r'\1 ... \2', text)
        
        return text.strip()
    
    def _number_to_words(self, num):
        """Convert numbers to words for better TTS"""
        if num == 0: return "zero"
        if num < 20:
            words = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
                    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
                    "seventeen", "eighteen", "nineteen"]
            return words[num]
        elif num < 100:
            tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
            return tens[num // 10] + ("" if num % 10 == 0 else " " + self._number_to_words(num % 10))
        else:
            return str(num)  # Fallback for larger numbers
    
    def _numpy_to_audiosegment(self, audio_data, sample_rate=24000):
        """Convert numpy array to AudioSegment with proper formatting"""
        try:
            # Ensure proper data type and range
            if audio_data.dtype != np.int16:
                if audio_data.dtype in [np.float32, np.float64]:
                    audio_data = np.clip(audio_data, -1.0, 1.0)
                    audio_data = (audio_data * 32767).astype(np.int16)
                else:
                    audio_data = audio_data.astype(np.int16)
            
            # Create AudioSegment
            audio_segment = AudioSegment(
                audio_data.tobytes(),
                frame_rate=sample_rate,
                sample_width=audio_data.dtype.itemsize,
                channels=1
            )
            
            return audio_segment
            
        except Exception as e:
            print(f"[TTS] Error converting to AudioSegment: {e}")
            return None
    
    def _apply_emotional_modulation(self, audio_segment, emotion):
        """Apply emotional modulation to audio"""
        try:
            if emotion not in self.emotional_modulation:
                return audio_segment
            
            modulation = self.emotional_modulation[emotion]
            
            # Speed adjustment
            if modulation['speed'] != 1.0:
                audio_segment = audio_segment.speedup(playback_speed=modulation['speed'])
            
            # Volume adjustment
            if modulation['volume'] != 1.0:
                volume_change = 20 * np.log10(modulation['volume'])
                audio_segment = audio_segment + volume_change
            
            # Pitch adjustment (simplified using speed and then length correction)
            if modulation['pitch'] != 1.0:
                # This is a simplified pitch shift - for better results, use specialized libraries
                pitch_factor = modulation['pitch']
                temp_audio = audio_segment.speedup(playback_speed=pitch_factor)
                # Stretch back to original duration
                original_duration = len(audio_segment)
                temp_audio = temp_audio[:original_duration]
            
            return audio_segment
            
        except Exception as e:
            print(f"[TTS] Error in emotional modulation: {e}")
            return audio_segment
    
    def _apply_speaker_style(self, audio_segment, speaker_style):
        """Apply speaker-specific style adjustments"""
        try:
            # Example speaker styles
            styles = {
                'formal': {'volume': 0.95, 'speed': 0.95},
                'casual': {'volume': 1.05, 'speed': 1.05},
                'energetic': {'volume': 1.15, 'speed': 1.1},
                'calm': {'volume': 0.9, 'speed': 0.9}
            }
            
            if speaker_style in styles:
                style = styles[speaker_style]
                
                if 'volume' in style:
                    volume_change = 20 * np.log10(style['volume'])
                    audio_segment = audio_segment + volume_change
                
                if 'speed' in style:
                    audio_segment = audio_segment.speedup(playback_speed=style['speed'])
            
            return audio_segment
            
        except Exception as e:
            print(f"[TTS] Error in speaker style: {e}")
            return audio_segment

# Global enhanced TTS manager
tts_manager = EnhancedTTSManager()

def generate_and_play_enhanced_kokoro(text, language="en", emotion="neutral", speaker_style=None):
    """Enhanced Kokoro TTS generation and playback"""
    global tts_start_time, current_playback
    
    try:
        if not text or not text.strip():
            return
        
        print(f"[Enhanced TTS] Generating: '{text}' (lang={language}, emotion={emotion})")
        
        # Set TTS start time for barge-in timing
        tts_start_time = time.time()
        
        # Generate audio with enhancements
        audio_segment = tts_manager.synthesize_with_emotion(text, language, emotion, speaker_style)
        
        if audio_segment is None:
            print("[Enhanced TTS] Failed to generate audio")
            return
        
        # Normalize audio levels
        audio_segment = audio_segment.normalize(headroom=0.1)
        
        # Add to playback queue
        playback_queue.put(audio_segment)
        
        # Track for echo prevention
        LAST_FEW_BUDDY.append(text)
        
    except Exception as e:
        print(f"[Enhanced TTS] Error: {e}")

# ========== ENHANCED CONVERSATION MANAGEMENT ==========
class ConversationManager:
    def __init__(self):
        self.conversation_state = {
            'topic': None,
            'subtopic': None,
            'context_depth': 0,
            'emotional_tone': 'neutral',
            'conversation_flow': deque(maxlen=20),
            'topic_transitions': deque(maxlen=10),
            'user_engagement_level': 0.5
        }
        self.response_templates = self._load_response_templates()
        self.context_analyzer = ContextAnalyzer()
        self.personality_engine = PersonalityEngine()
        
    def _load_response_templates(self):
        """Load response templates for different conversation types"""
        return {
            'greeting': [
                "Hello {name}! How are you doing today?",
                "Hey there {name}! What's on your mind?",
                "Hi {name}! Good to see you again!"
            ],
            'acknowledgment': [
                "I understand what you mean.",
                "That makes sense to me.",
                "I see your point."
            ],
            'follow_up': [
                "Tell me more about that.",
                "That's interesting. What happened next?",
                "How did that make you feel?"
            ],
            'topic_transition': [
                "Speaking of {previous_topic}, have you thought about {new_topic}?",
                "That reminds me of something about {new_topic}.",
                "On a related note, what do you think about {new_topic}?"
            ],
            'empathy': [
                "That must have been {emotion} for you.",
                "I can imagine how {emotion} you felt.",
                "It sounds like you were feeling {emotion}."
            ]
        }
    
    def analyze_conversation_context(self, user_input, user_name, conversation_history):
        """Analyze current conversation context"""
        context = {
            'current_topic': self._extract_main_topic(user_input),
            'emotional_state': self._analyze_emotional_state(user_input, conversation_history),
            'conversation_type': self._classify_conversation_type(user_input),
            'engagement_indicators': self._assess_engagement(user_input, conversation_history),
            'response_strategy': None
        }
        
        # Determine response strategy
        context['response_strategy'] = self._determine_response_strategy(context, conversation_history)
        
        # Update conversation state
        self._update_conversation_state(context, user_input)
        
        return context
    
    def _extract_main_topic(self, text):
        """Extract main topic from user input"""
        # Enhanced topic extraction using keywords and patterns
        topic_keywords = {
            'work': ['work', 'job', 'office', 'boss', 'colleague', 'project', 'meeting', 'deadline'],
            'family': ['family', 'parent', 'mom', 'dad', 'sister', 'brother', 'child', 'kids'],
            'health': ['health', 'doctor', 'sick', 'medicine', 'hospital', 'pain', 'tired'],
            'technology': ['computer', 'phone', 'app', 'software', 'AI', 'internet', 'tech'],
            'entertainment': ['movie', 'music', 'book', 'game', 'show', 'netflix', 'youtube'],
            'weather': ['weather', 'rain', 'sun', 'snow', 'cold', 'hot', 'temperature'],
            'food': ['food', 'eat', 'restaurant', 'cook', 'dinner', 'lunch', 'hungry'],
            'travel': ['travel', 'trip', 'vacation', 'flight', 'hotel', 'visit', 'journey'],
            'relationships': ['friend', 'relationship', 'dating', 'love', 'partner', 'boyfriend', 'girlfriend'],
            'education': ['school', 'study', 'learn', 'course', 'exam', 'university', 'homework']
        }
        
        text_lower = text.lower()
        topic_scores = {}
        
        for topic, keywords in topic_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                topic_scores[topic] = score
        
        if topic_scores:
            return max(topic_scores, key=topic_scores.get)
        
        return 'general'
    
    def _analyze_emotional_state(self, text, history):
        """Analyze emotional state from text and history"""
        emotions = memory_manager._detect_emotion(text)
        
        # Consider emotional trajectory from history
        if len(history) > 0:
            recent_emotions = []
            for interaction in history[-3:]:  # Last 3 interactions
                if 'emotion' in interaction:
                    recent_emotions.extend(interaction['emotion'])
            
            # Weighted combination of current and recent emotions
            if recent_emotions:
                emotion_counts = {}
                for emotion in emotions + recent_emotions:
                    emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                
                primary_emotion = max(emotion_counts, key=emotion_counts.get)
                return primary_emotion
        
        return emotions[0] if emotions else 'neutral'
    
    def _classify_conversation_type(self, text):
        """Classify the type of conversation"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['?', 'what', 'how', 'why', 'when', 'where']):
            return 'question'
        elif any(word in text_lower for word in ['please', 'can you', 'could you', 'would you']):
            return 'request'
        elif any(word in text_lower for word in ['problem', 'issue', 'wrong', 'broken', 'help']):
            return 'problem_solving'
        elif any(word in text_lower for word in ['feel', 'think', 'believe', 'opinion']):
            return 'opinion_sharing'
        elif any(word in text_lower for word in ['tell', 'story', 'happened', 'experience']):
            return 'storytelling'
        else:
            return 'casual_chat'
    
    def _assess_engagement(self, text, history):
        """Assess user engagement level"""
        engagement_indicators = {
            'length': len(text.split()) > 5,  # Longer responses indicate engagement
            'questions': '?' in text,  # Questions show interest
            'details': any(word in text.lower() for word in ['because', 'since', 'actually', 'specifically']),
            'emotional_words': any(word in text.lower() for word in ['amazing', 'terrible', 'excited', 'worried']),
            'follow_up': len(history) > 0 and any(word in text.lower() for word in ['also', 'and', 'plus', 'additionally'])
        }
        
        engagement_score = sum(engagement_indicators.values()) / len(engagement_indicators)
        return engagement_score
    
    def _determine_response_strategy(self, context, history):
        """Determine the best response strategy"""
        topic = context['current_topic']
        emotion = context['emotional_state']
        conv_type = context['conversation_type']
        engagement = context['engagement_indicators']
        
        # Strategy selection logic
        if emotion in ['sad', 'angry', 'worried']:
            return 'empathetic'
        elif conv_type == 'question':
            return 'informative'
        elif conv_type == 'problem_solving':
            return 'helpful'
        elif engagement > 0.7:
            return 'engaging'
        elif len(history) == 0:
            return 'welcoming'
        else:
            return 'conversational'
    
    def _update_conversation_state(self, context, user_input):
        """Update internal conversation state"""
        self.conversation_state['topic'] = context['current_topic']
        self.conversation_state['emotional_tone'] = context['emotional_state']
        self.conversation_state['user_engagement_level'] = context['engagement_indicators']
        
        # Track topic transitions
        if (self.conversation_state['topic'] and 
            context['current_topic'] != self.conversation_state['topic']):
            self.conversation_state['topic_transitions'].append({
                'from': self.conversation_state['topic'],
                'to': context['current_topic'],
                'timestamp': time.time()
            })
        
        # Update conversation flow
        self.conversation_state['conversation_flow'].append({
            'user_input': user_input,
            'context': context,
            'timestamp': time.time()
        })
    
    def generate_contextual_response(self, user_input, user_name, conversation_history, context):
        """Generate a contextual response based on analysis"""
        strategy = context['response_strategy']
        topic = context['current_topic']
        emotion = context['emotional_state']
        
        # Get user context from memory
        user_context = memory_manager.get_user_context(user_name)
        
        # Build response based on strategy
        if strategy == 'empathetic':
            response = self._generate_empathetic_response(user_input, emotion, user_context)
        elif strategy == 'informative':
            response = self._generate_informative_response(user_input, topic, user_context)
        elif strategy == 'helpful':
            response = self._generate_helpful_response(user_input, topic, user_context)
        elif strategy == 'engaging':
            response = self._generate_engaging_response(user_input, topic, user_context)
        elif strategy == 'welcoming':
            response = self._generate_welcoming_response(user_name, user_context)
        else:
            response = self._generate_conversational_response(user_input, topic, user_context)
        
        # Add personality flavor
        response = self.personality_engine.add_personality_flavor(response, context)
        
        return response
    
    def _generate_empathetic_response(self, user_input, emotion, user_context):
        """Generate an empathetic response"""
        empathy_starters = [
            f"I can hear that you're feeling {emotion}.",
            f"It sounds like this is really {emotion} for you.",
            f"I understand why you'd feel {emotion} about this."
        ]
        
        follow_ups = [
            "Would you like to talk more about what's going on?",
            "Sometimes it helps to share what you're thinking.",
            "I'm here to listen if you want to continue."
        ]
        
        starter = random.choice(empathy_starters)
        follow_up = random.choice(follow_ups)
        
        return f"{starter} {follow_up}"
    
    def _generate_informative_response(self, user_input, topic, user_context):
        """Generate an informative response"""
        # This would typically involve knowledge base lookup or API calls
        info_responses = {
            'weather': "Let me help you with weather information. What location are you interested in?",
            'technology': "That's an interesting tech question. Let me share what I know about that.",
            'health': "Health is really important. While I can share general information, please consult healthcare professionals for medical advice.",
            'work': "Work-related questions can be complex. What specific aspect would you like to explore?",
            'general': "That's a great question. Let me think about the best way to help you with that."
        }
        
        return info_responses.get(topic, info_responses['general'])
    
    def _generate_helpful_response(self, user_input, topic, user_context):
        """Generate a helpful problem-solving response"""
        helper_responses = [
            "Let's work through this step by step. What's the main challenge you're facing?",
            "I'd like to help you figure this out. Can you tell me more about the situation?",
            "Problems can feel overwhelming, but we can break this down. What's happening exactly?"
        ]
        
        return random.choice(helper_responses)
    
    def _generate_engaging_response(self, user_input, topic, user_context):
        """Generate an engaging response to maintain conversation flow"""
        engaging_responses = [
            "That's really interesting! Tell me more about that.",
            "I love hearing about this kind of thing. What's your take on it?",
            "You've got me curious now. What happened next?",
            "That reminds me of something fascinating. Have you ever considered...?"
        ]
        
        return random.choice(engaging_responses)
    
    def _generate_welcoming_response(self, user_name, user_context):
        """Generate a welcoming response for new conversations"""
        welcome_responses = [
            f"Hello {user_name}! It's great to chat with you. What's on your mind today?",
            f"Hi there {user_name}! How are things going for you?",
            f"Hey {user_name}! Good to see you. What would you like to talk about?"
        ]
        
        return random.choice(welcome_responses)
    
    def _generate_conversational_response(self, user_input, topic, user_context):
        """Generate a general conversational response"""
        conversational_responses = [
            "That's interesting. I'd love to hear more about your thoughts on that.",
            "I see what you mean. What's your experience been like with that?",
            "That's a good point. How do you usually handle situations like this?",
            "I appreciate you sharing that. What do you think about it?"
        ]
        
        return random.choice(conversational_responses)

class ContextAnalyzer:
    """Analyzes conversation context for better responses"""
    
    def __init__(self):
        self.context_patterns = {}
        self.conversation_threads = {}
    
    def analyze_context_depth(self, conversation_history, current_topic):
        """Analyze how deep the conversation has gone into a topic"""
        topic_mentions = 0
        for interaction in conversation_history[-10:]:  # Last 10 interactions
            if current_topic in interaction.get('topics', []):
                topic_mentions += 1
        
        return min(topic_mentions / 3.0, 1.0)  # Normalize to 0-1 scale

class PersonalityEngine:
    """Manages Buddy's personality and response style"""
    
    def __init__(self):
        self.personality_traits = self._load_personality_traits()
        self.mood_modifiers = {
            'cheerful': ['!', 'That sounds great', 'Awesome'],
            'sassy': ['Well,', 'Oh really?', 'Interesting choice'],
            'thoughtful': ['Hmm,', 'Let me think about that', 'That\'s worth considering'],
            'energetic': ['Wow!', 'That\'s exciting!', 'Amazing!']
        }
    
    def _load_personality_traits(self):
        """Load personality configuration"""
        default_traits = {
            'curiosity': 0.8,
            'empathy': 0.9,
            'humor': 0.6,
            'helpfulness': 0.95,
            'assertiveness': 0.5,
            'optimism': 0.7
        }
        
        if os.path.exists(PERSONALITY_TRAITS_PATH):
            try:
                with open(PERSONALITY_TRAITS_PATH, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        
        return default_traits
    
    def add_personality_flavor(self, response, context):
        """Add personality-based modifications to response"""
        traits = self.personality_traits
        
        # Add curiosity-based follow-ups
        if traits.get('curiosity', 0) > 0.7 and context['conversation_type'] != 'question':
            if random.random() < 0.3:  # 30% chance
                curiosity_additions = [
                    " What do you think about that?",
                    " I'm curious about your perspective on this.",
                    " How does that make you feel?"
                ]
                response += random.choice(curiosity_additions)
        
        # Add empathy
        if traits.get('empathy', 0) > 0.8 and context['emotional_state'] != 'neutral':
            empathy_phrases = [
                "I understand ",
                "I can imagine ",
                "That makes sense "
            ]
            if not any(phrase in response for phrase in empathy_phrases):
                if random.random() < 0.4:  # 40% chance
                    response = random.choice(empathy_phrases) + "that " + response.lower()
        
        # Add humor (carefully)
        if traits.get('humor', 0) > 0.6 and context['emotional_state'] in ['happy', 'neutral']:
            if random.random() < 0.2:  # 20% chance
                humor_additions = [
                    " (That's my digital intuition speaking!)",
                    " At least, that's what my circuits are telling me!",
                    " Though I might be a bit biased as an AI!"
                ]
                response += random.choice(humor_additions)
        
        return response

# Global conversation manager
conversation_manager = ConversationManager()

# ========== ENHANCED LISTENING AND TRANSCRIPTION ==========
def enhanced_vad_and_listen():
    """Enhanced VAD with better noise handling and speaker adaptation"""
    blocksize = int(MIC_SAMPLE_RATE * 0.02)  # 20ms blocks
    min_speech_frames = 8  # Reduced for faster response
    silence_threshold = SILENCE_TIMEOUT
    max_recording_time = 12  # Maximum recording duration
    
    try:
        with sd.InputStream(device=MIC_DEVICE_INDEX, samplerate=MIC_SAMPLE_RATE, 
                           channels=1, blocksize=blocksize, dtype='int16') as stream:
            
            print("\n[Enhanced Buddy] === LISTENING === ")
            
            frame_buffer = []
            speech_detected = 0
            noise_floor_samples = []
            
            # Calibrate noise floor
            print("[Enhanced Buddy] Calibrating noise floor...")
            for _ in range(25):  # 0.5 seconds of calibration
                frame, _ = stream.read(blocksize)
                mic_audio = frame.flatten()
                processed_audio = apply_enhanced_aec(mic_audio.tobytes())
                noise_floor_samples.extend(processed_audio)
            
            noise_floor = np.percentile(np.abs(noise_floor_samples), 75)
            adaptive_threshold = max(noise_floor * 3, 200)
            
            print(f"[Enhanced Buddy] Noise floor: {noise_floor:.1f}, Threshold: {adaptive_threshold:.1f}")
            
            while True:
                frame, _ = stream.read(blocksize)
                mic_audio = frame.flatten()
                
                # Apply enhanced AEC
                processed_audio = apply_enhanced_aec(mic_audio.tobytes())
                
                # Downsample for VAD
                audio_16k = downsample_enhanced(processed_audio, MIC_SAMPLE_RATE, 16000)
                
                # Enhanced voice activity detection
                is_voice = is_voice_activity(audio_16k, threshold_multiplier=1.2)
                is_speech_enhanced = enhanced_vad.is_speech_enhanced(audio_16k)
                
                # Combined decision
                speech_detected_now = is_voice and is_speech_enhanced
                
                if speech_detected_now:
                    frame_buffer.append(audio_16k)
                    speech_detected += 1
                    
                    if speech_detected >= min_speech_frames:
                        print("[Enhanced Buddy] Speech detected - Recording...")
                        
                        # Start recording
                        audio_recording = frame_buffer.copy()
                        last_speech_time = time.time()
                        recording_start = time.time()
                        
                        frame_buffer.clear()
                        
                        # Continue recording until silence or timeout
                        while (time.time() - last_speech_time < silence_threshold and 
                               time.time() - recording_start < max_recording_time):
                            
                            frame, _ = stream.read(blocksize)
                            mic_audio = frame.flatten()
                            
                            # Apply AEC
                            processed_audio = apply_enhanced_aec(mic_audio.tobytes())
                            audio_16k = downsample_enhanced(processed_audio, MIC_SAMPLE_RATE, 16000)
                            
                            audio_recording.append(audio_16k)
                            
                            # Check for continued speech
                            if (is_voice_activity(audio_16k, threshold_multiplier=0.8) and 
                                enhanced_vad.is_speech_enhanced(audio_16k)):
                                last_speech_time = time.time()
                        
                        print("[Enhanced Buddy] Recording complete - Processing...")
                        
                        # Concatenate and return
                        final_audio = np.concatenate(audio_recording, axis=0).astype(np.int16)
                        
                        # Apply final noise reduction
                        try:
                            final_audio_float = final_audio.astype(np.float32) / 32768.0
                            final_audio_clean = nr.reduce_noise(y=final_audio_float, sr=16000)
                            final_audio = (final_audio_clean * 32767).astype(np.int16)
                        except:
                            pass  # Fallback to original if noise reduction fails
                        
                        return final_audio
                
                else:
                    if len(frame_buffer) > 0:
                        frame_buffer.clear()
                    speech_detected = max(0, speech_detected - 1)
    
    except Exception as e:
        print(f"[Enhanced Buddy] Listening error: {e}")
        return np.array([], dtype=np.int16)

def enhanced_listen_and_transcribe(conversation_history):
    """Enhanced listening with better echo prevention and transcription"""
    
    # Wait for any ongoing playback to finish
    while buddy_talking.is_set():
        time.sleep(0.05)
    
    # Additional settling time
    time.sleep(0.2)
    
    # Clear any residual audio in reference buffer
    with ref_audio_lock:
        ref_audio_buffer[:] = np.zeros_like(ref_audio_buffer)
    
    # Enhanced listening
    audio = enhanced_vad_and_listen()
    
    if len(audio) < 1600:  # Less than 0.1 seconds at 16kHz
        print("[Enhanced Buddy] Audio too short, ignoring")
        return "..."
    
    # Save for debugging
    try:
        write("temp_enhanced_input.wav", 16000, audio)
        if DEBUG_MODE:
            print(f"[Enhanced Buddy] Saved audio: {len(audio)} samples, "
                  f"RMS: {np.sqrt(np.mean(audio.astype(np.float32)**2)):.1f}")
    except Exception as e:
        if DEBUG:
            print(f"[Enhanced Buddy] Error saving audio: {e}")
    
    # Enhanced transcription
    text = stt_stream(audio).strip()
    
    if not text:
        return "..."
    
    # Enhanced filtering
    if enhanced_echo_detection(text, conversation_history):
        print(f"[Enhanced Buddy] Filtered echo: {text}")
        return "..."
    
    if enhanced_noise_detection(text):
        print(f"[Enhanced Buddy] Filtered noise: {text}")
        return "..."
    
    print(f"[Enhanced Buddy] === USER SAID: \"{text}\" ===")
    return text

def enhanced_echo_detection(text, conversation_history):
    """Enhanced echo detection with multiple criteria"""
    if not text or len(text.strip()) < 3:
        return True
    
    cleaned_text = re.sub(r'[^\w\s]', '', text.strip().lower())
    
    # Check against recent Buddy responses
    for buddy_response in list(LAST_FEW_BUDDY)[-4:]:
        buddy_cleaned = re.sub(r'[^\w\s]', '', buddy_response.strip().lower())
        
        # Exact match
        if cleaned_text == buddy_cleaned:
            return True
        
        # Subsequence match
        if cleaned_text in buddy_cleaned or buddy_cleaned in cleaned_text:
            return True
        
        # Fuzzy similarity
        similarity = difflib.SequenceMatcher(None, cleaned_text, buddy_cleaned).ratio()
        if similarity > 0.85:
            return True
    
    # Check against conversation history
    if conversation_history:
        recent_responses = [h.get('buddy', '') for h in conversation_history[-3:]]
        for response in recent_responses:
            response_cleaned = re.sub(r'[^\w\s]', '', response.strip().lower())
            if cleaned_text in response_cleaned:
                return True
    
    return False

def enhanced_noise_detection(text):
    """Enhanced noise and meaningless input detection"""
    if not text or len(text.strip()) < 2:
        return True
    
    # Remove punctuation for analysis
    cleaned = re.sub(r'[^\w\s]', '', text.strip().lower())
    words = cleaned.split()
    
    # Too short
    if len(words) == 1 and len(cleaned) < 4:
        return True
    
    # Known noise patterns
    noise_patterns = [
        r'^(uh+|um+|er+|ah+|mm+|hm+)$',
        r'^(yeah|yep|ok|okay|sure|right|yes|no)$',
        r'^(what|huh|eh)$'
    ]
    
    for pattern in noise_patterns:
        if re.match(pattern, cleaned):
            return True
    
    # Repetitive characters (like "aaaa" or "hmmm")
    if len(set(cleaned)) < 3 and len(cleaned) > 4:
        return True
    
    # Very low information content
    if len(words) == 1 and words[0] in ['the', 'and', 'but', 'for', 'are', 'you']:
        return True
    
    return False
# ========== API INTEGRATIONS AND EXTERNAL SERVICES ==========
class APIManager:
    def __init__(self):
        self.weather_api_key = WEATHERAPI_KEY
        self.serpapi_key = SERPAPI_KEY
        self.home_assistant_url = HOME_ASSISTANT_URL
        self.home_assistant_token = HOME_ASSISTANT_TOKEN
        self.api_cache = {}
        self.cache_timeout = 300  # 5 minutes
        
    def get_weather_info(self, location="auto"):
        """Get weather information with caching"""
        cache_key = f"weather_{location}"
        
        # Check cache first
        if cache_key in self.api_cache:
            cached_data, timestamp = self.api_cache[cache_key]
            if time.time() - timestamp < self.cache_timeout:
                return cached_data
        
        try:
            if not self.weather_api_key:
                return "Weather API key not configured."
            
            url = f"http://api.weatherapi.com/v1/current.json"
            params = {
                'key': self.weather_api_key,
                'q': location,
                'aqi': 'no'
            }
            
            response = requests.get(url, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()
                weather_info = {
                    'location': data['location']['name'],
                    'temperature': data['current']['temp_c'],
                    'condition': data['current']['condition']['text'],
                    'humidity': data['current']['humidity'],
                    'wind_speed': data['current']['wind_kph']
                }
                
                # Cache the result
                self.api_cache[cache_key] = (weather_info, time.time())
                
                return (f"The weather in {weather_info['location']} is "
                       f"{weather_info['temperature']}°C with {weather_info['condition']}. "
                       f"Humidity is {weather_info['humidity']}% and wind speed is "
                       f"{weather_info['wind_speed']} km/h.")
            else:
                return "Sorry, I couldn't get the weather information right now."
                
        except Exception as e:
            print(f"[Weather API] Error: {e}")
            return "I'm having trouble accessing weather data at the moment."
    
    def search_web(self, query, num_results=3):
        """Search the web using SerpAPI"""
        cache_key = f"search_{query}"
        
        # Check cache
        if cache_key in self.api_cache:
            cached_data, timestamp = self.api_cache[cache_key]
            if time.time() - timestamp < self.cache_timeout:
                return cached_data
        
        try:
            if not self.serpapi_key:
                return "Web search API key not configured."
            
            params = {
                'engine': 'google',
                'q': query,
                'api_key': self.serpapi_key,
                'num': num_results
            }
            
            response = requests.get(SERPAPI_ENDPOINT, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                
                if 'organic_results' in data:
                    results = []
                    for result in data['organic_results'][:num_results]:
                        results.append({
                            'title': result.get('title', ''),
                            'snippet': result.get('snippet', ''),
                            'link': result.get('link', '')
                        })
                    
                    # Cache results
                    self.api_cache[cache_key] = (results, time.time())
                    
                    # Format response
                    response_text = f"Here's what I found about '{query}':\n\n"
                    for i, result in enumerate(results, 1):
                        response_text += f"{i}. {result['title']}\n{result['snippet']}\n\n"
                    
                    return response_text
                else:
                    return "I couldn't find any relevant search results."
            else:
                return "Sorry, I'm having trouble with web search right now."
                
        except Exception as e:
            print(f"[Search API] Error: {e}")
            return "I encountered an error while searching the web."
    
    def control_smart_home(self, action, device=None):
        """Control smart home devices via Home Assistant"""
        try:
            if not self.home_assistant_url or not self.home_assistant_token:
                return "Smart home control is not configured."
            
            headers = {
                'Authorization': f'Bearer {self.home_assistant_token}',
                'Content-Type': 'application/json'
            }
            
            # Simple command mapping
            commands = {
                'lights on': {'domain': 'light', 'service': 'turn_on'},
                'lights off': {'domain': 'light', 'service': 'turn_off'},
                'music play': {'domain': 'media_player', 'service': 'media_play'},
                'music stop': {'domain': 'media_player', 'service': 'media_stop'}
            }
            
            command_key = f"{action} {device}".strip().lower()
            if command_key in commands:
                cmd = commands[command_key]
                url = f"{self.home_assistant_url}/api/services/{cmd['domain']}/{cmd['service']}"
                
                payload = {}
                if device:
                    payload['entity_id'] = f"{cmd['domain']}.{device}"
                
                response = requests.post(url, headers=headers, json=payload, timeout=5)
                if response.status_code == 200:
                    return f"Successfully executed: {action} {device or ''}"
                else:
                    return "I couldn't execute that smart home command."
            else:
                return "I don't recognize that smart home command."
                
        except Exception as e:
            print(f"[Smart Home] Error: {e}")
            return "I'm having trouble controlling smart home devices."

# Global API manager
api_manager = APIManager()

# ========== ENHANCED MAIN INTERACTION SYSTEM ==========
class BuddySystem:
    def __init__(self):
        self.is_running = False
        self.current_user = None
        self.conversation_history = []
        self.session_start_time = time.time()
        self.interaction_count = 0
        self.last_interaction_time = 0
        self.system_health = {
            'audio_ok': True,
            'tts_ok': True,
            'memory_ok': True,
            'api_ok': True
        }
        
        # Start worker threads
        self._start_worker_threads()
        
        # Load user session
        self._initialize_session()
    
    def _start_worker_threads(self):
        """Start all background worker threads"""
        try:
            # TTS worker
            self.tts_thread = threading.Thread(target=tts_worker, daemon=True)
            self.tts_thread.start()
            
            # Audio playback worker
            self.playback_thread = threading.Thread(target=audio_playback_worker, daemon=True)
            self.playback_thread.start()
            
            # Memory auto-save worker
            self.memory_thread = threading.Thread(target=self._memory_autosave_worker, daemon=True)
            self.memory_thread.start()
            
            print("[Buddy System] All worker threads started successfully")
            
        except Exception as e:
            print(f"[Buddy System] Error starting worker threads: {e}")
            self.system_health['audio_ok'] = False
    
    def _memory_autosave_worker(self):
        """Periodically save memory to disk"""
        while True:
            try:
                time.sleep(60)  # Save every minute
                memory_manager.save_memories_to_disk()
                if DEBUG_MODE:
                    print("[Memory] Auto-saved memories to disk")
            except Exception as e:
                print(f"[Memory] Auto-save error: {e}")
                self.system_health['memory_ok'] = False
    
    def _initialize_session(self):
        """Initialize user session"""
        try:
            # Identify or register user
            self.current_user = self.identify_or_register_user()
            
            # Load conversation history
            self.conversation_history = self.load_user_conversation_history(self.current_user)
            
            # Welcome message
            self._send_welcome_message()
            
            print(f"[Buddy System] Session initialized for user: {self.current_user}")
            
        except Exception as e:
            print(f"[Buddy System] Session initialization error: {e}")
            self.current_user = "Guest"
            self.conversation_history = []
    
    def identify_or_register_user(self):
        """Enhanced user identification with voice recognition"""
        if FAST_MODE:
            # Fast mode - use last known user or default
            last_user = get_last_user()
            if last_user:
                speak_async(f"Welcome back, {last_user}! Ready to chat?", emotion="cheerful")
                return last_user
            else:
                speak_async("Hello! I'm Buddy, your AI assistant. What's your name?", emotion="friendly")
        else:
            # Full identification mode
            speak_async("Hello! I'm your enhanced AI assistant Buddy. Let me identify you by your voice.", 
                       emotion="welcoming")
        
        # Wait for playback to finish
        playback_queue.join()
        
        # Listen for identification
        identification_attempts = 0
        max_attempts = 3
        
        while identification_attempts < max_attempts:
            try:
                audio = enhanced_vad_and_listen()
                if len(audio) > 1600:  # Minimum audio length
                    
                    # Try speaker identification first
                    speaker_name, confidence = detect_and_identify_speaker(audio)
                    
                    if speaker_name and confidence > 0.85:
                        speak_async(f"Welcome back, {speaker_name}! Great to see you again.", emotion="happy")
                        set_last_user(speaker_name)
                        return speaker_name
                    
                    # Fallback to name recognition
                    text = stt_stream(audio).strip()
                    if text and not enhanced_noise_detection(text):
                        # Extract name from speech
                        name = self._extract_name_from_speech(text)
                        if name:
                            # Register new user or update existing
                            if name not in known_users:
                                speaker_system.register_speaker(name, audio)
                                known_users[name] = generate_embedding(name).tolist()
                                with open(known_users_path, "w", encoding="utf-8") as f:
                                    json.dump(known_users, f, indent=2, ensure_ascii=False)
                                speak_async(f"Nice to meet you, {name}! I'll remember your voice.", emotion="friendly")
                            else:
                                speak_async(f"Good to see you again, {name}!", emotion="cheerful")
                            
                            set_last_user(name)
                            return name
                
                identification_attempts += 1
                if identification_attempts < max_attempts:
                    speak_async("I didn't catch that. Could you please say your name clearly?", emotion="patient")
                    playback_queue.join()
                
            except Exception as e:
                print(f"[Identification] Error: {e}")
                identification_attempts += 1
        
        # Fallback to guest mode
        guest_name = f"Guest_{int(time.time() % 10000)}"
        speak_async(f"I'll call you {guest_name} for now. We can chat anytime!", emotion="friendly")
        return guest_name
    
    def _extract_name_from_speech(self, text):
        """Extract name from speech patterns"""
        text_lower = text.lower()
        
        # Common name patterns
        name_patterns = [
            r"(?:my name is|i'm|i am|call me|this is)\s+([a-zA-Z]+)",
            r"^([a-zA-Z]+)$",  # Single word responses
            r"([a-zA-Z]+)(?:\s+here|speaking)",
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, text_lower)
            if match:
                name = match.group(1).title()
                # Validate name (basic check)
                if len(name) >= 2 and name.isalpha():
                    return name
        
        return None
    
    def load_user_conversation_history(self, user_name):
        """Load conversation history for user"""
        try:
            history_path = f"conversation_history_{user_name}.json"
            if os.path.exists(history_path):
                with open(history_path, "r", encoding="utf-8") as f:
                    history = json.load(f)
                    return history[-50:]  # Keep last 50 interactions
            return []
        except Exception as e:
            print(f"[History] Error loading for {user_name}: {e}")
            return []
    
    def save_user_conversation_history(self, user_name, history):
        """Save conversation history for user"""
        try:
            history_path = f"conversation_history_{user_name}.json"
            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(history[-100:], f, indent=2, ensure_ascii=False)  # Keep last 100
        except Exception as e:
            print(f"[History] Error saving for {user_name}: {e}")
    
    def _send_welcome_message(self):
        """Send contextual welcome message"""
        try:
            current_time = time.time()
            hour = time.localtime(current_time).tm_hour
            
            # Time-based greeting
            if 5 <= hour < 12:
                time_greeting = "Good morning"
                emotion = "energetic"
            elif 12 <= hour < 17:
                time_greeting = "Good afternoon"
                emotion = "cheerful"
            elif 17 <= hour < 22:
                time_greeting = "Good evening"
                emotion = "warm"
            else:
                time_greeting = "Hello"
                emotion = "calm"
            
            # Check if returning user
            if self.conversation_history:
                last_interaction = self.conversation_history[-1]
                last_time = last_interaction.get('timestamp', 0)
                time_since_last = current_time - last_time
                
                if time_since_last < 3600:  # Less than 1 hour
                    message = f"{time_greeting}, {self.current_user}! Welcome back. What's on your mind?"
                elif time_since_last < 86400:  # Less than 1 day
                    message = f"{time_greeting}, {self.current_user}! Good to see you again today."
                else:
                    message = f"{time_greeting}, {self.current_user}! It's been a while. How have you been?"
            else:
                message = f"{time_greeting}, {self.current_user}! I'm excited to chat with you. What would you like to talk about?"
            
            speak_async(message, emotion=emotion)
            playback_queue.join()
            
        except Exception as e:
            print(f"[Welcome] Error: {e}")
            speak_async(f"Hello {self.current_user}! Ready to chat?", emotion="friendly")
    
    def process_user_input(self, user_input):
        """Process user input and generate response"""
        try:
            # Sanitize input
            user_input = sanitize_user_prompt(user_input)
            
            # Analyze conversation context
            context = conversation_manager.analyze_conversation_context(
                user_input, self.current_user, self.conversation_history
            )
            
            # Check for special commands
            special_response = self._handle_special_commands(user_input, context)
            if special_response:
                return special_response, context
            
            # Check for API requests
            api_response = self._handle_api_requests(user_input)
            if api_response:
                return api_response, context
            
            # Generate contextual response
            response = conversation_manager.generate_contextual_response(
                user_input, self.current_user, self.conversation_history, context
            )
            
            # Add dynamic elements
            response = self._enhance_response(response, context)
            
            return response, context
            
        except Exception as e:
            print(f"[Processing] Error: {e}")
            return "I'm having a bit of trouble processing that. Could you try again?", {}
    
    def _handle_special_commands(self, user_input, context):
        """Handle special system commands"""
        user_input_lower = user_input.lower()
        
        # System commands
        if any(phrase in user_input_lower for phrase in ['system status', 'how are you doing', 'system health']):
            health_report = []
            for component, status in self.system_health.items():
                health_report.append(f"{component}: {'✓' if status else '✗'}")
            
            uptime = time.time() - self.session_start_time
            uptime_str = f"{int(uptime // 3600)}h {int((uptime % 3600) // 60)}m"
            
            return f"System status: {', '.join(health_report)}. Uptime: {uptime_str}. " \
                   f"We've had {self.interaction_count} interactions this session."
        
        # Memory commands
        if 'remember that' in user_input_lower:
            # Extract information to remember
            info = user_input_lower.replace('remember that', '').strip()
            if info:
                # Store in user memory
                user_memory = load_user_memory(self.current_user)
                if 'notes' not in user_memory:
                    user_memory['notes'] = []
                user_memory['notes'].append({
                    'content': info,
                    'timestamp': time.time()
                })
                save_user_memory(self.current_user, user_memory)
                return "I'll remember that for you!"
        
        # Mood commands
        mood_command = detect_mood_command(user_input)
        if mood_command:
            session_emotion_mode[self.current_user] = mood_command
            return f"Switching to {mood_command} mode! How's this?"
        
        # Intent-based reactions
        intent = detect_user_intent(user_input)
        intent_response = handle_intent_reaction(intent)
        if intent_response:
            return intent_response
        
        return None
    
    def _handle_api_requests(self, user_input):
        """Handle API-based requests"""
        user_input_lower = user_input.lower()
        
        # Weather requests
        weather_triggers = ['weather', 'temperature', 'forecast', 'how hot', 'how cold', 'raining']
        if any(trigger in user_input_lower for trigger in weather_triggers):
            # Extract location if mentioned
            location_match = re.search(r'in ([a-zA-Z\s]+)', user_input_lower)
            location = location_match.group(1).strip() if location_match else "auto"
            return api_manager.get_weather_info(location)
        
        # Search requests
        search_triggers = ['search for', 'look up', 'find information about', 'what is', 'tell me about']
        for trigger in search_triggers:
            if trigger in user_input_lower:
                query = user_input_lower.replace(trigger, '').strip()
                if query:
                    return api_manager.search_web(query)
        
        # Smart home requests
        home_triggers = ['turn on', 'turn off', 'play music', 'stop music', 'lights']
        if any(trigger in user_input_lower for trigger in home_triggers):
            # Simple command parsing
            if 'lights on' in user_input_lower:
                return api_manager.control_smart_home('lights', 'on')
            elif 'lights off' in user_input_lower:
                return api_manager.control_smart_home('lights', 'off')
            elif 'play music' in user_input_lower:
                return api_manager.control_smart_home('music', 'play')
            elif 'stop music' in user_input_lower:
                return api_manager.control_smart_home('music', 'stop')
        
        return None
    
    def _enhance_response(self, response, context):
        """Add dynamic enhancements to response"""
        try:
            # Add user-specific context
            user_context = memory_manager.get_user_context(self.current_user)
            
            # Add emotional context if relevant
            if context.get('emotional_state') != 'neutral':
                emotion = context['emotional_state']
                if emotion in ['sad', 'angry', 'worried'] and 'I understand' not in response:
                    response = f"I can sense you're feeling {emotion}. " + response
            
            # Add conversation continuity
            if len(self.conversation_history) > 0:
                last_topic = self.conversation_history[-1].get('topics', ['general'])[0]
                current_topic = context.get('current_topic', 'general')
                
                if last_topic != current_topic and last_topic != 'general':
                    if random.random() < 0.3:  # 30% chance
                        response += f" By the way, how did things go with {last_topic}?"
            
            # Add time-sensitive elements
            current_hour = time.localtime().tm_hour
            if 'tired' in response.lower() and current_hour > 22:
                response += " It is getting pretty late!"
            
            return response
            
        except Exception as e:
            print(f"[Enhancement] Error: {e}")
            return response
    
    def run_conversation_loop(self):
        """Main conversation loop"""
        self.is_running = True
        print(f"\n[Buddy System] Starting enhanced conversation loop for {self.current_user}")
        
        try:
            while self.is_running:
                try:
                    # Listen for user input
                    user_input = enhanced_listen_and_transcribe(self.conversation_history)
                    
                    # Skip empty or filtered input
                    if not user_input or user_input == "...":
                        continue
                    
                    # Check for exit commands
                    if self._check_exit_commands(user_input):
                        break
                    
                    # Process input and generate response
                    response, context = self.process_user_input(user_input)
                    
                    # Store interaction in memory
                    interaction = {
                        'timestamp': time.time(),
                        'user': user_input,
                        'buddy': response,
                        'context': context,
                        'emotion': context.get('emotional_state', 'neutral'),
                        'topics': context.get('current_topic', ['general'])
                    }
                    
                    self.conversation_history.append(interaction)
                    memory_manager.store_interaction(
                        self.current_user, user_input, response, context
                    )
                    
                    # Update user memory
                    update_user_memory(self.current_user, user_input)
                    update_thematic_memory(self.current_user, user_input)
                    
                    # Determine response emotion
                    response_emotion = self._determine_response_emotion(context)
                    
                    # Speak response
                    speak_async(response, emotion=response_emotion)
                    
                    # Update counters
                    self.interaction_count += 1
                    self.last_interaction_time = time.time()
                    
                    # Periodic maintenance
                    if self.interaction_count % 20 == 0:
                        self._perform_maintenance()
                    
                except KeyboardInterrupt:
                    print("\n[Buddy System] Interrupted by user")
                    break
                except Exception as e:
                    print(f"[Buddy System] Loop error: {e}")
                    speak_async("I encountered an error. Let me restart...", emotion="apologetic")
                    time.sleep(1)
                    continue
        
        finally:
            self._cleanup_session()
    
    def _check_exit_commands(self, user_input):
        """Check for conversation exit commands"""
        exit_phrases = [
            'goodbye', 'bye', 'see you later', 'talk to you later', 
            'gotta go', 'exit', 'quit', 'stop', 'end conversation'
        ]
        
        user_input_lower = user_input.lower()
        if any(phrase in user_input_lower for phrase in exit_phrases):
            # Send goodbye message
            goodbyes = [
                f"Goodbye {self.current_user}! It was great talking with you.",
                f"See you later {self.current_user}! Take care!",
                f"Bye {self.current_user}! Looking forward to our next chat."
            ]
            
            speak_async(random.choice(goodbyes), emotion="warm")
            playback_queue.join()
            return True
        
        return False
    
    def _determine_response_emotion(self, context):
        """Determine appropriate emotion for response"""
        user_emotion = context.get('emotional_state', 'neutral')
        conversation_type = context.get('conversation_type', 'casual_chat')
        
        # Map user emotions to response emotions
        emotion_mapping = {
            'happy': 'cheerful',
            'excited': 'energetic',
            'sad': 'empathetic',
            'angry': 'calm',
            'worried': 'reassuring',
            'neutral': 'friendly'
        }
        
        response_emotion = emotion_mapping.get(user_emotion, 'friendly')
        
        # Adjust based on conversation type
        if conversation_type == 'question':
            response_emotion = 'informative'
        elif conversation_type == 'problem_solving':
            response_emotion = 'helpful'
        
        # Check for session emotion mode
        if self.current_user in session_emotion_mode:
            response_emotion = session_emotion_mode[self.current_user]
        
        return response_emotion
    
    def _perform_maintenance(self):
        """Perform periodic system maintenance"""
        try:
            # Save conversation history
            self.save_user_conversation_history(self.current_user, self.conversation_history)
            
            # Clean up old cache entries
            current_time = time.time()
            for cache in [api_manager.api_cache, tts_manager.voice_cache]:
                expired_keys = []
                for key, (data, timestamp) in cache.items():
                    if current_time - timestamp > 1800:  # 30 minutes
                        expired_keys.append(key)
                
                for key in expired_keys:
                    del cache[key]
            
            # Update system health
            self._check_system_health()
            
            if DEBUG_MODE:
                print(f"[Maintenance] Completed. History: {len(self.conversation_history)} items")
            
        except Exception as e:
            print(f"[Maintenance] Error: {e}")
    
    def _check_system_health(self):
        """Check system component health"""
        try:
            # Check audio system
            self.system_health['audio_ok'] = not playback_queue.empty() or not buddy_talking.is_set()
            
            # Check TTS system
            self.system_health['tts_ok'] = tts_manager.kokoro is not None
            
            # Check memory system
            self.system_health['memory_ok'] = len(memory_manager.short_term_memory) >= 0
            
            # Check API system
            self.system_health['api_ok'] = api_manager is not None
            
        except Exception as e:
            print(f"[Health Check] Error: {e}")
            self.system_health = {k: False for k in self.system_health.keys()}
    
    def _cleanup_session(self):
        """Clean up session resources"""
        try:
            print("[Buddy System] Cleaning up session...")
            
            # Stop audio playback
            stop_enhanced_playback()
            
            # Save final conversation history
            self.save_user_conversation_history(self.current_user, self.conversation_history)
            
            # Save memories
            memory_manager.save_memories_to_disk()
            
            # Save known users
            with open(known_users_path, "w", encoding="utf-8") as f:
                json.dump(known_users, f, indent=2, ensure_ascii=False)
            
            # Clear queues
            while not tts_queue.empty():
                try:
                    tts_queue.get_nowait()
                    tts_queue.task_done()
                except queue.Empty:
                    break
            
            print("[Buddy System] Session cleanup completed")
            
        except Exception as e:
            print(f"[Cleanup] Error: {e}")

# ========== CONFIGURATION VALIDATION ==========
def validate_configuration():
    """Validate system configuration before startup"""
    issues = []
    
    # Check required model files
    required_files = ["kokoro-v1.0.onnx", "voices-v1.0.bin"]
    for file in required_files:
        if not os.path.exists(file):
            issues.append(f"Missing required file: {file}")
    
    # Check audio device
    try:
        devices = sd.query_devices()
        if MIC_DEVICE_INDEX >= len(devices):
            issues.append(f"Invalid mic device index: {MIC_DEVICE_INDEX} (max: {len(devices)-1})")
        else:
            device_info = devices[MIC_DEVICE_INDEX]
            if device_info['max_input_channels'] < 1:
                issues.append(f"Device {MIC_DEVICE_INDEX} has no input channels")
    except Exception as e:
        issues.append(f"Audio device check failed: {e}")
    
    # Check Whisper connection
    try:
        async def check_whisper():
            try:
                async with websockets.connect(FASTER_WHISPER_WS, ping_timeout=3) as ws:
                    return True
            except:
                return False
        
        if not asyncio.run(check_whisper()):
            issues.append("Cannot connect to Whisper server - make sure faster-whisper-server is running on localhost:9090")
    except Exception as e:
        issues.append(f"Whisper connection check failed: {e}")
    
    # Check write permissions
    test_files = ["test_buddy_permissions.tmp"]
    for test_file in test_files:
        try:
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
        except Exception as e:
            issues.append(f"Cannot write to current directory: {e}")
            break
    
    return issues

def show_system_info():
    """Show system information"""
    print(f"🐍 Python: {sys.version}")
    print(f"💾 Current directory: {os.getcwd()}")
    print(f"🎤 Audio devices available: {len(sd.query_devices())}")
    print(f"🎤 Selected mic device: {MIC_DEVICE_INDEX}")
    
    # Show selected device info
    try:
        device = sd.query_devices()[MIC_DEVICE_INDEX]
        print(f"🎤 Device name: {device['name']}")
        print(f"🎤 Max input channels: {device['max_input_channels']}")
        print(f"🎤 Default sample rate: {device['default_samplerate']}")
    except:
        print("🎤 Could not get device info")
    
    print(f"🌐 Whisper server: {FASTER_WHISPER_WS}")
    print(f"🧠 Fast mode: {FAST_MODE}")
    print(f"🔊 Debug mode: {DEBUG}")
Fix 4: Enhanced Main Function (Replace the existing main() function)
Find the existing main() function and replace it entirely with:

Python

def main():
    try:
        print("=" * 70)
        print("🤖 ENHANCED BUDDY AI ASSISTANT")
        print("=" * 70)
        print(f"📅 Session started: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
        print(f"👤 Current user: Daveydrz")
        print("🔧 Enhanced features: Advanced AEC, VAD, Multi-speaker, Memory AI")
        print("=" * 70)
        
        # Show system information
        show_system_info()
        print("=" * 70)
        
        # Validate configuration
        print("🔍 Validating configuration...")
        config_issues = validate_configuration()
        
        if config_issues:
            print("⚠️  Configuration Issues Found:")
            for i, issue in enumerate(config_issues, 1):
                print(f"   {i}. {issue}")
            print()
            
            # Check for critical issues
            critical_issues = [issue for issue in config_issues if any(critical in issue.lower() 
                             for critical in ['missing required file', 'invalid mic device', 'cannot write'])]
            
            if critical_issues:
                print("❌ Critical issues found. Please fix these before continuing:")
                for issue in critical_issues:
                    print(f"   • {issue}")
                return
            else:
                response = input("⚠️  Non-critical issues found. Continue anyway? (y/N): ")
                if response.lower() != 'y':
                    print("Exiting...")
                    return
        else:
            print("✅ Configuration validation passed!")
        
        print("=" * 70)
        print("🚀 Starting Buddy system...")
        
        # Initialize system components
        print("📊 Loading memory systems...")
        memory_manager.load_memories_from_disk()
        
        print("🎭 Loading speaker profiles...")
        if os.path.exists('speaker_profiles.json'):
            try:
                with open('speaker_profiles.json', 'r', encoding='utf-8') as f:
                    profiles_data = json.load(f)
                    # Convert lists back to numpy arrays
                    for name, profile in profiles_data.items():
                        converted_profile = {}
                        for key, value in profile.items():
                            if isinstance(value, list) and key == 'embedding':
                                converted_profile[key] = np.array(value)
                            else:
                                converted_profile[key] = value
                        speaker_system.speaker_profiles[name] = converted_profile
                print(f"📊 Loaded {len(speaker_system.speaker_profiles)} speaker profiles")
            except Exception as e:
                print(f"⚠️  Error loading speaker profiles: {e}")
        
        # Initialize Buddy system
        print("🤖 Initializing Buddy system...")
        buddy_system = BuddySystem()
        
        # Setup signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            print(f"\n[Signal] Received signal {signum}, shutting down gracefully...")
            buddy_system.is_running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Setup exit handler
        def cleanup_on_exit():
            print("\n[Exit] Performing final cleanup...")
            try:
                # Save speaker profiles
                if speaker_system.speaker_profiles:
                    profiles_to_save = {}
                    for name, profile in speaker_system.speaker_profiles.items():
                        profiles_to_save[name] = {}
                        for key, value in profile.items():
                            if isinstance(value, np.ndarray):
                                profiles_to_save[name][key] = value.tolist()
                            else:
                                profiles_to_save[name][key] = value
                    
                    with open('speaker_profiles.json', 'w', encoding='utf-8') as f:
                        json.dump(profiles_to_save, f, indent=2, ensure_ascii=False)
                    print("[Exit] Saved speaker profiles")
                
                # Save memories
                memory_manager.save_memories_to_disk()
                print("[Exit] Saved memories")
                
            except Exception as e:
                print(f"[Exit] Cleanup error: {e}")
        
        atexit.register(cleanup_on_exit)
        
        # Play startup chime
        print("🔊 Playing startup chime...")
        try:
            play_chime()
        except Exception as e:
            print(f"⚠️  Chime error: {e}")
        
        print("=" * 70)
        print("🎉 BUDDY IS READY! Say something to start chatting...")
        print("=" * 70)
        
        # Start main conversation loop
        buddy_system.run_conversation_loop()
        
    except KeyboardInterrupt:
        print("\n[Main] 🛑 Shutdown requested by user (Ctrl+C)")
    except Exception as e:
        print(f"\n[Main] ❌ Fatal error occurred: {e}")
        import traceback
        traceback.print_exc()
        print("\n💡 Tip: Check that all dependencies are installed and Whisper server is running")
    finally:
        print("\n[Main] 👋 Buddy system shutdown complete")
        print("Thank you for using Enhanced Buddy AI Assistant!")
Fix 5: Add Missing Worker Function Fixes
Find the tts_worker() function and replace it with this enhanced version:

Python
def tts_worker():
    """Enhanced TTS worker with better error handling"""
    print("[Enhanced TTS Worker] Started successfully")
    
    while True:
        try:
            item = tts_queue.get()
            
            if item is None:
                print("[Enhanced TTS Worker] Shutdown signal received")
                break
            
            # Parse TTS request with better handling
            try:
                if isinstance(item, tuple):
                    if len(item) == 2:
                        text, lang = item
                        emotion = "neutral"
                        style = None
                    elif len(item) == 3:
                        text, lang, extra = item
                        if isinstance(extra, dict):
                            emotion = extra.get('emotion', 'neutral')
                            style = extra.get('style', None)
                        else:
                            emotion = str(extra) if extra else "neutral"
                            style = None
                    elif len(item) == 4:
                        text, lang, emotion, style = item
                    else:
                        text, lang, emotion, style = item[0], item[1], item[2] if len(item) > 2 else "neutral", None
                else:
                    text, lang, emotion, style = str(item), "en", "neutral", None
            except Exception as parse_error:
                print(f"[Enhanced TTS Worker] Parse error: {parse_error}, using defaults")
                text, lang, emotion, style = str(item), "en", "neutral", None
            
            # Process TTS if we have valid text
            if text and str(text).strip():
                try:
                    print(f"[Enhanced TTS Worker] Processing: '{text}' (lang={lang}, emotion={emotion})")
                    generate_and_play_enhanced_kokoro(text, lang, emotion, style)
                except Exception as tts_error:
                    print(f"[Enhanced TTS Worker] TTS generation error: {tts_error}")
                    # Try with simpler parameters
                    try:
                        generate_and_play_enhanced_kokoro(text, "en", "neutral", None)
                    except Exception as fallback_error:
                        print(f"[Enhanced TTS Worker] Fallback TTS failed: {fallback_error}")
            
            tts_queue.task_done()
            
        except Exception as e:
            print(f"[Enhanced TTS Worker] Unexpected error: {e}")
            try:
                tts_queue.task_done()
            except:
                pass
            time.sleep(0.1)  # Brief pause before retrying

# ========== MAIN EXECUTION ==========
def main():
    """Main function to start Buddy system"""
    try:
        print("=" * 60)
        print("🤖 ENHANCED BUDDY AI ASSISTANT")
        print("=" * 60)
        print(f"📅 Session started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"👤 System user: Daveydrz")
        print("🔧 Enhanced features: AEC, VAD, Multi-speaker, Memory")
        print("=" * 60)
        
        # Initialize and run Buddy system
        buddy_system = BuddySystem()
        
        # Optional: Play startup chime
        try:
            play_chime()
        except:
            pass  # Continue without chime if it fails
        
        # Start main conversation loop
        buddy_system.run_conversation_loop()
        
    except KeyboardInterrupt:
        print("\n[Main] Shutting down Buddy system...")
    except Exception as e:
        print(f"[Main] Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n[Main] Buddy system shutdown complete")

# ========== ENHANCED WORKER FUNCTIONS ==========
def tts_worker():
    """Enhanced TTS worker with better error handling"""
    print("[Enhanced TTS Worker] Started")
    
    while True:
        try:
            item = tts_queue.get()
            
            if item is None:
                print("[Enhanced TTS Worker] Shutdown signal received")
                break
            
            # Parse TTS request
            if isinstance(item, tuple):
                if len(item) == 2:
                    text, lang = item
                    emotion = "neutral"
                    style = None
                elif len(item) == 3:
                    text, lang, extra = item
                    if isinstance(extra, dict):
                        emotion = extra.get('emotion', 'neutral')
                        style = extra.get('style', None)
                    else:
                        emotion = extra
                        style = None
                else:
                    text, lang, emotion, style = item
            else:
                text, lang, emotion, style = item, "en", "neutral", None
            
            if text and text.strip():
                print(f"[Enhanced TTS Worker] Processing: '{text}' (lang={lang}, emotion={emotion})")
                
                # Generate and queue audio
                generate_and_play_enhanced_kokoro(text, lang, emotion, style)
            
            tts_queue.task_done()
            
        except Exception as e:
            print(f"[Enhanced TTS Worker] Error: {e}")
            tts_queue.task_done()  # Mark as done even on error

def audio_playback_worker():
    """Enhanced audio playback worker with better synchronization"""
    global current_playback
    
    print("[Enhanced Playback Worker] Started")
    
    while True:
        try:
            audio = playback_queue.get()
            
            if audio is None:
                print("[Enhanced Playback Worker] Shutdown signal received")
                break
            
            print("[Enhanced Playback Worker] Playing audio...")
            
            # Set buddy talking flag
            buddy_talking.set()
            
            # Convert audio for reference buffer injection
            samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32767.0
            samples_16k = resample_poly(samples, 16000, audio.frame_rate)
            samples_16k = np.clip(samples_16k, -1.0, 1.0)
            samples_16k_int16 = (samples_16k * 32767).astype(np.int16)
            
            # Start background VAD for interruption
            enhanced_background_vad_thread = threading.Thread(
                target=enhanced_background_vad_listener, 
                daemon=True
            )
            enhanced_background_vad_thread.start()
            
            # Play audio with reference injection
            with playback_lock:
                try:
                    # Inject audio into reference buffer for AEC
                    chunk_size = WEBRTC_FRAME_SIZE
                    total_chunks = len(samples_16k_int16) // chunk_size
                    
                    # Start playback
                    current_playback = play(audio)
                    
                    # Inject reference audio during playback
                    for i in range(total_chunks):
                        if full_duplex_interrupt_flag.is_set():
                            print("[Enhanced Playback] Interruption detected, stopping...")
                            if current_playback and hasattr(current_playback, 'stop'):
                                current_playback.stop()
                            break
                        
                        chunk = samples_16k_int16[i * chunk_size:(i + 1) * chunk_size]
                        update_reference_buffer(chunk)
                        time.sleep(0.01)  # Small delay to synchronize
                    
                    # Wait for playback to complete
                    if current_playback and hasattr(current_playback, 'wait_done'):
                        current_playback.wait_done()
                    
                except Exception as e:
                    print(f"[Enhanced Playback] Playback error: {e}")
                
                finally:
                    # Cleanup
                    buddy_talking.clear()
                    current_playback = None
                    
                    # Clear reference buffer
                    with ref_audio_lock:
                        ref_audio_buffer[:] = np.zeros_like(ref_audio_buffer)
                    
                    # Reset interrupt flag
                    full_duplex_interrupt_flag.clear()
            
            playback_queue.task_done()
            
        except Exception as e:
            print(f"[Enhanced Playback Worker] Error: {e}")
            playback_queue.task_done()

# ========== ENTRY POINT ==========
if __name__ == "__main__":
    try:
        # Load memories at startup
        memory_manager.load_memories_from_disk()
        
        # Initialize speaker system
        if os.path.exists('speaker_profiles.json'):
            try:
                with open('speaker_profiles.json', 'r', encoding='utf-8') as f:
                    speaker_system.speaker_profiles = json.load(f)
                print("[Startup] Loaded speaker profiles")
            except Exception as e:
                print(f"[Startup] Error loading speaker profiles: {e}")
        
        # Start main system
        main()
        
    except Exception as e:
        print(f"[Startup] Fatal error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Save speaker profiles
        try:
            with open('speaker_profiles.json', 'w', encoding='utf-8') as f:
                # Convert numpy arrays to lists for JSON serialization
                profiles_to_save = {}
                for name, profile in speaker_system.speaker_profiles.items():
                    profiles_to_save[name] = {}
                    for key, value in profile.items():
                        if isinstance(value, np.ndarray):
                            profiles_to_save[name][key] = value.tolist()
                        else:
                            profiles_to_save[name][key] = value
                
                json.dump(profiles_to_save, f, indent=2, ensure_ascii=False)
            print("[Shutdown] Saved speaker profiles")
        except Exception as e:
            print(f"[Shutdown] Error saving speaker profiles: {e}")
        
        # Final cleanup
        print("[Shutdown] Enhanced Buddy system terminated")

