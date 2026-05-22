# main.py
import asyncio
import struct
import math
import nest_asyncio
import numpy as np
import sounddevice as sd
import soundfile as sf

from qasync import QEventLoop
from bleak import BleakScanner, BleakClient
import pyqtgraph.opengl as gl
from stl import mesh
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QMatrix4x4

import aar_core # IMPORTING OUR STANDALONE MODULE

nest_asyncio.apply()

# ============================================================
# CONFIG
# ============================================================
DEVICE_NAME = "Tiresias_DK"
CHAR_UUID = "12345678-1234-5678-1234-56789abcdef1"
STL_FILE = "david-lynch.stl"

latest_quaternion = (1, 0, 0, 0)
tare_quaternion = None

ATTENTION_SIGMA = 20
ATTENTION_BOOST_DB = 10.0
REFERENCE_DISTANCE = 1.0

sources = [
    {"name":"Laura", "pos":np.array([1.41, -1.41, 0.]), "file":"laura.mp3"},
    {"name":"Coop",  "pos":np.array([1.41, 1.41, 0.]),  "file":"cooper.mp3"}
]

audio = []
for s in sources:
    x, fs = sf.read(s["file"], dtype="float32")
    if x.ndim > 1: x = x.mean(axis=1)
    audio.append(x)

FS = fs
positions = [0 for _ in sources]
MAX_ITD = 0.0007
MAX_SHIFT_SAMPLES = int(math.ceil(MAX_ITD * FS)) + 2

# ============================================================
# AUDIO CALLBACK
# ============================================================
def audio_callback(outdata, frames, time, status):
    try:
        mix = np.zeros((frames, 2), dtype=np.float32)
        
        if not hasattr(audio_callback, "prev_gain"):
            audio_callback.prev_gain = [0.0 for _ in sources]
            audio_callback.prev_az = [0.0 for _ in sources]

        alpha = 0.98
        attention_data = aar_core.compute_attention(latest_quaternion, sources, REFERENCE_DISTANCE, ATTENTION_SIGMA, ATTENTION_BOOST_DB)

        # Unpack the new 5-element tuple, we only need the first two for audio
        for i, (name, target_gain, _, _, _) in enumerate(attention_data):
            
            prev_gain = audio_callback.prev_gain[i]
            # Smoothly interpolate toward the new target gain
            audio_callback.prev_gain[i] = (alpha * prev_gain + (1 - alpha) * target_gain)
            gain_ramp = np.linspace(prev_gain, audio_callback.prev_gain[i], frames, dtype=np.float32)[:, None]

            chunk = audio[i]
            p0 = positions[i]
            
            indices = np.arange(p0 - MAX_SHIFT_SAMPLES, p0 + frames)
            segment = np.take(chunk, indices, mode='wrap')
            positions[i] = (p0 + frames) % len(chunk)

            target_az = aar_core.source_relative_angle(latest_quaternion, sources[i]["pos"])
            prev_az = audio_callback.prev_az[i]
            audio_callback.prev_az[i] = target_az

            stereo = aar_core.spatialize(segment, target_az, prev_az, frames, FS, MAX_ITD, MAX_SHIFT_SAMPLES)
            mix += gain_ramp * stereo

        peak = np.max(np.abs(mix))
        if peak > 1: mix /= peak
        outdata[:] = mix

    except Exception as e:
        print("\nAudio error:", e)
        outdata.fill(0)

# ============================================================
# BLE CALLBACK
# ============================================================
def notification_handler(sender, data):
    global latest_quaternion, tare_quaternion

    if len(data) != 16: return
    q_raw = struct.unpack("<ffff", data)

    if tare_quaternion is None:
        tare_quaternion = aar_core.quaternion_conjugate(q_raw)
        print("\nForward direction calibrated.\n")

    q = aar_core.quaternion_multiply(tare_quaternion, q_raw)
    latest_quaternion = q


    attention_data = aar_core.compute_attention(q, sources, REFERENCE_DISTANCE, ATTENTION_SIGMA, ATTENTION_BOOST_DB)
    text = []

    for name, final_gain, att_pct, dist_factor, target_db in attention_data:
        # Formatting: Name | Distance Multiplier | AAR Boost (dB) | Final Multiplier
        text.append(f"{name}: Dist x{dist_factor:.2f} | AAR {target_db:>+5.1f}dB | Total x{final_gain:.3f}")

    # Separating sources with a double pipe for clarity
    print("\r" + " || ".join(text), end="")

# ============================================================
# BOILERPLATE TASKS (BLE + UI)
# ============================================================
async def find_device():
    print("Scanning for BLE devices...\n")
    devices = await BleakScanner.discover(timeout=5.0)

    for d in devices:
        if d.name and DEVICE_NAME in d.name:
            print(f"Found device: {d.name}")
            print(f"Address: {d.address}")
            return d
    return None

async def ble_task():
    device = await find_device()
    if device is None:
        print("Device not found.")
        return

    print("\nConnecting...\n")

    async with BleakClient(device) as client:
        print("Connected.")
        print("Receiving head orientation...\n")

        await client.start_notify(CHAR_UUID, notification_handler)

        while True:
            await asyncio.sleep(0.01)

app = QApplication.instance() or QApplication([])

view = gl.GLViewWidget()
view.show()
view.setCameraPosition(distance=5)
view.addItem(gl.GLGridItem())
view.addItem(gl.GLAxisItem())

your_mesh = mesh.Mesh.from_file(STL_FILE)
vertices = your_mesh.vectors.reshape(-1, 3)
faces = np.arange(vertices.shape[0]).reshape(-1, 3)
vertices -= vertices.mean(axis=0)
vertices /= np.max(np.linalg.norm(vertices, axis=1))

head_mesh = gl.GLMeshItem(meshdata=gl.MeshData(vertexes=vertices, faces=faces), smooth=True, drawEdges=False, shader='shaded')
view.addItem(head_mesh)

forward_line = gl.GLLinePlotItem(width=4, antialias=True)
view.addItem(forward_line)

for s in sources:
    view.addItem(gl.GLScatterPlotItem(pos=np.array([s["pos"]]), size=30))
    view.addItem(gl.GLLinePlotItem(pos=np.array([[0,0,0], s["pos"]]), width=1))

def update_visual():
    rot = aar_core.quaternion_to_matrix(latest_quaternion)
    head_mesh.setTransform(QMatrix4x4(*rot.flatten()))
    forward = aar_core.quat_rotate(latest_quaternion, np.array([1, 0, 0]))
    forward_line.setData(pos=np.array([[0,0,0], forward * 3]))

timer = QTimer()
timer.timeout.connect(update_visual)
timer.start(16)

loop = QEventLoop(app)
asyncio.set_event_loop(loop)
loop.create_task(ble_task())

stream = sd.OutputStream(samplerate=FS, channels=2, callback=audio_callback, blocksize=512)
stream.start()

with loop:
    loop.run_forever()