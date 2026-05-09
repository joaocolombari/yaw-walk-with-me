import asyncio
import struct
import math
import nest_asyncio
import numpy as np

from qasync import QEventLoop
from bleak import BleakScanner, BleakClient

import pyqtgraph.opengl as gl

from stl import mesh

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QMatrix4x4

nest_asyncio.apply()

# ============================================================
# CONFIG
# ============================================================

DEVICE_NAME = "Tiresias_DK"

CHAR_UUID = "12345678-1234-5678-1234-56789abcdef1"

STL_FILE = "david-lynch.stl"

latest_quaternion = (1, 0, 0, 0)

tare_quaternion = None

triggered_left = False
triggered_right = False

# ============================================================
# QUATERNION -> EULER
# ============================================================

def quaternion_to_euler(w, x, y, z):

    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x*x + y*y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)

    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y*y + z*z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return (
        math.degrees(roll),
        math.degrees(pitch),
        math.degrees(yaw),
    )

# ============================================================
# QUATERNION -> ROTATION MATRIX
# ============================================================

def quaternion_to_matrix(q):

    w, x, y, z = q

    return np.array([

        [1 - 2*y*y - 2*z*z,
         2*x*y - 2*z*w,
         2*x*z + 2*y*w,
         0],

        [2*x*y + 2*z*w,
         1 - 2*x*x - 2*z*z,
         2*y*z - 2*x*w,
         0],

        [2*x*z - 2*y*w,
         2*y*z + 2*x*w,
         1 - 2*x*x - 2*y*y,
         0],

        [0, 0, 0, 1]

    ], dtype=np.float32)

# ============================================================
# Helpers to look forward
# ============================================================

def quaternion_conjugate(q):

    w, x, y, z = q

    return (w, -x, -y, -z)

def quaternion_multiply(q1, q2):

    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    return (

        w1*w2 - x1*x2 - y1*y2 - z1*z2,

        w1*x2 + x1*w2 + y1*z2 - z1*y2,

        w1*y2 - x1*z2 + y1*w2 + z1*x2,

        w1*z2 + x1*y2 - y1*x2 + z1*w2
    )

# ============================================================
# BLE CALLBACK
# ============================================================

def notification_handler(sender, data):

    global latest_quaternion
    global tare_quaternion

    global triggered_left
    global triggered_right

    if len(data) != 16:
        return

    q_raw = struct.unpack("<ffff", data)

    # ========================================================
    # TARE INICIAL
    # ========================================================

    if tare_quaternion is None:

        tare_quaternion = quaternion_conjugate(q_raw)

        print("\nForward direction calibrated.\n")

    # ========================================================
    # APLICA TARE
    # ========================================================

    q = quaternion_multiply(tare_quaternion, q_raw)

    latest_quaternion = q

    # ========================================================
    # EULER APENAS PARA DETECÇÃO
    # ========================================================

    roll, pitch, yaw = quaternion_to_euler(*q)

    # ========================================================
    # NORMALIZA YAW
    # ========================================================

    while yaw > 180:
        yaw -= 360

    while yaw < -180:
        yaw += 360

    # ========================================================
    # DETECÇÃO DIREITA
    # ========================================================

    if yaw > 90 and not triggered_right:

        print("\nfire walk with me\n")

        triggered_right = True

    if yaw < 70:

        triggered_right = False

    # ========================================================
    # DETECÇÃO ESQUERDA
    # ========================================================

    if yaw < -90 and not triggered_left:

        print("\nfire walk with me\n")

        triggered_left = True

    if yaw > -70:

        triggered_left = False

# ============================================================
# FIND DEVICE
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

# ============================================================
# BLE TASK
# ============================================================

async def ble_task():

    device = await find_device()

    if device is None:
        print("Device not found.")
        return

    print("\nConnecting...\n")

    async with BleakClient(device) as client:

        print("Connected.")
        print("Receiving head orientation...\n")

        await client.start_notify(
            CHAR_UUID,
            notification_handler
        )

        while True:
            await asyncio.sleep(0.01)

# ============================================================
# QT APPLICATION
# ============================================================

app = QApplication.instance()

if app is None:
    app = QApplication([])

# ============================================================
# 3D WINDOW
# ============================================================

view = gl.GLViewWidget()

view.show()

view.setWindowTitle("David Lynch Head Tracking")

view.setCameraPosition(distance=5)

# ============================================================
# GRID
# ============================================================

grid = gl.GLGridItem()

grid.scale(1, 1, 1)

view.addItem(grid)

# ============================================================
# AXES
# ============================================================

axis = gl.GLAxisItem()

axis.setSize(1, 1, 1)

view.addItem(axis)

# ============================================================
# LOAD STL
# ============================================================

your_mesh = mesh.Mesh.from_file(STL_FILE)

vertices = your_mesh.vectors.reshape(-1, 3)

faces = np.arange(vertices.shape[0]).reshape(-1, 3)

# center object
center = vertices.mean(axis=0)

vertices -= center

# normalize scale
scale = np.max(np.linalg.norm(vertices, axis=1))

vertices /= scale

# optional scale factor
vertices *= 2.0

# ============================================================
# CREATE GL MESH
# ============================================================

meshdata = gl.MeshData(
    vertexes=vertices,
    faces=faces
)

head_mesh = gl.GLMeshItem(
    meshdata=meshdata,
    smooth=True,
    drawEdges=False,
    shader='shaded',
)

view.addItem(head_mesh)

# ============================================================
# FORWARD VECTOR
# ============================================================

forward_line = gl.GLLinePlotItem(
    width=4,
    antialias=True
)

view.addItem(forward_line)

# ============================================================
# UPDATE VISUAL
# ============================================================

def quat_rotate(q, v):

    w, x, y, z = q

    qvec = np.array([x, y, z])

    uv = np.cross(qvec, v)
    uuv = np.cross(qvec, uv)

    return v + 2 * (w * uv + uuv)

def update_visual():

    q = latest_quaternion

    # ========================================================
    # ROTATE HEAD
    # ========================================================

    rot = quaternion_to_matrix(q)

    transform = QMatrix4x4(*rot.flatten())

    head_mesh.resetTransform()

    head_mesh.setTransform(transform)

    # ========================================================
    # FORWARD VECTOR
    # ========================================================

    forward = quat_rotate(q, np.array([1, 0, 0]))

    origin = np.array([0, 0, 0])

    forward_line.setData(
        pos=np.array([
            origin,
            forward * 3
        ])
    )

# ============================================================
# TIMER
# ============================================================

timer = QTimer()

timer.timeout.connect(update_visual)

timer.start(16)

# ============================================================
# QT + ASYNCIO LOOP
# ============================================================

loop = QEventLoop(app)

asyncio.set_event_loop(loop)

loop.create_task(ble_task())

with loop:
    loop.run_forever()