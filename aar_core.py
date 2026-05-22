# aar_core.py
import math
import numpy as np

# ============================================================
# QUATERNION MATH
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

    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)

def euler_to_quaternion(roll, pitch, yaw):
    # Inputs in degrees
    r, p, y = math.radians(roll), math.radians(pitch), math.radians(yaw)
    cy = math.cos(y * 0.5)
    sy = math.sin(y * 0.5)
    cp = math.cos(p * 0.5)
    sp = math.sin(p * 0.5)
    cr = math.cos(r * 0.5)
    sr = math.sin(r * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return (qw, qx, qy, qz)

def quaternion_to_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w, 0],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w, 0],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)

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

def quat_rotate(q, v):
    w, x, y, z = q
    qvec = np.array([x, y, z])
    uv = np.cross(qvec, v)
    uuv = np.cross(qvec, uv)
    return v + 2 * (w * uv + uuv)

# ============================================================
# SPATIAL COMPUTATIONS
# ============================================================

def source_relative_angle(q, pos):
    q_inv = quaternion_conjugate(q)
    source = pos.astype(np.float32)
    source /= np.linalg.norm(source)
    relative = quat_rotate(q_inv, source)
    return np.degrees(np.arctan2(relative[1], relative[0]))

def compute_attention(q, sources, ref_dist, sigma, boost_db):
    forward = quat_rotate(q, np.array([1.0, 0.0, 0.0]))
    forward /= np.linalg.norm(forward)
    
    values = []
    for s in sources:
        direction = s["pos"].astype(np.float32)
        distance = np.linalg.norm(direction)
        direction /= distance
        
        # 1. Distance Attenuation (1/r pressure drop)
        dist_factor = ref_dist / max(distance, ref_dist)
        
        # 2. Attention Calculation (0.0 to 1.0)
        angle = np.degrees(np.arccos(np.clip(np.dot(forward, direction), -1.0, 1.0)))
        attention_norm = np.exp(-(angle**2) / (2 * sigma**2)) 
        
        # 3. ORIGINAL BEHAVIOR: Map attention strictly from 0dB to +boost_db
        target_db = boost_db * attention_norm
        boost_linear = 10**(target_db / 20.0)
        
        final_gain = dist_factor * boost_linear
        
        # Returning all 5 elements so your detailed log still works perfectly
        values.append((s["name"], final_gain, attention_norm * 100, dist_factor, target_db))

    return values

def spatialize(segment, target_az, prev_az, frames, fs, max_itd, max_shift_samples):
    if target_az - prev_az > 180: prev_az += 360
    elif target_az - prev_az < -180: prev_az -= 360

    az_ramp = np.linspace(prev_az, target_az, frames)
    az_ramp = np.radians(np.clip(az_ramp, -90, 90))

    itd_ramp = max_itd * np.sin(az_ramp)
    shift_ramp = itd_ramp * fs

    left_shift = np.maximum(0, -shift_ramp)
    right_shift = np.maximum(0, shift_ramp)

    idx = np.arange(frames) + max_shift_samples
    seg_idx = np.arange(len(segment))

    raw_left = np.interp(idx - left_shift, seg_idx, segment)
    raw_right = np.interp(idx - right_shift, seg_idx, segment)

    theta = (az_ramp + np.pi/2) / 2
    left = raw_left * np.sin(theta)
    right = raw_right * np.cos(theta)

    return np.column_stack([left, right])