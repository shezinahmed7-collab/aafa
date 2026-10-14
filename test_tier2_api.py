import urllib.request
import json
import base64
import cv2
import numpy as np

# Case A: Tilted phone with fingers (Tier 2: 0.35 confidence, neutral pose)
h, w = 360, 640
img_phone = np.full((h, w, 3), 180, dtype=np.uint8)
cv2.rectangle(img_phone, (200, 180), (380, 320), (120, 140, 195), -1)
pts = np.array([[240, 190], [340, 230], [310, 310], [210, 270]], np.int32)
cv2.fillPoly(img_phone, [pts], (25, 25, 25))
cv2.polylines(img_phone, [pts], True, (5, 5, 5), 3)

_, buf = cv2.imencode('.jpg', img_phone)
b64_phone = base64.b64encode(buf).decode('utf-8')

payload_a = {
    'image': b64_phone,
    'detections': [{
        'box': [200, 180, 380, 320],
        'confidence': 0.35,
        'class_id': 67,
        'name': 'cell phone'
    }],
    'yaw': 0.0,
    'pitch': 0.0,
    'roll': 0.0,
    'workspace_roi': [150, 100, 500, 350]
}

req_a = urllib.request.Request(
    'http://127.0.0.1:8000/api/verify/angled-phone',
    data=json.dumps(payload_a).encode('utf-8'),
    headers={'Content-Type': 'application/json'}
)
res_a = json.loads(urllib.request.urlopen(req_a).read().decode())

print("Case A (Angled phone in Tier 2):")
print("  Tier 2 verified count:", res_a['tier2_verified_count'])
print("  Confirmed phones count:", len(res_a['confirmed_phones']))
if res_a['confirmed_phones']:
    cp = res_a['confirmed_phones'][0]
    print("  has_rigid_edge:", cp['has_rigid_edge'])
    print("  has_phone_core:", cp['has_phone_core'])
    print("  skin_ratio:", round(cp['skin_ratio'], 3))
    print("  core_ratio:", round(cp['core_ratio'], 3))

# Case B: Bare Hand (100% skin, no edge/core)
img_hand = np.full((h, w, 3), 180, dtype=np.uint8)
cv2.circle(img_hand, (290, 250), 60, (120, 140, 195), -1)
_, buf_h = cv2.imencode('.jpg', img_hand)
b64_hand = base64.b64encode(buf_h).decode('utf-8')

payload_b = {
    'image': b64_hand,
    'detections': [{
        'box': [230, 190, 350, 310],
        'confidence': 0.35,
        'class_id': 67,
        'name': 'cell phone'
    }],
    'yaw': 0.0,
    'pitch': 0.0,
    'roll': 0.0,
    'workspace_roi': [150, 100, 500, 350]
}

req_b = urllib.request.Request(
    'http://127.0.0.1:8000/api/verify/angled-phone',
    data=json.dumps(payload_b).encode('utf-8'),
    headers={'Content-Type': 'application/json'}
)
res_b = json.loads(urllib.request.urlopen(req_b).read().decode())

print("\nCase B (Bare Hand in Tier 2):")
print("  Tier 2 verified count:", res_b['tier2_verified_count'])
print("  Confirmed phones count:", len(res_b['confirmed_phones']))
print("  Suppressed items count:", len(res_b['suppressed_items']))
if res_b['suppressed_items']:
    print("  Suppression reason:", res_b['suppressed_items'][0]['reason'])
