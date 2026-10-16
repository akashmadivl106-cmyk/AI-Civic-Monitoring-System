"""
CivicGuard / AI Civic Monitoring — Safe Verification Pipeline Test Harness
==========================================================================
Isolated, non-destructive test suite for the verification pipeline and
data persistence layer.

SAFETY GUARANTEES:
- Operates strictly inside an isolated temporary directory (tempfile).
- Never modifies, overwrites, or deletes production files:
    * logs/alerts.txt
    * logs/violation_history.json
    * zones_config.json
    * evidence/*
    * model weights
    * datasets
- Uses synthetic detections and in-memory fixtures.
- Cleans up all temporary test artifacts upon completion.

Run with:
    python test_pipeline.py
"""

import os
import sys
import json
import time
import shutil
import tempfile
import numpy as np

# ============================================================
# 1. SETUP ISOLATED TEST SANDBOX BEFORE IMPORTING APPLICATION
# ============================================================

_sandbox = tempfile.TemporaryDirectory(prefix="civicguard_safe_test_", ignore_cleanup_errors=True)
_sandbox_dir = _sandbox.name

# Configure environment variables so dashboard_pro binds to the sandbox
os.environ["CIVICGUARD_ZONES_CONFIG"] = os.path.join(_sandbox_dir, "zones_config.json")
os.environ["CIVICGUARD_EVIDENCE_DIR"] = os.path.join(_sandbox_dir, "evidence")
os.environ["CIVICGUARD_LOGS_DIR"] = os.path.join(_sandbox_dir, "logs")
os.environ["CIVICGUARD_HISTORY_FILE"] = os.path.join(_sandbox_dir, "logs", "violation_history.json")
os.environ["CIVICGUARD_ALERTS_FILE"] = os.path.join(_sandbox_dir, "logs", "alerts.txt")

# Ensure sandbox directories exist
os.makedirs(os.path.join(_sandbox_dir, "logs"), exist_ok=True)
os.makedirs(os.path.join(_sandbox_dir, "evidence", "garbage"), exist_ok=True)
os.makedirs(os.path.join(_sandbox_dir, "evidence", "trespassing"), exist_ok=True)
os.makedirs(os.path.join(_sandbox_dir, "evidence", "noparking"), exist_ok=True)

# Now safely import application module
import dashboard_pro as dp

# STRICT ASSERTION: Verify that all paths in dp point exclusively inside the sandbox
assert dp.HISTORY_FILE.startswith(_sandbox_dir), f"UNSAFE: HISTORY_FILE points to {dp.HISTORY_FILE}"
assert dp.ALERTS_FILE.startswith(_sandbox_dir), f"UNSAFE: ALERTS_FILE points to {dp.ALERTS_FILE}"
assert dp.ZONES_CONFIG_FILE.startswith(_sandbox_dir), f"UNSAFE: ZONES_CONFIG_FILE points to {dp.ZONES_CONFIG_FILE}"
assert dp.EVIDENCE_DIR.startswith(_sandbox_dir), f"UNSAFE: EVIDENCE_DIR points to {dp.EVIDENCE_DIR}"

results = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    results.append((name, status, detail))
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))


def reset_state():
    """Resets in-memory tracking states and clears only sandbox test files."""
    dp.garbage_tracks.clear()
    dp.trespass_tracks.clear()
    dp.nopark_tracks.clear()
    dp.violation_history.clear()
    dp._next_event_id = 1
    dp.active_event = None
    dp.last_alert = "No Active Alerts"
    dp.last_garbage_save = 0
    dp.last_trespass_save = 0
    dp.last_nopark_save = 0

    # Clean only sandbox evidence folders
    for cat in ("garbage", "trespassing", "noparking"):
        d = os.path.join(_sandbox_dir, "evidence", cat)
        if os.path.isdir(d):
            for fname in os.listdir(d):
                try:
                    os.remove(os.path.join(d, fname))
                except OSError:
                    pass

    # Clean only sandbox log files
    for f in (dp.ALERTS_FILE, dp.HISTORY_FILE):
        if os.path.exists(f):
            try:
                os.remove(f)
            except OSError:
                pass


def blank_frame():
    return 255 * np.ones((480, 640, 3), dtype="uint8")


def person(pid, cx, cy, conf=0.8):
    x1, y1, x2, y2 = cx - 20, cy - 40, cx + 20, cy + 40
    return {
        "id": pid,
        "bbox": (x1, y1, x2, y2),
        "center": (cx, cy),
        "ground_point": dp.person_ground_point((x1, y1, x2, y2)),
        "confidence": conf,
        "label": "person",
    }


def vehicle(vid, cx, cy, conf=0.8):
    x1, y1, x2, y2 = cx - 40, cy - 25, cx + 40, cy + 25
    return {
        "id": vid,
        "bbox": (x1, y1, x2, y2),
        "center": (cx, cy),
        "ground_point": dp.vehicle_ground_point((x1, y1, x2, y2)),
        "confidence": conf,
        "label": "car",
    }


print("\n============================================================")
print("CIVICGUARD ISOLATED TEST HARNESS — EXECUTING IN SANDBOX")
print(f"Sandbox Location: {_sandbox_dir}")
print("============================================================\n")

# ============================================================
# TEST 0 — Backward compatibility: loading a legacy history file
# ============================================================

_legacy_records = [
    {
        "timestamp": "2026-08-01 10:15:00",
        "violation": "LITTERING",
        "confidence": 0.82,
        "risk": 70,
        "persistence": 12,
        "association": 0.6,
        "evidence": "evidence/garbage/20260801_101500.jpg",
    },
    {
        "timestamp": "2026-08-02 11:20:00",
        "violation": "TRESPASSING",
        "confidence": 0.9,
        "risk": 85,
        "persistence": 15,
        "evidence": "evidence/trespassing/20260802_112000.jpg",
    },
]

with open(dp.HISTORY_FILE, "w", encoding="utf-8") as _f:
    json.dump(_legacy_records, _f, indent=2)

dp.violation_history = dp.load_history()
dp._next_event_id, _backfilled = dp._backfill_missing_ids(dp.violation_history)

check("TEST 0a: Legacy history loads without crashing in sandbox", len(dp.violation_history) == 2)
check("TEST 0b: All legacy records preserved", len(dp.violation_history) == len(_legacy_records))
ids = [r.get("id") for r in dp.violation_history]
check("TEST 0c: Every record assigned unique integer id", all(isinstance(i, int) for i in ids) and len(set(ids)) == len(ids), f"ids={ids}")
check("TEST 0d: _next_event_id continues correctly", dp._next_event_id == max(ids) + 1)

# ============================================================
# TEST 1 — Garbage alone -> no littering verified
# ============================================================

reset_state()
frame_width = 640

for i in range(15):
    state = dp.process_garbage_relation(
        track_id=1, garbage_center=(300, 300), garbage_confidence=0.75,
        persons=[], frame_width=frame_width,
    )

check("TEST 1: Garbage alone never verifies", state["verified"] is False and len(dp.violation_history) == 0)

# ============================================================
# TEST 2 — Person briefly near garbage -> no verification
# ============================================================

reset_state()

for i in range(2):  # below MIN_ASSOCIATION_FRAMES (5)
    state = dp.process_garbage_relation(
        track_id=2, garbage_center=(300, 300), garbage_confidence=0.75,
        persons=[person(101, 305, 305)], frame_width=frame_width,
    )

check("TEST 2: Brief association alone never verifies", state["verified"] is False and len(dp.violation_history) == 0)

# ============================================================
# TEST 3 — Full littering sequence: associate, separate, stabilize, persist
# ============================================================

reset_state()
track_id = 3

for i in range(7):
    state = dp.process_garbage_relation(
        track_id=track_id, garbage_center=(300, 300), garbage_confidence=0.8,
        persons=[person(102, 305, 302)], frame_width=frame_width,
    )

assoc_ok = state["association_frames"] >= dp.MIN_ASSOCIATION_FRAMES
far_x = 300 + int(frame_width * dp.SEPARATION_DISTANCE_RATIO) + 50

for i in range(7):
    state = dp.process_garbage_relation(
        track_id=track_id, garbage_center=(300, 300), garbage_confidence=0.8,
        persons=[person(102, far_x, 300)], frame_width=frame_width,
    )

sep_ok = state["separation_frames"] >= dp.MIN_SEPARATION_FRAMES
stat_ok = state["stationary_frames"] >= dp.MIN_STATIONARY_FRAMES

time.sleep(dp.MIN_GARBAGE_PERSISTENCE + 0.1)
state = dp.process_garbage_relation(
    track_id=track_id, garbage_center=(300, 300), garbage_confidence=0.8,
    persons=[person(102, far_x, 300)], frame_width=frame_width,
)

verified_ok = state["verified"] is True and state["littering_verified_this_frame"] is True

if state["littering_verified_this_frame"]:
    avg_conf = sum(state["confidence_history"]) / len(state["confidence_history"])
    assoc_score = min(state["association_frames"] / 20.0, 1.0)
    risk = dp.calculate_risk(avg_conf, state["stationary_frames"], assoc_score)
    dp.create_verified_event(
        violation="LITTERING", category="garbage", confidence=avg_conf, risk=risk,
        frame=blank_frame(), bbox=(280, 280, 320, 320),
        persistence=state["stationary_frames"], association=assoc_score, evidence_dir="garbage",
    )

event_created = len(dp.violation_history) == 1 and dp.violation_history[0]["violation"] == "LITTERING"
evidence_saved = event_created and dp.violation_history[0]["evidence"] and os.path.exists(dp.violation_history[0]["evidence"])

check("TEST 3a: Association stage reached", assoc_ok)
check("TEST 3b: Separation stage reached", sep_ok)
check("TEST 3c: Stability stage reached", stat_ok)
check("TEST 3d: LITTERING VERIFIED after full sequence", verified_ok)
check("TEST 3e: Verified event written to history + evidence saved in sandbox", event_created and evidence_saved)

# ============================================================
# TEST 4 — Person outside trespass polygon -> no trespassing
# ============================================================

reset_state()

for i in range(15):
    p = person(201, 20, 20)
    inside = dp.point_in_zone(p["ground_point"], dp.TRESPASS_ZONE)
    verified_now, entry = dp.update_zone_track(dp.trespass_tracks, p["id"], inside, p["confidence"], None)

check("TEST 4: Person outside polygon never verifies", inside is False and entry["frames_inside"] == 0 and len(dp.violation_history) == 0)

# ============================================================
# TEST 5 — Person briefly inside trespass polygon -> no verification
# ============================================================

reset_state()

for i in range(3):
    p = person(202, 150, 260)
    inside = dp.point_in_zone(p["ground_point"], dp.TRESPASS_ZONE)
    verified_now, entry = dp.update_zone_track(dp.trespass_tracks, p["id"], inside, p["confidence"], None)

check("TEST 5: Brief polygon entry never verifies", inside is True and verified_now is False and len(dp.violation_history) == 0)

# ============================================================
# TEST 6 — Person persists inside trespass polygon -> VERIFIED
# ============================================================

reset_state()

for i in range(dp.VERIFY_FRAMES + 2):
    p = person(203, 150, 260)
    inside = dp.point_in_zone(p["ground_point"], dp.TRESPASS_ZONE)
    verified_now, entry = dp.update_zone_track(dp.trespass_tracks, p["id"], inside, p["confidence"], None)
    if verified_now:
        avg_conf = sum(entry["confidence_history"]) / len(entry["confidence_history"])
        risk = dp.calculate_risk(avg_conf, entry["frames_inside"], context_score=1.0)
        dp.create_verified_event(
            violation="TRESPASSING", category="trespassing", confidence=avg_conf, risk=risk,
            frame=blank_frame(), bbox=p["bbox"], persistence=entry["frames_inside"],
            association=None, evidence_dir="trespassing",
        )
        break

event_ok = len(dp.violation_history) == 1 and dp.violation_history[0]["violation"] == "TRESPASSING"
no_assoc_field = "association" not in dp.violation_history[0] if event_ok else False

check("TEST 6a: Sustained polygon presence -> TRESPASSING VERIFIED", verified_now and event_ok)
check("TEST 6b: Trespassing event correctly omits association", no_assoc_field)

# ============================================================
# TEST 7 — Vehicle briefly crosses no-parking polygon -> no verification
# ============================================================

reset_state()

for i in range(4):
    v = vehicle(301, 480, 275)
    inside = dp.point_in_zone(v["ground_point"], dp.NO_PARKING_ZONE)
    verified_now, entry = dp.update_zone_track(dp.nopark_tracks, v["id"], inside, v["confidence"], None)

check("TEST 7: Vehicle briefly crossing zone never verifies", inside is True and verified_now is False and len(dp.violation_history) == 0)

# ============================================================
# TEST 8 — Vehicle persists inside no-parking polygon -> VERIFIED
# ============================================================

reset_state()

for i in range(dp.VERIFY_FRAMES + 2):
    v = vehicle(302, 480, 275)
    inside = dp.point_in_zone(v["ground_point"], dp.NO_PARKING_ZONE)
    verified_now, entry = dp.update_zone_track(dp.nopark_tracks, v["id"], inside, v["confidence"], None)
    if verified_now:
        avg_conf = sum(entry["confidence_history"]) / len(entry["confidence_history"])
        risk = dp.calculate_risk(avg_conf, entry["frames_inside"], context_score=1.0)
        dp.create_verified_event(
            violation="NO PARKING", category="noparking", confidence=avg_conf, risk=risk,
            frame=blank_frame(), bbox=v["bbox"], persistence=entry["frames_inside"],
            association=None, evidence_dir="noparking",
        )
        break

event_ok8 = len(dp.violation_history) == 1 and dp.violation_history[0]["violation"] == "NO PARKING"
check("TEST 8: Sustained polygon presence -> NO PARKING VERIFIED", verified_now and event_ok8)

# ============================================================
# TEST 9 — Continuous violation does not spam; exit + reentry creates new event
# ============================================================

reset_state()
event_count_after_first = None

for i in range(dp.VERIFY_FRAMES + 40):
    v = vehicle(303, 480, 275)
    inside = dp.point_in_zone(v["ground_point"], dp.NO_PARKING_ZONE)
    verified_now, entry = dp.update_zone_track(dp.nopark_tracks, v["id"], inside, v["confidence"], None)
    if verified_now:
        avg_conf = sum(entry["confidence_history"]) / len(entry["confidence_history"])
        risk = dp.calculate_risk(avg_conf, entry["frames_inside"], context_score=1.0)
        dp.create_verified_event(
            violation="NO PARKING", category="noparking", confidence=avg_conf, risk=risk,
            frame=blank_frame(), bbox=v["bbox"], persistence=entry["frames_inside"],
            association=None, evidence_dir="noparking",
        )
        if event_count_after_first is None:
            event_count_after_first = len(dp.violation_history)

check("TEST 9a: No duplicate event spam for continuous presence", len(dp.violation_history) == 1 == event_count_after_first)

# Exit zone for 3 frames
for i in range(3):
    dp.update_zone_track(dp.nopark_tracks, 303, False, 0.8, None)

# Re-enter zone
for i in range(dp.VERIFY_FRAMES + 2):
    v = vehicle(303, 480, 275)
    inside = dp.point_in_zone(v["ground_point"], dp.NO_PARKING_ZONE)
    verified_now, entry = dp.update_zone_track(dp.nopark_tracks, v["id"], inside, v["confidence"], None)
    if verified_now:
        avg_conf = sum(entry["confidence_history"]) / len(entry["confidence_history"])
        risk = dp.calculate_risk(avg_conf, entry["frames_inside"], context_score=1.0)
        dp.create_verified_event(
            violation="NO PARKING", category="noparking", confidence=avg_conf, risk=risk,
            frame=blank_frame(), bbox=v["bbox"], persistence=entry["frames_inside"],
            association=None, evidence_dir="noparking",
        )
        break

check("TEST 9b: Exit + re-entry creates second verified event", len(dp.violation_history) == 2)

# ============================================================
# TEST 10 — Dashboard API reflects verified events
# ============================================================

client = dp.app.test_client()
resp = client.get("/api/status")
data = resp.get_json()

check("TEST 10a: Active event populated", dp.active_event is not None and dp.active_event["violation"] == "NO PARKING")
check("TEST 10b: /api/status counts match", data["counts"]["noparking"] == 2)
check("TEST 10c: /api/status last_event_id matches", data["last_event_id"] == dp.violation_history[-1]["id"])

events_resp = client.get("/api/events")
events_data = events_resp.get_json()
check("TEST 10d: Violations table HTML fragment rendered", "NO PARKING" in events_data["violations_table_html"])
check("TEST 10e: Evidence gallery fragment rendered", "evidence-card" in events_data["evidence_gallery_html"])

# ============================================================
# TEST 11 — Bounding box aware distance for tall persons
# ============================================================

reset_state()
tall_person_bbox = (280, 150, 340, 430)
tall_person_center = ((280 + 340) // 2, (150 + 430) // 2)
garbage_near_feet = (345, 420)

old_dist = dp.distance(garbage_near_feet, tall_person_center)
new_dist = dp.point_to_bbox_distance(garbage_near_feet, tall_person_bbox)
threshold = frame_width * dp.PERSON_GARBAGE_DISTANCE_RATIO

check("TEST 11a: Geometry confirms bbox-distance superiority", old_dist > threshold and new_dist <= threshold)

tall_p = {
    "id": 401, "bbox": tall_person_bbox, "center": tall_person_center,
    "ground_point": dp.person_ground_point(tall_person_bbox), "confidence": 0.85, "label": "person",
}

for i in range(dp.MIN_ASSOCIATION_FRAMES + 2):
    state = dp.process_garbage_relation(
        track_id=11, garbage_center=garbage_near_feet, garbage_confidence=0.8,
        persons=[tall_p], frame_width=frame_width,
    )

check("TEST 11b: Association successfully accumulates", state["association_frames"] >= dp.MIN_ASSOCIATION_FRAMES)

# ============================================================
# TEST 12 — Sustained holding of garbage never verifies littering
# ============================================================

reset_state()
holder = person(501, 305, 302)

for i in range(40):
    state = dp.process_garbage_relation(
        track_id=12, garbage_center=(300, 300), garbage_confidence=0.8,
        persons=[holder], frame_width=frame_width,
    )

check("TEST 12: Continuous holding never verifies littering", state["verified"] is False and state["separation_frames"] == 0 and len(dp.violation_history) == 0)

# ============================================================
# TEST 13 — Zone Calibration API (in sandbox)
# ============================================================

reset_state()
new_polygon = [[10, 10], [250, 10], [250, 250], [10, 250]]

zone_post = client.post("/api/zones", json={"trespass_zone": new_polygon})
zone_post_data = zone_post.get_json()

check("TEST 13a: Valid zone POST returns 200 ok", zone_post.status_code == 200 and zone_post_data.get("ok") is True)
check("TEST 13b: Live polygon updated in memory", dp.TRESPASS_ZONE.tolist() == new_polygon)
check("TEST 13c: Polygon persisted into sandbox config file", os.path.exists(dp.ZONES_CONFIG_FILE))

bad_post = client.post("/api/zones", json={"trespass_zone": [[1, 1], [2, 2]]})
check("TEST 13d: Invalid polygon (<3 vertices) rejected with 400", bad_post.status_code == 400)

# ============================================================
# TEST 14 — (PHASE 1) Legacy Record Normalization
# ============================================================

legacy_sample_1 = {"timestamp": "2026-08-20 11:14:49", "type": "trespassing", "confidence": 0.901, "risk": 81, "id": 1}
norm_1 = dp.normalize_event_record(legacy_sample_1)
check("TEST 14a: Legacy 'type: trespassing' mapped to violation & category", norm_1["violation"] == "TRESPASSING" and norm_1["category"] == "trespassing")

legacy_sample_2 = {"timestamp": "2026-08-20 11:14:55", "type": "garbage", "confidence": 0.817, "risk": 80, "id": 3}
norm_2 = dp.normalize_event_record(legacy_sample_2)
check("TEST 14b: Legacy 'type: garbage' mapped to LITTERING & garbage", norm_2["violation"] == "LITTERING" and norm_2["category"] == "garbage")

legacy_sample_3 = {"timestamp": "2026-08-21 09:41:30", "violation": "TRESPASSING", "confidence": 0.712, "risk": 72, "evidence": "evidence/trespassing/20260821_094130.jpg", "id": 6}
norm_3 = dp.normalize_event_record(legacy_sample_3)
check("TEST 14c: Missing category automatically derived from violation", norm_3["category"] == "trespassing" and norm_3["evidence_filename"] == "20260821_094130.jpg")

# ============================================================
# TEST 15 — (PHASE 1) Windows & POSIX Evidence Path Resolution
# ============================================================

windows_path_event = {
    "violation": "TRESPASSING",
    "category": "trespassing",
    "evidence": "evidence\\trespassing\\20260821_094130_322197.jpg",
}
windows_url = dp.template_evidence_url(windows_path_event)
check("TEST 15a: Windows backslash path correctly resolved to URL", windows_url == "/evidence/trespassing/20260821_094130_322197.jpg", f"url={windows_url}")

posix_path_event = {
    "violation": "LITTERING",
    "category": "garbage",
    "evidence": "evidence/garbage/20260825_093321_937991.jpg",
}
posix_url = dp.template_evidence_url(posix_path_event)
check("TEST 15b: POSIX forward-slash path correctly resolved to URL", posix_url == "/evidence/garbage/20260825_093321_937991.jpg", f"url={posix_url}")

win_filename = dp.template_evidence_filename("evidence\\noparking\\20260902_163957.jpg")
check("TEST 15c: template_evidence_filename extracts clean filename from Windows path", win_filename == "20260902_163957.jpg")

# ============================================================
# TEST 16 — (PHASE 1) Atomic Persistence Safety
# ============================================================

test_target_file = os.path.join(_sandbox_dir, "test_atomic.json")
test_payload = {"system": "CivicGuard", "active": True, "count": 42}

atomic_ok = dp.atomic_json_dump(test_target_file, test_payload)
check("TEST 16a: atomic_json_dump writes valid JSON", atomic_ok and os.path.exists(test_target_file))

with open(test_target_file, "r", encoding="utf-8") as _f:
    read_back = json.load(_f)
check("TEST 16b: Atomic file content matches written data", read_back == test_payload)

# Verify no leftover temp files exist in sandbox
temp_leftovers = [f for f in os.listdir(_sandbox_dir) if ".tmp_" in f]
check("TEST 16c: Atomic write cleans up all temporary files", len(temp_leftovers) == 0, f"leftovers={temp_leftovers}")

# ============================================================
# TEST 17 — (PHASE 1) Evidence Route Security & Traversal Resistance
# ============================================================

# Create a sample evidence image in sandbox
sample_img_path = os.path.join(_sandbox_dir, "evidence", "trespassing", "sample.jpg")
with open(sample_img_path, "wb") as _f:
    _f.write(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xdb")

valid_req = client.get("/evidence/trespassing/sample.jpg")
check("TEST 17a: Authorized evidence category request succeeds", valid_req.status_code == 200)
valid_req.close()

invalid_cat = client.get("/evidence/unauthorized_cat/sample.jpg")
check("TEST 17b: Unauthorized evidence category blocked with 404", invalid_cat.status_code == 404)
invalid_cat.close()

traversal_req = client.get("/evidence/trespassing/../violation_history.json")
check("TEST 17c: Directory traversal attempt blocked with 404", traversal_req.status_code == 404)
traversal_req.close()

# ============================================================
# SUMMARY & CLEANUP
# ============================================================

print("\n" + "=" * 60)
passed = sum(1 for _, s, _ in results if s == "PASS")
failed = sum(1 for _, s, _ in results if s == "FAIL")
print(f"RESULTS: {passed} passed, {failed} failed, {len(results)} total")
if failed:
    print("\nFAILED TESTS:")
    for name, status, detail in results:
        if status == "FAIL":
            print(f"  - {name}: {detail}")
print("=" * 60)

# Clean up sandbox
_sandbox.cleanup()
print(f"[INFO] Test sandbox {_sandbox_dir} was cleanly removed.")
print("[INFO] Production files and directories are completely untouched.")

if failed > 0:
    sys.exit(1)
