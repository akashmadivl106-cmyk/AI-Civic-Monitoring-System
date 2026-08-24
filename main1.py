from ultralytics import YOLO
import cv2
import os
import time
import math
import json
from datetime import datetime
from collections import deque


# ============================================================
# AI CIVIC MONITORING SYSTEM
# Adaptive Context-Aware Civic Violation Verification
#
# Violations:
#   1. Littering
#   2. Trespassing
#   3. No Parking
#
# IMPORTANT:
# Garbage detection alone does NOT mean littering.
#
# Littering verification uses:
#   Person-Garbage association
#   Temporal persistence
#   Garbage-person separation
#   Garbage motion/stability
#   Person departure
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

CAMERA_ID = 0

# Detection thresholds
GARBAGE_CONFIDENCE = 0.60
PERSON_CONFIDENCE = 0.55
VEHICLE_CONFIDENCE = 0.55

# Number of frames required for verification
MIN_ASSOCIATION_FRAMES = 5
MIN_SEPARATION_FRAMES = 5
MIN_STATIONARY_FRAMES = 8

# Minimum time before another evidence image
EVIDENCE_COOLDOWN = 5

# Maximum distance between person and garbage
# expressed relative to image width
PERSON_GARBAGE_DISTANCE_RATIO = 0.18

# After garbage separates from person,
# how far apart should they become?
SEPARATION_DISTANCE_RATIO = 0.25

# Maximum garbage movement to consider it stationary
STATIONARY_MOVEMENT_PIXELS = 25

# How long garbage should remain after person leaves
MIN_GARBAGE_PERSISTENCE = 2.0


# ============================================================
# ZONES
# ============================================================

# Trespassing zone
T_X1, T_Y1 = 150, 100
T_X2, T_Y2 = 500, 400

# No parking zone
P_X1, P_Y1 = 200, 100
P_X2, P_Y2 = 500, 400


# ============================================================
# LOAD MODELS
# ============================================================

print("Loading garbage model...")

garbage_model = YOLO(
    "runs/detect/train/weights/best.pt"
)

print("Loading YOLOv8 model...")

coco_model = YOLO(
    "yolov8n.pt"
)

print("Models loaded successfully.")


# ============================================================
# DIRECTORIES
# ============================================================

os.makedirs("evidence/garbage", exist_ok=True)
os.makedirs("evidence/trespassing", exist_ok=True)
os.makedirs("evidence/noparking", exist_ok=True)
os.makedirs("logs", exist_ok=True)


# ============================================================
# EVENT HISTORY
# ============================================================

history_file = "logs/violation_history.json"


def load_history():

    if not os.path.exists(history_file):
        return []

    try:

        with open(history_file, "r") as file:
            return json.load(file)

    except:

        return []


violation_history = load_history()


# ============================================================
# LOGGING
# ============================================================

def log_alert(
    violation,
    confidence,
    risk,
    evidence,
    extra=""
):

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    message = (
        f"[{timestamp}] "
        f"{violation} VERIFIED | "
        f"confidence={confidence:.2f} | "
        f"risk={risk} | "
        f"evidence={evidence}"
    )

    if extra:
        message += " | " + extra

    print(message)

    with open(
        "logs/alerts.txt",
        "a",
        encoding="utf-8"
    ) as file:

        file.write(message + "\n")


def save_history(
    violation,
    confidence,
    risk,
    evidence
):

    event = {

        "timestamp":
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

        "violation": violation,

        "confidence":
            round(float(confidence), 3),

        "risk":
            int(risk),

        "evidence":
            evidence

    }

    violation_history.append(event)

    with open(
        history_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            violation_history,
            file,
            indent=4
        )


# ============================================================
# RISK SCORE
# ============================================================

def calculate_risk(
    confidence,
    persistence,
    context_score=0.0
):

    score = (

        confidence * 45

        +

        min(
            persistence,
            60
        ) * 0.5

        +

        context_score * 35
    )

    return min(
        int(score),
        100
    )


# ============================================================
# DISTANCE FUNCTION
# ============================================================

def distance(point1, point2):

    return math.sqrt(

        (point1[0] - point2[0]) ** 2

        +

        (point1[1] - point2[1]) ** 2

    )


# ============================================================
# GARBAGE TRACK STATE
# ============================================================

garbage_tracks = {}


# ============================================================
# GARBAGE TRACK INITIALIZATION
# ============================================================

def create_garbage_track(
    track_id,
    center,
    confidence
):

    garbage_tracks[track_id] = {

        "center_history":
            deque(maxlen=30),

        "confidence_history":
            deque(maxlen=30),

        "association_frames":
            0,

        "separation_frames":
            0,

        "stationary_frames":
            0,

        "person_near":
            False,

        "person_left":
            False,

        "associated_person":
            None,

        "first_seen":
            time.time(),

        "separation_time":
            None,

        "verified":
            False,

        "last_event_time":
            0,

        "last_center":
            center

    }

    garbage_tracks[
        track_id
    ][
        "center_history"
    ].append(center)

    garbage_tracks[
        track_id
    ][
        "confidence_history"
    ].append(confidence)


# ============================================================
# GARBAGE-PERSON RELATION
# ============================================================

def process_garbage_relation(
    track_id,
    garbage_center,
    garbage_confidence,
    persons,
    frame_width
):

    if track_id not in garbage_tracks:

        create_garbage_track(
            track_id,
            garbage_center,
            garbage_confidence
        )

    state = garbage_tracks[track_id]

    state[
        "center_history"
    ].append(
        garbage_center
    )

    state[
        "confidence_history"
    ].append(
        garbage_confidence
    )


    # --------------------------------------------------------
    # FIND CLOSEST PERSON
    # --------------------------------------------------------

    closest_person = None
    closest_distance = float("inf")

    for person in persons:

        d = distance(
            garbage_center,
            person["center"]
        )

        if d < closest_distance:

            closest_distance = d

            closest_person = person


    # Normalize distance according to image width

    association_distance = (
        frame_width *
        PERSON_GARBAGE_DISTANCE_RATIO
    )

    separation_distance = (
        frame_width *
        SEPARATION_DISTANCE_RATIO
    )


    # ========================================================
    # STAGE 1
    # PERSON-GARBAGE ASSOCIATION
    # ========================================================

    if (
        closest_person is not None
        and
        closest_distance <= association_distance
    ):

        state[
            "association_frames"
        ] += 1

        state[
            "person_near"
        ] = True

        state[
            "associated_person"
        ] = closest_person["id"]

    else:

        # Do not immediately erase association.
        # We need temporal separation.

        if state[
            "person_near"
        ]:

            if (
                closest_person is None
                or
                closest_distance >=
                separation_distance
            ):

                state[
                    "separation_frames"
                ] += 1

            else:

                state[
                    "separation_frames"
                ] = 0


    # ========================================================
    # STAGE 2
    # GARBAGE SEPARATION
    # ========================================================

    separated = (

        state[
            "association_frames"
        ]
        >=
        MIN_ASSOCIATION_FRAMES

        and

        state[
            "separation_frames"
        ]
        >=
        MIN_SEPARATION_FRAMES
    )


    if separated:

        if state[
            "separation_time"
        ] is None:

            state[
                "separation_time"
            ] = time.time()


        state[
            "person_left"
        ] = True


    # ========================================================
    # STAGE 3
    # GARBAGE STABILITY
    # ========================================================

    movement = distance(

        state["last_center"],

        garbage_center

    )

    state[
        "last_center"
    ] = garbage_center


    if movement <= STATIONARY_MOVEMENT_PIXELS:

        state[
            "stationary_frames"
        ] += 1

    else:

        state[
            "stationary_frames"
        ] = 0


    # ========================================================
    # STAGE 4
    # FINAL LITTERING VERIFICATION
    # ========================================================

    persistence_after_separation = 0

    if state[
        "separation_time"
    ] is not None:

        persistence_after_separation = (

            time.time()

            -

            state[
                "separation_time"
            ]
        )


    littering_verified = (

        state[
            "association_frames"
        ]
        >=
        MIN_ASSOCIATION_FRAMES

        and

        state[
            "separation_frames"
        ]
        >=
        MIN_SEPARATION_FRAMES

        and

        state[
            "stationary_frames"
        ]
        >=
        MIN_STATIONARY_FRAMES

        and

        persistence_after_separation
        >=
        MIN_GARBAGE_PERSISTENCE
    )


    return (

        littering_verified,

        state,

        closest_person,

        closest_distance
    )


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(
    CAMERA_ID
)

if not cap.isOpened():

    print(
        "ERROR: Could not open camera."
    )

    exit()


# ============================================================
# ALERT COOLDOWNS
# ============================================================

last_garbage_save = 0
last_trespass_save = 0
last_nopark_save = 0


# ============================================================
# COUNTERS
# ============================================================

trespass_counter = 0
nopark_counter = 0

VERIFY_FRAMES = 10


# ============================================================
# MAIN LOOP
# ============================================================

print(
    "AI Civic Monitoring System started."
)

print(
    "Press Q to exit."
)


while True:

    success, frame = cap.read()

    if not success:

        print(
            "Camera frame unavailable."
        )

        break


    frame_height, frame_width = (
        frame.shape[:2]
    )


    # ========================================================
    # DRAW ZONES
    # ========================================================

    cv2.rectangle(

        frame,

        (T_X1, T_Y1),

        (T_X2, T_Y2),

        (0, 0, 255),

        2
    )

    cv2.putText(

        frame,

        "TRESPASS ZONE",

        (T_X1, T_Y1 - 10),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.6,

        (0, 0, 255),

        2
    )


    cv2.rectangle(

        frame,

        (P_X1, P_Y1),

        (P_X2, P_Y2),

        (255, 0, 0),

        2
    )

    cv2.putText(

        frame,

        "NO PARKING ZONE",

        (P_X1, P_Y1 - 10),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.6,

        (255, 0, 0),

        2
    )


    # ========================================================
    # PERSON + VEHICLE DETECTION
    # ========================================================

    coco_results = coco_model.track(

        frame,

        persist=True,

        tracker="bytetrack.yaml",

        conf=PERSON_CONFIDENCE,

        verbose=False
    )


    persons = []

    vehicles = []


    for result in coco_results:

        if result.boxes is None:
            continue


        for box in result.boxes:

            confidence = float(
                box.conf[0]
            )

            if confidence < PERSON_CONFIDENCE:
                continue


            cls = int(
                box.cls[0]
            )

            label = (
                coco_model.names[cls]
            )


            x1, y1, x2, y2 = map(

                int,

                box.xyxy[0]
            )


            center_x = (
                x1 + x2
            ) // 2

            center_y = (
                y1 + y2
            ) // 2


            track_id = -1

            if box.id is not None:

                track_id = int(
                    box.id[0]
                )


            # =================================================
            # PERSON
            # =================================================

            if label == "person":

                person_data = {

                    "id":
                        track_id,

                    "center":
                        (center_x, center_y),

                    "bbox":
                        (x1, y1, x2, y2),

                    "confidence":
                        confidence
                }

                persons.append(
                    person_data
                )


                cv2.rectangle(

                    frame,

                    (x1, y1),

                    (x2, y2),

                    (0, 255, 0),

                    2
                )

                cv2.putText(

                    frame,

                    f"PERSON {confidence:.2f}",

                    (x1, y1 - 8),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.5,

                    (0, 255, 0),

                    2
                )


            # =================================================
            # VEHICLES
            # =================================================

            if label in [

                "car",
                "motorcycle",
                "bus",
                "truck"

            ]:

                vehicles.append({

                    "id":
                        track_id,

                    "center":
                        (center_x, center_y),

                    "bbox":
                        (x1, y1, x2, y2),

                    "confidence":
                        confidence,

                    "label":
                        label
                })


                cv2.rectangle(

                    frame,

                    (x1, y1),

                    (x2, y2),

                    (255, 255, 0),

                    2
                )

                cv2.putText(

                    frame,

                    f"{label} {confidence:.2f}",

                    (x1, y1 - 8),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.5,

                    (255, 255, 0),

                    2
                )


    # ========================================================
    # GARBAGE DETECTION + TRACKING
    # ========================================================

    garbage_results = garbage_model.track(

        frame,

        persist=True,

        tracker="bytetrack.yaml",

        conf=GARBAGE_CONFIDENCE,

        verbose=False
    )


    garbage_verified_this_frame = False


    for result in garbage_results:

        if result.boxes is None:
            continue


        for box in result.boxes:

            confidence = float(
                box.conf[0]
            )

            if confidence < GARBAGE_CONFIDENCE:

                continue


            cls = int(
                box.cls[0]
            )

            label = (
                garbage_model.names[cls]
            )


            x1, y1, x2, y2 = map(

                int,

                box.xyxy[0]
            )


            center_x = (
                x1 + x2
            ) // 2

            center_y = (
                y1 + y2
            ) // 2


            garbage_center = (
                center_x,
                center_y
            )


            # =================================================
            # TRACK ID
            # =================================================

            if box.id is not None:

                garbage_id = int(
                    box.id[0]
                )

            else:

                # fallback
                garbage_id = (
                    center_x * 10000
                    +
                    center_y
                )


            # =================================================
            # PROCESS PERSON-GARBAGE RELATION
            # =================================================

            (

                littering_verified,

                state,

                closest_person,

                closest_distance

            ) = process_garbage_relation(

                garbage_id,

                garbage_center,

                confidence,

                persons,

                frame_width
            )


            # =================================================
            # DRAW GARBAGE BOX
            # =================================================

            cv2.rectangle(

                frame,

                (x1, y1),

                (x2, y2),

                (0, 255, 255),

                2
            )


            cv2.putText(

                frame,

                f"{label} {confidence:.2f}",

                (x1, y1 - 10),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.5,

                (0, 255, 255),

                2
            )


            # =================================================
            # DRAW PERSON-GARBAGE CONNECTION
            # =================================================

            if closest_person is not None:

                px, py = (
                    closest_person["center"]
                )

                cv2.line(

                    frame,

                    garbage_center,

                    (px, py),

                    (255, 0, 255),

                    2
                )


                cv2.putText(

                    frame,

                    f"D:{int(closest_distance)}",

                    (
                        center_x,
                        center_y + 20
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.45,

                    (255, 0, 255),

                    1
                )


            # =================================================
            # DISPLAY VERIFICATION STATUS
            # =================================================

            if state[
                "association_frames"
            ] < MIN_ASSOCIATION_FRAMES:

                status = "SEARCHING PERSON"

            elif state[
                "separation_frames"
            ] < MIN_SEPARATION_FRAMES:

                status = "PERSON-GARBAGE ASSOCIATED"

            elif state[
                "stationary_frames"
            ] < MIN_STATIONARY_FRAMES:

                status = "GARBAGE SEPARATED"

            else:

                status = "LITTERING VERIFIED"


            cv2.putText(

                frame,

                status,

                (
                    x1,
                    y2 + 20
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.5,

                (0, 255, 255),

                2
            )


            # =================================================
            # FINAL LITTERING EVENT
            # =================================================

            if littering_verified:

                garbage_verified_this_frame = True


                # Prevent repeated evidence
                if (

                    time.time()
                    -
                    state[
                        "last_event_time"
                    ]

                    >=
                    EVIDENCE_COOLDOWN

                ):

                    state[
                        "last_event_time"
                    ] = time.time()

                    state[
                        "verified"
                    ] = True


                    # ------------------------------------------------
                    # RISK SCORE
                    # ------------------------------------------------

                    avg_confidence = sum(

                        state[
                            "confidence_history"
                        ]

                    ) / max(

                        len(
                            state[
                                "confidence_history"
                            ]
                        ),

                        1
                    )


                    context_score = min(

                        state[
                            "association_frames"
                        ] / 20.0,

                        1.0
                    )


                    risk = calculate_risk(

                        avg_confidence,

                        state[
                            "stationary_frames"
                        ],

                        context_score
                    )


                    # ------------------------------------------------
                    # SAVE EVIDENCE
                    # ------------------------------------------------

                    timestamp = (
                        datetime.now()
                        .strftime(
                            "%Y%m%d_%H%M%S_%f"
                        )
                    )


                    filename = (

                        f"evidence/garbage/"
                        f"{timestamp}.jpg"
                    )


                    cv2.imwrite(

                        filename,

                        frame
                    )


                    # ------------------------------------------------
                    # LOG
                    # ------------------------------------------------

                    log_alert(

                        "LITTERING",

                        avg_confidence,

                        risk,

                        filename,

                        extra=(

                            f"garbage_id={garbage_id} | "

                            f"association_frames="
                            f"{state['association_frames']} | "

                            f"separation_frames="
                            f"{state['separation_frames']} | "

                            f"stationary_frames="
                            f"{state['stationary_frames']}"
                        )
                    )


                    save_history(

                        "LITTERING",

                        avg_confidence,

                        risk,

                        filename
                    )


                    cv2.putText(

                        frame,

                        f"LITTERING VERIFIED | RISK:{risk}",

                        (30, 50),

                        cv2.FONT_HERSHEY_SIMPLEX,

                        0.8,

                        (0, 0, 255),

                        3
                    )


    # ========================================================
    # TRESPASSING
    # ========================================================

    trespassing = False

    trespass_confidence = 0


    for person in persons:

        cx, cy = person[
            "center"
        ]


        inside_zone = (

            T_X1 < cx < T_X2

            and

            T_Y1 < cy < T_Y2
        )


        if inside_zone:

            trespassing = True

            trespass_confidence = max(

                trespass_confidence,

                person["confidence"]
            )


    if trespassing:

        trespass_counter += 1

    else:

        trespass_counter = 0


    if trespass_counter >= VERIFY_FRAMES:

        cv2.putText(

            frame,

            "TRESPASSING VERIFIED",

            (30, 90),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.8,

            (0, 0, 255),

            3
        )


        current_time = time.time()


        if (

            current_time
            -
            last_trespass_save
            >=
            EVIDENCE_COOLDOWN

        ):

            risk = calculate_risk(

                trespass_confidence,

                trespass_counter,

                1.0
            )


            timestamp = (

                datetime.now()
                .strftime(
                    "%Y%m%d_%H%M%S_%f"
                )
            )


            filename = (

                f"evidence/trespassing/"
                f"{timestamp}.jpg"
            )


            cv2.imwrite(

                filename,

                frame
            )


            log_alert(

                "TRESPASSING",

                trespass_confidence,

                risk,

                filename,

                extra=(
                    f"persistence="
                    f"{trespass_counter}"
                )
            )


            save_history(

                "TRESPASSING",

                trespass_confidence,

                risk,

                filename
            )


            last_trespass_save = (
                current_time
            )


    # ========================================================
    # NO PARKING
    # ========================================================

    noparking = False

    nopark_confidence = 0


    for vehicle in vehicles:

        cx, cy = vehicle[
            "center"
        ]


        inside_zone = (

            P_X1 < cx < P_X2

            and

            P_Y1 < cy < P_Y2
        )


        if inside_zone:

            noparking = True

            nopark_confidence = max(

                nopark_confidence,

                vehicle[
                    "confidence"
                ]
            )


    if noparking:

        nopark_counter += 1

    else:

        nopark_counter = 0


    if nopark_counter >= VERIFY_FRAMES:

        cv2.putText(

            frame,

            "NO PARKING VERIFIED",

            (30, 130),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.8,

            (255, 0, 0),

            3
        )


        current_time = time.time()


        if (

            current_time
            -
            last_nopark_save
            >=
            EVIDENCE_COOLDOWN

        ):

            risk = calculate_risk(

                nopark_confidence,

                nopark_counter,

                1.0
            )


            timestamp = (

                datetime.now()
                .strftime(
                    "%Y%m%d_%H%M%S_%f"
                )
            )


            filename = (

                f"evidence/noparking/"
                f"{timestamp}.jpg"
            )


            cv2.imwrite(

                filename,

                frame
            )


            log_alert(

                "NO PARKING",

                nopark_confidence,

                risk,

                filename,

                extra=(
                    f"persistence="
                    f"{nopark_counter}"
                )
            )


            save_history(

                "NO PARKING",

                nopark_confidence,

                risk,

                filename
            )


            last_nopark_save = (
                current_time
            )


    # ========================================================
    # SYSTEM STATUS
    # ========================================================

    cv2.putText(

        frame,

        "CMCVF MONITORING",

        (20, frame_height - 20),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.7,

        (255, 255, 255),

        2
    )


    cv2.putText(

        frame,

        f"Persons: {len(persons)}",

        (20, frame_height - 50),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.55,

        (255, 255, 255),

        2
    )


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(

        "AI Civic Monitoring System - Adaptive",

        frame
    )


    key = cv2.waitKey(1) & 0xFF


    if key == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

print(
    "AI Civic Monitoring System stopped."
)