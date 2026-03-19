# thor_server.py
# AI2-THOR server with real text-based skill primitives.
# Run in ANY conda env (no separate env needed):
#   pip install ai2thor pyzmq
#   python thor_server.py

import json
import base64
import math

import zmq
import numpy as np
from ai2thor.controller import Controller


# ── Scenes per room type ──
ROOM_SCENES = {
    "kitchen":     ["FloorPlan1",  "FloorPlan2",  "FloorPlan3",  "FloorPlan4"],
    "living_room": ["FloorPlan201","FloorPlan202","FloorPlan203","FloorPlan204"],
    "bedroom":     ["FloorPlan301","FloorPlan302","FloorPlan303","FloorPlan304"],
    "bathroom":    ["FloorPlan401","FloorPlan402","FloorPlan403","FloorPlan404"],
}

# ── Task → scene mapping ──
TASK_SCENES = {
    "CoffeeSetupMug":            "FloorPlan1",
    "TurnOnStove":               "FloorPlan2",
    "BoilPot":                   "FloorPlan3",
    "TurnOnMicrowave":           "FloorPlan1",
    "OpenFridge":                "FloorPlan1",
    "PickPlaceCounterToCabinet": "FloorPlan2",
    "WashDishes":                "FloorPlan3",
    "WatchTV":                   "FloorPlan201",
    "ReadBook":                  "FloorPlan201",
    "GoToSleep":                 "FloorPlan301",
    "BrushTeeth":                "FloorPlan401",
}


def make_controller(scene: str) -> Controller:
    return Controller(
        scene=scene,
        agentMode="default",
        visibilityDistance=1.5,
        gridSize=0.25,
        snapToGrid=True,
        renderDepthImage=False,
        renderInstanceSegmentation=False,
        width=640,
        height=480,
        fieldOfView=90,
    )


# ══════════════════════════════════════════════════════
# SKILL PRIMITIVES
# Each skill returns (success, message, event)
# ══════════════════════════════════════════════════════

def _find_object(controller: Controller, object_type: str) -> dict | None:
    """Find nearest visible object by type."""
    event = controller.step("Pass")
    candidates = [
        o for o in event.metadata["objects"]
        if o["objectType"].lower() == object_type.lower().replace(" ", "")
        or object_type.lower() in o["objectType"].lower()
    ]
    if not candidates:
        # try partial match
        candidates = [
            o for o in event.metadata["objects"]
            if object_type.lower() in o["objectId"].lower()
        ]
    if not candidates:
        return None
    return min(candidates, key=lambda x: x["distance"])


def _navigate_to(controller: Controller, object_id: str) -> tuple[bool, str]:
    """Teleport agent to best interactable position for object."""
    event = controller.step(
        action="GetInteractablePoses",
        objectId=object_id,
        horizons=[0, 30],
        standings=[True],
    )
    poses = event.metadata.get("actionReturn", [])
    if not poses:
        return False, f"No interactable pose for {object_id}"

    agent = event.metadata["agent"]["position"]
    best  = min(poses, key=lambda p: (
        (p["x"] - agent["x"])**2 + (p["z"] - agent["z"])**2
    ))

    event = controller.step(
        action="TeleportFull",
        position={"x": best["x"], "y": best["y"], "z": best["z"]},
        rotation={"x": 0, "y": best["rotation"], "z": 0},
        horizon=best["horizon"],
        standing=True,
    )
    if event.metadata["lastActionSuccess"]:
        return True, f"Navigated to {object_id}"
    return False, event.metadata.get("errorMessage", "Navigation failed")


def skill_navigate(controller: Controller, object_type: str, **_) -> tuple[bool, str]:
    """Navigate to nearest object of given type."""
    obj = _find_object(controller, object_type)
    if not obj:
        return False, f"Cannot find '{object_type}' in scene"
    return _navigate_to(controller, obj["objectId"])

# thor_server.py — thay hàm skill_find

def skill_find(controller: Controller, object_type: str, **_) -> tuple[bool, str]:
    """Find and face an object."""
    obj = _find_object(controller, object_type)
    if not obj:
        return False, f"Object '{object_type}' not found in scene"

    # Tính góc cần quay
    event = controller.step("Pass")
    agent = event.metadata["agent"]
    pos   = agent["position"]
    rot   = agent["rotation"]["y"]

    dx = obj["position"]["x"] - pos["x"]
    dz = obj["position"]["z"] - pos["z"]
    target_angle = math.degrees(math.atan2(dx, dz)) % 360

    # Quay đến đúng hướng bằng RotateRight/RotateLeft
    diff = (target_angle - rot + 360) % 360
    if diff > 180:
        # quay trái nhanh hơn
        steps = int((360 - diff) / 90) + 1
        for _ in range(steps):
            controller.step("RotateLeft")
    else:
        steps = int(diff / 90) + 1
        for _ in range(steps):
            controller.step("RotateRight")

    return True, f"Found {object_type} at distance {obj['distance']:.2f}m"


def skill_grab(controller: Controller, object_type: str, **_) -> tuple[bool, str]:
    """Navigate to and pick up object."""
    obj = _find_object(controller, object_type)
    if not obj:
        return False, f"Cannot find '{object_type}'"
    if not obj.get("pickupable", False):
        return False, f"'{object_type}' is not pickupable"

    nav_ok, nav_msg = _navigate_to(controller, obj["objectId"])
    if not nav_ok:
        return False, nav_msg

    event = controller.step(action="PickupObject", objectId=obj["objectId"])
    if event.metadata["lastActionSuccess"]:
        return True, f"Picked up {object_type}"
    return False, event.metadata.get("errorMessage", "Pickup failed")


def skill_place(controller: Controller, object_type: str, target: str, **_) -> tuple[bool, str]:
    """Place held object onto a receptacle."""
    recep = _find_object(controller, target)
    if not recep:
        return False, f"Cannot find receptacle '{target}'"

    nav_ok, nav_msg = _navigate_to(controller, recep["objectId"])
    if not nav_ok:
        return False, nav_msg

    # find what we're holding
    event = controller.step("Pass")
    held  = [o for o in event.metadata["objects"] if o.get("isPickedUp")]
    if not held:
        return False, "Agent is not holding anything"

    event = controller.step(
        action="PutObject",
        objectId=held[0]["objectId"],
        receptacleObjectId=recep["objectId"],
    )
    if event.metadata["lastActionSuccess"]:
        return True, f"Placed {held[0]['objectType']} on {target}"
    return False, event.metadata.get("errorMessage", "Place failed")


def skill_open(controller: Controller, object_type: str, **_) -> tuple[bool, str]:
    """Navigate to and open an object."""
    obj = _find_object(controller, object_type)
    if not obj:
        return False, f"Cannot find '{object_type}'"

    nav_ok, nav_msg = _navigate_to(controller, obj["objectId"])
    if not nav_ok:
        return False, nav_msg

    event = controller.step(action="OpenObject", objectId=obj["objectId"])
    if event.metadata["lastActionSuccess"]:
        return True, f"Opened {object_type}"
    return False, event.metadata.get("errorMessage", "Open failed")


def skill_close(controller: Controller, object_type: str, **_) -> tuple[bool, str]:
    """Navigate to and close an object."""
    obj = _find_object(controller, object_type)
    if not obj:
        return False, f"Cannot find '{object_type}'"

    nav_ok, nav_msg = _navigate_to(controller, obj["objectId"])
    if not nav_ok:
        return False, nav_msg

    event = controller.step(action="CloseObject", objectId=obj["objectId"])
    if event.metadata["lastActionSuccess"]:
        return True, f"Closed {object_type}"
    return False, event.metadata.get("errorMessage", "Close failed")


def skill_turnon(controller: Controller, object_type: str, **_) -> tuple[bool, str]:
    """Navigate to and toggle object on."""
    obj = _find_object(controller, object_type)
    if not obj:
        return False, f"Cannot find '{object_type}'"
    if not obj.get("toggleable", False):
        return False, f"'{object_type}' is not toggleable"

    nav_ok, nav_msg = _navigate_to(controller, obj["objectId"])
    if not nav_ok:
        return False, nav_msg

    event = controller.step(action="ToggleObjectOn", objectId=obj["objectId"])
    if event.metadata["lastActionSuccess"]:
        return True, f"Turned on {object_type}"
    return False, event.metadata.get("errorMessage", "Toggle on failed")


def skill_turnoff(controller: Controller, object_type: str, **_) -> tuple[bool, str]:
    """Navigate to and toggle object off."""
    obj = _find_object(controller, object_type)
    if not obj:
        return False, f"Cannot find '{object_type}'"

    nav_ok, nav_msg = _navigate_to(controller, obj["objectId"])
    if not nav_ok:
        return False, nav_msg

    event = controller.step(action="ToggleObjectOff", objectId=obj["objectId"])
    if event.metadata["lastActionSuccess"]:
        return True, f"Turned off {object_type}"
    return False, event.metadata.get("errorMessage", "Toggle off failed")


def skill_wash(controller: Controller, object_type: str, **_) -> tuple[bool, str]:
    """Grab object, navigate to sink, put it in sink."""
    grab_ok, grab_msg = skill_grab(controller, object_type)
    if not grab_ok:
        return False, grab_msg

    sink = _find_object(controller, "Sink")
    if not sink:
        sink = _find_object(controller, "SinkBasin")
    if not sink:
        return False, "Cannot find sink in scene"

    nav_ok, nav_msg = _navigate_to(controller, sink["objectId"])
    if not nav_ok:
        return False, nav_msg

    event = controller.step("Pass")
    held  = [o for o in event.metadata["objects"] if o.get("isPickedUp")]
    if held:
        controller.step(
            action="PutObject",
            objectId=held[0]["objectId"],
            receptacleObjectId=sink["objectId"],
        )
    return True, f"Washed {object_type} in sink"


def skill_sit(controller: Controller, object_type: str, **_) -> tuple[bool, str]:
    """Navigate to furniture and sit."""
    obj = _find_object(controller, object_type)
    if not obj:
        return False, f"Cannot find '{object_type}'"
    nav_ok, nav_msg = _navigate_to(controller, obj["objectId"])
    if not nav_ok:
        return False, nav_msg
    return True, f"Agent is sitting on {object_type}"


def skill_wait(controller: Controller, **_) -> tuple[bool, str]:
    """Hold position — do nothing."""
    controller.step("Pass")
    return True, "Agent waited"


# ── Skill registry ──
SKILLS = {
    "Navigate": skill_navigate,
    "Find":     skill_find,
    "Grab":     skill_grab,
    "Place":    skill_place,
    "PutIn":    skill_place,   # alias
    "Open":     skill_open,
    "Close":    skill_close,
    "TurnOn":   skill_turnon,
    "TurnOff":  skill_turnoff,
    "Wash":     skill_wash,
    "Sit":      skill_sit,
    "LieOn":    skill_sit,     # alias
    "Serve":    skill_place,   # alias
    "Wait":     skill_wait,
}


# ══════════════════════════════════════════════════════
# OBSERVATION BUILDER
# ══════════════════════════════════════════════════════

def build_obs(event) -> str:
    """Build human-readable observation from AI2-THOR event."""
    agent = event.metadata["agent"]
    pos   = agent["position"]

    lines = [
        f"Location: ({pos['x']:.2f}, {pos['z']:.2f}), facing {agent['rotation']['y']:.0f}°",
    ]

    # held objects
    held = [o for o in event.metadata["objects"] if o.get("isPickedUp")]
    if held:
        lines.append(f"Holding: {', '.join(o['objectType'] for o in held)}")
    else:
        lines.append("Holding: nothing")

    # nearby objects (within 2m)
    nearby = sorted(
        [o for o in event.metadata["objects"] if o["distance"] < 2.0],
        key=lambda x: x["distance"]
    )[:8]
    if nearby:
        lines.append("Nearby objects:")
        for o in nearby:
            state_parts = []
            if o.get("isOpen")       is not None: state_parts.append("open" if o["isOpen"] else "closed")
            if o.get("isToggled")    is not None: state_parts.append("on" if o["isToggled"] else "off")
            if o.get("isPickedUp"):               state_parts.append("held")
            state = f" [{', '.join(state_parts)}]" if state_parts else ""
            lines.append(f"  - {o['objectType']}{state} ({o['distance']:.1f}m)")

    if not event.metadata["lastActionSuccess"]:
        lines.append(f"Last action failed: {event.metadata.get('errorMessage', '')}")

    return "\n".join(lines)


def get_visible_objects(event) -> list[str]:
    """Return list of visible object types for LLM context."""
    seen = set()
    result = []
    for o in event.metadata["objects"]:
        if o["visible"] and o["objectType"] not in seen:
            seen.add(o["objectType"])
            result.append(o["objectType"])
    return sorted(result)


# ══════════════════════════════════════════════════════
# SERVER
# ══════════════════════════════════════════════════════

def to_python(obj):
    """Recursively convert numpy types → native Python."""
    if isinstance(obj, np.ndarray):    return obj.tolist()
    if isinstance(obj, np.integer):    return int(obj)
    if isinstance(obj, np.floating):   return float(obj)
    if isinstance(obj, np.bool_):      return bool(obj)
    if isinstance(obj, dict):          return {k: to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [to_python(v) for v in obj]
    return obj


class ThorServer:
    def __init__(self, port: int = 5555):
        self.controller   = None
        self.current_task = None
        self.step_count   = 0
        self.total_reward = 0.0

        self.ctx    = zmq.Context()
        self.socket = self.ctx.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{port}")
        print(f"[ThorServer] Listening on port {port}")
        print(f"[ThorServer] Available skills: {list(SKILLS.keys())}")

    def handle(self, msg: dict) -> dict:
        cmd = msg.get("cmd")

        # ── ping ──
        if cmd == "ping":
            return {"status": "ok"}

        # ── reset ──
        if cmd == "reset":
            task  = msg.get("task", "FloorPlan1")
            scene = TASK_SCENES.get(task, task)  # fallback: use task as scene name

            try:
                if self.controller:
                    self.controller.stop()

                print(f"[ThorServer] Loading scene: {scene} (task: {task})")
                self.controller   = make_controller(scene)
                self.current_task = task
                self.step_count   = 0
                self.total_reward = 0.0

                event = self.controller.step("Pass")
                obs   = build_obs(event)
                vis   = get_visible_objects(event)

                return {
                    "status":            "ok",
                    "obs":               obs,
                    "visible_objects":   vis,
                    "scene":             scene,
                    "available_actions": list(SKILLS.keys()),
                }
            except Exception as e:
                import traceback
                return {"status": "error", "msg": str(e), "trace": traceback.format_exc()}

        # ── step ──
        if cmd == "step":
            if self.controller is None:
                return {"status": "error", "msg": "Call reset first"}

            action      = msg.get("action", "Wait")
            object_type = msg.get("object", "")
            target      = msg.get("target", "")

            skill_fn = SKILLS.get(action)
            if not skill_fn:
                return {
                    "status":  "error",
                    "msg":     f"Unknown action '{action}'. Available: {list(SKILLS.keys())}",
                    "obs":     build_obs(self.controller.step("Pass")),
                    "success": False,
                    "reward":  0.0,
                    "done":    False,
                }

            try:
                print(f"[ThorServer] Executing: {action} {object_type}"
                      + (f" → {target}" if target else ""))

                success, message = skill_fn(
                    self.controller,
                    object_type=object_type,
                    target=target,
                )

                event  = self.controller.step("Pass")
                obs    = build_obs(event)
                reward = 1.0 if success else 0.0
                self.total_reward += reward
                self.step_count   += 1

                print(f"[ThorServer] {'OK' if success else 'FAIL'}: {message}")

                return {
                    "status":       "ok",
                    "obs":          obs,
                    "success":      success,
                    "reward":       reward,
                    "total_reward": self.total_reward,
                    "done":         False,
                    "msg":          message,
                    "step_count":   self.step_count,
                    "visible_objects": get_visible_objects(event),
                }

            except Exception as e:
                import traceback
                print(f"[ThorServer] Exception: {e}")
                return {
                    "status":  "error",
                    "msg":     str(e),
                    "obs":     "",
                    "success": False,
                    "reward":  0.0,
                    "done":    False,
                }

        # ── get_frame ──
        if cmd == "get_frame":
            if self.controller is None:
                return {"status": "error", "msg": "no controller"}
            try:
                import cv2
                event = self.controller.step("Pass")
                frame = event.frame  # RGB numpy array (H, W, 3)
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                _, buf = cv2.imencode(".jpg", frame_bgr,
                                      [cv2.IMWRITE_JPEG_QUALITY, 85])
                return {
                    "status":       "ok",
                    "frame":        base64.b64encode(buf).decode(),
                    "total_reward": self.total_reward,
                    "step_count":   self.step_count,
                }
            except Exception as e:
                return {"status": "error", "msg": str(e)}


        # ── get_objects ──
        if cmd == "get_objects":
            if self.controller is None:
                return {"status": "error", "msg": "no controller"}
            event = self.controller.step("Pass")
            objects = []
            for o in event.metadata["objects"]:
                objects.append({
                    "objectType":  o["objectType"],
                    "objectId":    o["objectId"],
                    "visible":     o.get("visible", False),
                    "pickupable":  o.get("pickupable", False),
                    "openable":    o.get("openable", False),
                    "toggleable":  o.get("toggleable", False),
                    "receptacle":  o.get("receptacle", False),
                    "isOpen":      o.get("isOpen", False),
                    "isToggled":   o.get("isToggled", False),
                    "distance":    round(o.get("distance", 999.0), 2),
                    "position":    o.get("position", {}),
                })
            return {
                "status":  "ok",
                "objects": objects,
                "scene":   self.current_task,
                "obs":     build_obs(event),
                "visible_objects": get_visible_objects(event),
            }

        # ── get_state ──
        if cmd == "get_state":
            if self.controller is None:
                return {"status": "error", "msg": "no controller"}
            event = self.controller.step("Pass")
            agent = event.metadata["agent"]
            held  = [o["objectType"] for o in event.metadata["objects"]
                     if o.get("isPickedUp")]
            return {
                "status":       "ok",
                "position":     agent["position"],
                "rotation":     agent["rotation"],
                "held_objects": held,
                "step_count":   self.step_count,
                "total_reward": self.total_reward,
            }

        return {"status": "error", "msg": f"Unknown command: {cmd}"}

    def run(self):
        print("[ThorServer] Ready. Waiting for requests...")
        while True:
            try:
                raw  = self.socket.recv_string()
                msg  = json.loads(raw)
                resp = self.handle(msg)
                self.socket.send_string(json.dumps(to_python(resp)))
            except KeyboardInterrupt:
                print("\n[ThorServer] Shutting down.")
                break
            except Exception as e:
                print(f"[ThorServer] Unhandled: {e}")
                try:
                    self.socket.send_string(json.dumps({
                        "status": "error", "msg": str(e)
                    }))
                except Exception:
                    pass

        if self.controller:
            self.controller.stop()
        self.ctx.destroy()


if __name__ == "__main__":
    ThorServer(port=5555).run()