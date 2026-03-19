# knowledge.py — updated for AI2-THOR

# ── AI2-THOR scenes grouped by room ──
THOR_SCENES = {
    "kitchen":     [f"FloorPlan{i}"  for i in range(1,   31)],
    "living_room": [f"FloorPlan{i}"  for i in range(201, 231)],
    "bedroom":     [f"FloorPlan{i}"  for i in range(301, 331)],
    "bathroom":    [f"FloorPlan{i}"  for i in range(401, 431)],
}

# ── Tasks: human label → scene + description ──
TASKS = {
    "☕ Make coffee":           {"scene": "FloorPlan1",  "desc": "Make a cup of coffee using the coffee machine"},
    "☕ Boil water":            {"scene": "FloorPlan2",  "desc": "Boil water in a pot on the stove"},
    "🍳 Turn on stove":         {"scene": "FloorPlan2",  "desc": "Turn on the stove burner"},
    "🍳 Cook an egg":           {"scene": "FloorPlan3",  "desc": "Pick up an egg and place it in a pan on the stove"},
    "🧊 Get item from fridge":  {"scene": "FloorPlan1",  "desc": "Open the fridge and retrieve an item"},
    "🍽️ Set up dishes":         {"scene": "FloorPlan4",  "desc": "Place plates and cups on the dining table"},
    "🚿 Wash an apple":         {"scene": "FloorPlan1",  "desc": "Pick up an apple and wash it in the sink"},
    "📡 Heat food in microwave":{"scene": "FloorPlan1",  "desc": "Place food in microwave and turn it on"},
    "📺 Watch TV":              {"scene": "FloorPlan201","desc": "Turn on the television and sit on the sofa"},
    "📚 Read a book":           {"scene": "FloorPlan201","desc": "Pick up a book and sit on the sofa to read"},
    "🧹 Clean living room":     {"scene": "FloorPlan202","desc": "Pick up clutter and put items in their place"},
    "😴 Go to sleep":           {"scene": "FloorPlan301","desc": "Turn off the light and lie on the bed"},
    "⏰ Set alarm clock":        {"scene": "FloorPlan301","desc": "Pick up and interact with the alarm clock"},
    "👕 Get clothes":            {"scene": "FloorPlan302","desc": "Open the dresser and pick up clothes"},
    "🪥 Brush teeth":           {"scene": "FloorPlan401","desc": "Pick up toothbrush, use sink, brush teeth"},
    "🧴 Wash hands":            {"scene": "FloorPlan401","desc": "Turn on sink faucet and wash hands with soap"},
    "🚿 Turn on shower":        {"scene": "FloorPlan402","desc": "Turn on the shower faucet"},
}

CATEGORIES = {
    "☕ Kitchen":     [k for k in TASKS if any(e in k for e in ["☕","🍳","🧊","🍽️","🚿 Wash","📡"])],
    "📺 Living room": [k for k in TASKS if any(e in k for e in ["📺","📚","🧹"])],
    "😴 Bedroom":    [k for k in TASKS if any(e in k for e in ["😴","⏰","👕"])],
    "🪥 Bathroom":   [k for k in TASKS if any(e in k for e in ["🪥","🧴","🚿 Turn"])],
}

# ── Keyword → task label for free-text matching ──
KEYWORD_MAP = {
    "coffee":     "☕ Make coffee",
    "boil":       "☕ Boil water",
    "stove":      "🍳 Turn on stove",
    "egg":        "🍳 Cook an egg",
    "fridge":     "🧊 Get item from fridge",
    "dish":       "🍽️ Set up dishes",
    "apple":      "🚿 Wash an apple",
    "microwave":  "📡 Heat food in microwave",
    "tv":         "📺 Watch TV",
    "television": "📺 Watch TV",
    "book":       "📚 Read a book",
    "read":       "📚 Read a book",
    "clean":      "🧹 Clean living room",
    "sleep":      "😴 Go to sleep",
    "bed":        "😴 Go to sleep",
    "alarm":      "⏰ Set alarm clock",
    "clothes":    "👕 Get clothes",
    "brush":      "🪥 Brush teeth",
    "teeth":      "🪥 Brush teeth",
    "hands":      "🧴 Wash hands",
    "shower":     "🚿 Turn on shower",
}

# ── Skill actions for LLM prompt ──
ROBOT_ACTIONS = [
    "Navigate",  # Move to object
    "Find",      # Locate and face object
    "Grab",      # Pick up object
    "Place",     # Put object on surface
    "PutIn",     # Put object inside container
    "Open",      # Open container/door/appliance
    "Close",     # Close container/door/appliance
    "TurnOn",    # Switch on appliance
    "TurnOff",   # Switch off appliance
    "Wash",      # Wash object at sink
    "Sit",       # Sit on furniture
    "LieOn",     # Lie on bed/sofa
    "Serve",     # Place food on serving surface
    "Wait",      # Hold position
]


def match_task_from_text(text: str) -> str:
    text_lower = text.lower()
    for kw, label in KEYWORD_MAP.items():
        if kw in text_lower:
            return label
    return "☕ Make coffee"


def get_task_info(task_label: str) -> dict:
    return TASKS.get(task_label, {"scene": "FloorPlan1", "desc": task_label})