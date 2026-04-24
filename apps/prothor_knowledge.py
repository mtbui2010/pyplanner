# procthor_knowledge.py — Knowledge base for ProcTHOR procedural environments
#
# ProcTHOR houses are randomly generated and don't have fixed FloorPlan IDs.
# Tasks are defined by description + room category only; the actual house is
# chosen at runtime from the procthor-10k dataset.

# ── ProcTHOR room categories ──────────────────────────────────────────────────
# ProcTHOR generates multi-room houses. Rooms are identified by type at runtime.
PROCTHOR_ROOM_TYPES = [
    "Kitchen",
    "LivingRoom",
    "Bedroom",
    "Bathroom",
    "DiningRoom",
    "Hallway",
    "LaundryRoom",
    "Garage",
]

# ── Tasks: human label → room category + description ─────────────────────────
# `room` is a hint for the LLM planner — the agent must navigate to that room.
PROCTHOR_TASKS = {
    # Kitchen
    "☕ Make coffee":            {"room": "Kitchen",     "desc": "Navigate to the kitchen, find and use the coffee machine to make coffee"},
    "☕ Boil water":             {"room": "Kitchen",     "desc": "Navigate to the kitchen, find a pot and boil water on the stove"},
    "🍳 Turn on stove":          {"room": "Kitchen",     "desc": "Navigate to the kitchen and turn on the stove burner"},
    "🍳 Cook an egg":            {"room": "Kitchen",     "desc": "Navigate to the kitchen, pick up an egg and place it in a pan on the stove"},
    "🧊 Get item from fridge":   {"room": "Kitchen",     "desc": "Navigate to the kitchen, open the fridge and retrieve an item"},
    "🍽️ Set up dishes":          {"room": "DiningRoom",  "desc": "Navigate to the dining room, place plates and cups on the table"},
    "🚿 Wash an apple":          {"room": "Kitchen",     "desc": "Navigate to the kitchen, pick up an apple and wash it in the sink"},
    "📡 Heat food in microwave": {"room": "Kitchen",     "desc": "Navigate to the kitchen, place food in microwave and turn it on"},
    # Living room
    "📺 Watch TV":               {"room": "LivingRoom",  "desc": "Navigate to the living room, turn on the television and sit on the sofa"},
    "📚 Read a book":            {"room": "LivingRoom",  "desc": "Navigate to the living room, pick up a book and sit on the sofa to read"},
    "🧹 Clean living room":      {"room": "LivingRoom",  "desc": "Navigate to the living room, pick up clutter and put items in their place"},
    # Bedroom
    "😴 Go to sleep":            {"room": "Bedroom",     "desc": "Navigate to the bedroom, turn off the light and lie on the bed"},
    "⏰ Set alarm clock":         {"room": "Bedroom",     "desc": "Navigate to the bedroom, pick up and interact with the alarm clock"},
    "👕 Get clothes":             {"room": "Bedroom",     "desc": "Navigate to the bedroom, open the dresser and pick up clothes"},
    # Bathroom
    "🪥 Brush teeth":            {"room": "Bathroom",    "desc": "Navigate to the bathroom, pick up toothbrush, use sink, brush teeth"},
    "🧴 Wash hands":             {"room": "Bathroom",    "desc": "Navigate to the bathroom, turn on sink faucet and wash hands with soap"},
    "🚿 Turn on shower":         {"room": "Bathroom",    "desc": "Navigate to the bathroom and turn on the shower faucet"},
}

PROCTHOR_CATEGORIES = {
    "☕ Kitchen":     [k for k in PROCTHOR_TASKS if PROCTHOR_TASKS[k]["room"] in ("Kitchen", "DiningRoom")],
    "📺 Living room": [k for k in PROCTHOR_TASKS if PROCTHOR_TASKS[k]["room"] == "LivingRoom"],
    "😴 Bedroom":    [k for k in PROCTHOR_TASKS if PROCTHOR_TASKS[k]["room"] == "Bedroom"],
    "🪥 Bathroom":   [k for k in PROCTHOR_TASKS if PROCTHOR_TASKS[k]["room"] == "Bathroom"],
}

# ── Keyword → task label ──────────────────────────────────────────────────────
PROCTHOR_KEYWORD_MAP = {
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

# ── Skill actions (same primitives as iTHOR, but agent must Navigate first) ───
PROCTHOR_ROBOT_ACTIONS = [
    "MoveTo",   # Move to object / room
    "Find",       # Locate and face object
    "Pick",       # Pick up object
    "Place",      # Put object on surface
    "PutIn",      # Put object inside container
    "Open",       # Open container/door/appliance
    "Close",      # Close container/door/appliance
    "TurnOn",     # Switch on appliance
    "TurnOff",    # Switch off appliance
    "Wash",       # Wash object at sink
    "Sit",        # Sit on furniture
    "LieOn",      # Lie on bed/sofa
    "Serve",      # Place food on serving surface
    "Wait",       # Hold position
]

# ── ProcTHOR-specific dataset splits ─────────────────────────────────────────
PROCTHOR_SPLITS = ["train", "val", "test"]
PROCTHOR_DEFAULT_SPLIT = "train"

# Approximate house counts per split (procthor-10k)
PROCTHOR_SPLIT_SIZES = {
    "train": 10_000,
    "val":   1_000,
    "test":  1_000,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def procthor_match_task_from_text(text: str) -> str:
    text_lower = text.lower()
    for kw, label in PROCTHOR_KEYWORD_MAP.items():
        if kw in text_lower:
            return label
    return "☕ Make coffee"


def procthor_get_task_info(task_label: str) -> dict:
    return PROCTHOR_TASKS.get(task_label, {"room": "Kitchen", "desc": task_label})