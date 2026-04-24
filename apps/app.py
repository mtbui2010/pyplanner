# app.py — Daily Assistant Robot (AI2-THOR + Ollama + Streamlit)
# Run: streamlit run app.py
# Requires: python thor_server.py running in another terminal
#
# Setup (run once):
#   pip install -e ../pyplanner
# Or if pip install is not possible, the path-fallback below handles it automatically.

from __future__ import annotations
import sys
import os
import time
import html

# ── Auto-resolve pyplanner package ────────────────────────────────────
# Works whether installed via pip OR run directly from the project folder.
try:
    import pyplanner
except ModuleNotFoundError:
    _here   = os.path.dirname(os.path.abspath(__file__))           # thor_app/
    _parent = os.path.dirname(_here)                                # project/
    _pkg    = os.path.join(_parent, "pyplanner")                   # project/pyplanner/
    if os.path.isdir(_pkg):
        sys.path.insert(0, _pkg)
        import pyplanner
    else:
        raise ImportError(
            "Cannot find pyplanner package.\n"
            f"Looked in: {_pkg}\n"
            "Run:  pip install -e ../pyplanner"
        )
# ──────────────────────────────────────────────────────────────────────

import streamlit as st

# ── .env helpers ──────────────────────────────────────────────────────
_ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")

def _load_env() -> dict[str, str]:
    """Parse key=value lines from .env; ignore comments and blank lines."""
    result: dict[str, str] = {}
    if not os.path.exists(_ENV_PATH):
        return result
    with open(_ENV_PATH) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            result[k.strip()] = v.strip().strip('"').strip("'")
    return result

def _save_env(data: dict[str, str]) -> None:
    """Write/update keys in .env, preserving unrelated lines."""
    existing: list[str] = []
    if os.path.exists(_ENV_PATH):
        with open(_ENV_PATH) as f:
            existing = f.readlines()
    written = set()
    out: list[str] = []
    for line in existing:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            out.append(line)
            continue
        k = stripped.split("=", 1)[0].strip()
        if k in data:
            out.append(f'{k}="{data[k]}"\n')
            written.add(k)
        else:
            out.append(line)
    for k, v in data.items():
        if k not in written:
            out.append(f'{k}="{v}"\n')
    with open(_ENV_PATH, "w") as f:
        f.writelines(out)

_env_vals = _load_env()
# Also push loaded keys into os.environ so pyplanner backends pick them up
for _k, _v in _env_vals.items():
    if _k not in os.environ:
        os.environ[_k] = _v
# ──────────────────────────────────────────────────────────────────────

try:
    # Import directly when running inside 'apps/' directory
    from thor_knowledge import TASKS, CATEGORIES, get_task_info
    from sim_client import SimClient as ThorClient
    import prothor_knowledge
except ImportError:
    # Fallback for when running from project root
    from apps.thor_knowledge import TASKS, CATEGORIES, get_task_info
    from apps.sim_client import SimClient as ThorClient
    from apps import prothor_knowledge

from pyplanner import REGISTRY, DEFAULT_HOST, PROVIDER_MODELS, DEFAULT_BACKEND

# ══════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Daily Assistant Robot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.step-pending { padding:8px 12px; background:var(--color-background-secondary);
                border-radius:8px; border-left:3px solid var(--color-border-secondary);
                margin:3px 0; font-size:14px; }
.step-active  { padding:8px 12px; background:var(--color-background-info);
                border-radius:8px; border-left:3px solid var(--color-border-info);
                margin:3px 0; font-size:14px; font-weight:500; }
.step-done    { padding:8px 12px; background:var(--color-background-success);
                border-radius:8px; border-left:3px solid var(--color-border-success);
                margin:3px 0; font-size:14px; }
.step-fail    { padding:8px 12px; background:var(--color-background-danger);
                border-radius:8px; border-left:3px solid var(--color-border-danger);
                margin:3px 0; font-size:14px; }
.obs-box      { background:var(--color-background-secondary); border-radius:8px;
                padding:10px 14px; font-family:monospace; font-size:12px;
                line-height:1.6; white-space:pre-wrap; }
# .obs-box {
#   background: var(--color-background-secondary);
#   border-radius: 8px;
#   padding: 10px 14px;
#   font-family: monospace;
#   font-size: 12px;
#   line-height: 1.6;
#   white-space: pre-wrap;   /* vẫn nên giữ */
# }
.reason-tag   { font-size:11px; color:var(--color-text-secondary);
                font-style:italic; margin-left:8px; }
</style>
""", unsafe_allow_html=True)

# ── Scene ranges per room type ──
PLAN_RANGES = {
    "Kitchen":     (1,   30),
    "Living room": (201, 230),
    "Bedroom":     (301, 330),
    "Bathroom":    (401, 430),
}

# ══════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════
DEFAULTS = {
    "plan":            [],
    "log":             [],
    "completed":       [],
    "running":         False,
    "auto_running":    False,
    "obs":             "",
    "visible_objects": [],
    "task_label":      "",
    "task_scene":      "",
    "replan_count":    0,
    "max_replan":      3,
    "client":          None,
    "last_metrics":    None,   # PlanMetrics from most recent generate_plan
    "bench_history":   [],     # list of PlanMetrics across runs for comparison
    "planner_obj":     None,   # cached BasePlanner instance
    "plan_evaluation": None,   # LLM evaluation text of current plan
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Simulator defaults (must be set before widgets AND before client creation)
if "cfg_sim_type" not in st.session_state:
    st.session_state["cfg_sim_type"] = "ProcThor"
if "cfg_proc_split" not in st.session_state:
    st.session_state["cfg_proc_split"] = "train"
if "cfg_proc_house_idx" not in st.session_state:
    st.session_state["cfg_proc_house_idx"] = 1

# Client is recreated when host/port changes (see sidebar)
# thor_server.py is unified — both Thor and ProcThor use the same port
_sim_host   = st.session_state.get("cfg_sim_host", "localhost")
_sim_port   = int(st.session_state.get("cfg_sim_port", 5555))
_sim_type   = st.session_state.get("cfg_sim_type", "ProcThor")
_client_key = f"{_sim_host}:{_sim_port}"
if st.session_state.client is None or st.session_state.get("_client_key") != _client_key:
    st.session_state.client = ThorClient(
        server_url=f"tcp://{_sim_host}:{_sim_port}",
        simulator_type=_sim_type.lower(),
    )
    st.session_state["_client_key"] = _client_key

client: ThorClient = st.session_state.client

# Sync with running server on first load (prevent reset on browser reload)
if client.connected and "_server_synced" not in st.session_state:
    try:
        # Fetch current state without resetting
        state = client.get_objects()
        if state.get("status") == "ok":
            cur_scene = state.get("scene", "")
            # Update session state to match server (widgets will pick this up)
            if cur_scene and cur_scene.startswith("FloorPlan"):
                try:
                    num = int(cur_scene.replace("FloorPlan", ""))
                    for room, (lo, hi) in PLAN_RANGES.items():
                        if lo <= num <= hi:
                            st.session_state["cfg_scene_room"] = room
                            st.session_state["cfg_scene_num"]  = num
                            st.session_state["_last_manual_scene"] = cur_scene
                            # Also restore observation
                            st.session_state.obs = state.get("obs", "")
                            st.session_state.visible_objects = state.get("visible_objects_meta", 
                                                                         state.get("visible_objects", []))
                            st.session_state.task_scene = cur_scene
                            break
                except ValueError:
                    pass
    except Exception:
        pass
    st.session_state["_server_synced"] = True

# ══════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════
def fmt_step(step: dict) -> str:
    label = f"{step.get('action', '')}  {step.get('object', '')}"
    if step.get("target"):
        label += f"  →  {step['target']}"
    return label


def resolve_obj(name: str, visible: list[dict | str] | None = None) -> str:
    """
    Map any object name variant to the real AI2-THOR objectType.
    Handles both list[str] and list[dict] (meta objects).
    Tries: exact (case-insensitive) → CamelCase → substring → reverse substring.
    Falls back to original name if no match found.

    Uses st.session_state.visible_objects when visible is not supplied.
    """
    if not name:
        return name

    candidates = visible if visible is not None else st.session_state.get("visible_objects", [])
    if not candidates:
        return name          # no scene loaded — pass through
    
    # Normalize to list of names for matching
    cand_names = [c["name"] if isinstance(c, dict) else c for c in candidates]

    key = name.lower().replace("_", "").replace(" ", "")

    # 1. Exact (case-insensitive)
    for ot in cand_names:
        if ot.lower() == key:
            return ot

    # 2. CamelCase: coffee_machine → CoffeeMachine
    camel = "".join(w.capitalize() for w in name.replace("-", "_").split("_"))
    for ot in cand_names:
        if ot.lower() == camel.lower():
            return ot

    # 3. Substring: "coffee" → "CoffeeMachine"
    matches = [ot for ot in cand_names if key in ot.lower()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        return max(matches, key=len)   # pick most specific

    # 4. Reverse substring
    for ot in cand_names:
        if ot.lower().replace("_", "") in key or key in ot.lower().replace("_", ""):
            return ot

    return name   # no match — return original


# Ordered longest-first so "dining table" is replaced before "table" could
# match anything else.  Keys are lowercase; values are CamelCase identifiers.
_COMPOUND_MAP: list[tuple[str, str]] = [
    # rooms / zones
    ("living room",      "LivingRoom"),
    ("dining room",      "DiningRoom"),
    ("bed room",         "Bedroom"),
    # furniture / surfaces
    ("dining table",     "DiningTable"),
    ("coffee table",     "CoffeeTable"),
    ("night stand",      "NightStand"),
    ("nightstand",       "NightStand"),
    ("counter top",      "CounterTop"),
    ("book shelf",       "BookShelf"),
    ("book case",        "BookShelf"),
    ("tv stand",         "TVStand"),
    ("tv table",         "TVStand"),
    ("end table",        "SideTable"),
    ("side table",       "SideTable"),
    ("arm chair",        "ArmChair"),
    # appliances / containers
    ("coffee machine",   "CoffeeMachine"),
    ("coffee maker",     "CoffeeMachine"),
    ("toaster oven",     "Microwave"),
    ("garbage can",      "GarbageCan"),
    ("trash can",        "GarbageCan"),
    ("recycling bin",    "GarbageCan"),
    ("recycle bin",      "GarbageCan"),
    ("paper towel",      "PaperTowels"),
    ("dish soap",        "DishSponge"),
    ("soap bottle",      "SoapBottle"),
    ("hand soap",        "HandTowel"),
    ("toilet paper",     "ToiletPaper"),
    # lighting
    ("floor lamp",       "FloorLamp"),
    ("desk lamp",        "DeskLamp"),
    ("table lamp",       "DeskLamp"),
    ("light switch",     "LightSwitch"),
    # objects
    ("remote control",   "RemoteControl"),
    ("tv remote",        "RemoteControl"),
    ("alarm clock",      "AlarmClock"),
    ("credit card",      "CreditCard"),
    ("key chain",        "KeyChain"),
    ("tissue box",       "TissueBox"),
    ("spray bottle",     "SprayBottle"),
    ("dish sponge",      "DishSponge"),
    ("salt shaker",      "SaltShaker"),
    ("pepper shaker",    "PepperShaker"),
    ("wine bottle",      "WineBottle"),
    ("water bottle",     "Bottle"),
    ("glass bottle",     "Bottle"),
    ("soda can",         "SodaCan"),
    ("baseball bat",     "BaseballBat"),
    ("tennis racket",    "TennisRacket"),
    ("cell phone",       "CellPhone"),
    ("mobile phone",     "CellPhone"),
    ("toilet bowl",      "Toilet"),
    ("bath tub",         "Bathtub"),
    ("bath towel",       "Towel"),
    ("hand towel",       "Towel"),
]


def _normalize_task(task: str) -> str:
    """
    Replace known multi-word household phrases with their CamelCase equivalents
    before noun extraction, so 'living room' → 'LivingRoom' is one token
    and matches the plan object exactly.
    """
    import re as _re
    t = task.lower()
    for phrase, camel in _COMPOUND_MAP:
        # whole-word boundary replacement (case-insensitive already handled by lower())
        t = _re.sub(r"\b" + _re.escape(phrase) + r"\b", camel, t)
    return t


def evaluate_plan(plan: list[dict], task: str, obs: str, planner=None) -> str:
    """
    Rules-based plan evaluator — no LLM required.

    Implements a deterministic state-machine trace checker inspired by
    Linear Temporal Logic (LTL) safety / liveness properties and Büchi
    automaton acceptance conditions (ref: "Ensuring Safety in LLM-Driven
    Robotics: A Cross-Layer Sequence Supervision Mechanism").

    Safety properties checked (must hold at every step):
      P1  G( Pick   →  found ≠ ∅ )         Find must precede Pick
      P2  G( Pick   →  ¬Holding )           Cannot pick while already holding
      P3  G( Place  →  Holding )            Must hold something to place
      P4  G( Place(X) → Arrived(X) )        Must have MoveTo X before Place X
      P5  G( TurnOn|TurnOff → found ≠ ∅ )  Must Find before TurnOn/TurnOff
      P6  G( Find(X) → X is pickupable )    Find is ONLY for pickupable objects,
                                             NOT for rooms or furniture — use MoveTo

    Liveness / pairing properties (must hold over the whole trace):
      L1  F( Open(X) → ◇ Close(X) )        Every opened container is closed
      L2  Intent coverage: every significant
          noun in the task appears in ≥ 1
          step's object field              All mentioned objects/locations planned
    """
    import re as _re

    # Normalize multi-word phrases → single CamelCase tokens before extraction
    if task.lower() != "direct commands":
        task = _normalize_task(task)

    # ── constants ───────────────────────────────────────────────────
    _CONTAINERS = {
        "fridge", "cabinet", "drawer", "microwave", "box", "chest",
        "safe", "dishwasher", "bin", "bag", "closet", "wardrobe",
    }
    # Rooms and furniture: valid targets for MoveTo, INVALID targets for Find
    _ROOMS_AND_FURNITURE = {
        "kitchen", "livingroom", "bedroom", "bathroom", "hallway", "diningroom",
        "diningtable", "coffeetable", "sofa", "couch", "bed", "desk", "dresser",
        "bookshelf", "bookcase", "tvstand", "nightstand", "sidetable", "endtable",
        "armchair", "chair", "countertop", "sink", "bathtub", "toilet", "shower",
        "fridge", "refrigerator", "oven", "stove", "microwave", "dishwasher",
        "washer", "dryer", "cabinet", "drawer", "shelf",
    }
    # Words never extracted as task nouns (articles, verbs, prepositions, etc.)
    _STOP_WORDS = {
        "a", "an", "the", "in", "on", "at", "to", "of", "and", "or",
        "from", "into", "onto", "up", "put", "get", "bring", "take",
        "place", "grab", "pick", "find", "move", "go", "turn", "it",
        "its", "this", "that", "then", "could", "you", "me", "my",
        "please", "need", "want", "is", "are", "was", "be", "do",
        "throw", "wash", "heat", "make", "set", "read",
        "there", "next", "front", "back", "can", "would",
        "near", "beside", "left", "right", "side", "dirty", "clean",
        "some", "all", "any", "for", "with", "out", "off",
    }
    # Only Find is structurally implicit — it is always required before
    # Pick/Open/etc. and its object is already covered by the surrounding steps.
    # Open/Close/TurnOn/TurnOff/MoveTo/Place all reference real scene objects
    # that MUST be grounded in the task text.
    _IMPLICIT_ACTIONS = {"Find", "Pick"}

    # ── helpers ──────────────────────────────────────────────────────
    def _norm(s: str) -> str:
        return s.lower().replace(" ", "").replace("_", "")

    def _camel_words(s: str) -> set[str]:
        """Split CamelCase/PascalCase into lowercase words.
        'DiningTable' → {'dining', 'table'}
        'CoffeeMachine' → {'coffee', 'machine'}
        'Fridge' → {'fridge'}
        """
        parts = _re.findall(r"[A-Z][a-z]*|[a-z]+|[A-Z]+(?=[A-Z]|$)", s)
        return {p.lower() for p in parts} if parts else {s.lower()}

    def _task_nouns(text: str) -> set[str]:
        """
        Extract meaningful noun tokens from the task text as a set.
        Also includes 2-word combinations to match compound names like
        'dining table' → covers 'DiningTable'.
        """
        tokens = _re.findall(r"[a-zA-Z]+", text.lower())
        tokens = [t for t in tokens if t not in _STOP_WORDS and len(t) > 2]
        return set(tokens)

    def _in_task(obj: str, task_token_set: set[str]) -> bool:
        """
        True if the plan object is grounded in the task.
        Check 1 — word split: 'Apple'→{'apple'}, 'Fridge'→{'fridge'}
        Check 2 — normalized join: 'DiningTable'→'diningtable' matches
                  compound tokens produced by _normalize_task.
        """
        if _camel_words(obj) & task_token_set:
            return True
        return _norm(obj) in task_token_set

    def _in_objects(needle: str, objects: list[str]) -> bool:
        """True if the needle (task noun) is covered by any plan object.
        Checks both raw substring and CamelCase word decomposition.
        e.g. needle='dining' covers object='DiningTable' ✓
        """
        n = needle.lower()
        for o in objects:
            if not o:
                continue
            if n in _norm(o):        # raw substring: 'dining' in 'diningtable'
                return True
            if n in _camel_words(o): # word match: 'dining' in {'dining','table'}
                return True
        return False

    # ── state machine ────────────────────────────────────────────────
    holding:  str | None       = None
    found:    str | None       = None
    arrived:  str | None       = None
    open_containers: list[str] = []        # stack of opened containers

    step_results: list[str] = []
    violations:   list[str] = []

    plan_objects = [s.get("object", "") for s in plan]

    for i, step in enumerate(plan):
        action = step.get("action", "")
        obj    = step.get("object", "")
        label  = f"Step {i+1}: {action} {obj}"
        errs   = []

        # ── P1 + P2  (Pick) ─────────────────────────────────────────
        if action == "Pick":
            if found is None:
                errs.append("P1 VIOLATED — no Find before Pick (found = ∅)")
            if holding is not None:
                errs.append(f"P2 VIOLATED — already holding '{holding}'; Place first")
            if not errs:
                holding = found
                found   = None

        # ── P3 + P4  (Place) ────────────────────────────────────────
        elif action == "Place":
            if holding is None:
                errs.append("P3 VIOLATED — nothing held; Pick an object first")
            if arrived is None:
                errs.append("P4 VIOLATED — arrived = ∅; MoveTo receptacle first")
            elif _norm(obj) and arrived and _norm(obj) not in _norm(arrived) \
                    and _norm(arrived) not in _norm(obj):
                errs.append(
                    f"P4 VIOLATED — Place '{obj}' but arrived at '{arrived}' "
                    f"(MoveTo '{obj}' missing)"
                )
            if not errs:
                # close any open containers automatically on place? No — keep tracking.
                holding = None
                arrived = None

        # ── P5  (TurnOn / TurnOff require prior Find) ───────────────
        elif action in ("TurnOn", "TurnOff"):
            if found is None:
                errs.append(f"P5 VIOLATED — no Find before {action} (found = ∅)")

        # ── Open / Close (container pair — no prior Find required) ──
        elif action in ("Open", "Close"):
            if action == "Open":
                open_containers.append(obj)
            elif action == "Close":
                norm_obj = _norm(obj)
                open_containers = [
                    c for c in open_containers if _norm(c) != norm_obj
                ]

        # ── MoveTo — updates arrived ─────────────────────────────────
        elif action == "MoveTo":
            arrived = obj

        # ── Find — updates found; P6: must be pickupable object ──────
        elif action == "Find":
            found = obj
            if _norm(obj) in _ROOMS_AND_FURNITURE:
                errs.append(
                    f"P6 VIOLATED — Find '{obj}' targets a room/furniture; "
                    f"use MoveTo instead"
                )

        # ── format step result ───────────────────────────────────────
        if errs:
            for e in errs:
                violations.append(f"{label}: {e}")
            step_results.append(f"  ❌ {label}\n" + "\n".join(f"       {e}" for e in errs))
        else:
            step_results.append(f"  ✅ {label}")

    # ── L1: liveness — unclosed containers ───────────────────────────
    liveness_issues: list[str] = []
    for c in open_containers:
        msg = f"L1 VIOLATED — container '{c}' was opened but never closed"
        liveness_issues.append(msg)
        violations.append(msg)

    # ── L2 + L3: intent coverage & hallucination check ───────────────
    is_direct = task.lower() == "direct commands"
    covered_report:     list[str] = []
    missing_report:     list[str] = []
    hallucin_report:    list[str] = []

    if is_direct:
        covered_report = ["  ✅ Direct commands — no task text to check"]
    else:
        task_noun_set = _task_nouns(task)  # set[str]

        # L2 — Missing: task noun not covered by any plan step object
        for n in sorted(task_noun_set):
            if _in_objects(n, plan_objects):
                covered_report.append(f"  ✅ '{n}'")
            else:
                missing_report.append(
                    f"  ❌ '{n}' — mentioned in task but absent from plan steps"
                )
                violations.append(
                    f"L2 VIOLATED — task noun '{n}' not covered in any step"
                )

        # L3 — Hallucination check:
        #   Every plan object (for non-implicit actions) must be grounded in
        #   the task text by CamelCase word decomposition.
        #   Implicit actions (Find, Open, Close) are exempt — they introduce
        #   containers/objects that are structurally required but may not be
        #   literally named in the task (e.g. "get milk" → "Find Fridge").
        for step in plan:
            s_action = step.get("action", "")
            s_obj    = step.get("object", "")
            if not s_obj or s_action in _IMPLICIT_ACTIONS:
                continue
            if not _in_task(s_obj, task_noun_set):
                hallucin_report.append(
                    f"  ⚠ {s_action} '{s_obj}': "
                    f"{_camel_words(s_obj)} ∩ task_nouns = ∅"
                )
                violations.append(
                    f"L3 VIOLATED — '{s_obj}' ({s_action}) not grounded in task text"
                )

    # ── assemble report ───────────────────────────────────────────────
    lines: list[str] = []
    lines.append("═" * 52)
    lines.append("  PLAN SAFETY EVALUATION  (rules-based)")
    lines.append("═" * 52)
    lines.append("")

    # Section 1 — step-by-step trace
    lines.append("── TRACE VERIFICATION (P1–P5) ──")
    lines.extend(step_results)
    if liveness_issues:
        lines.append("")
        lines.append("── LIVENESS (L1: container pairing) ──")
        for li in liveness_issues:
            lines.append(f"  ❌ {li}")
    lines.append("")

    # Section 2 — intent coverage (L2)
    lines.append("── INTENT COVERAGE (L2) ──")
    lines.append(f"  Task: \"{task}\"")
    if covered_report:
        lines.append("  Covered:")
        lines.extend(covered_report)
    if missing_report:
        lines.append("  Missing (must be added):")
        lines.extend(missing_report)
    lines.append("")

    # Section 3 — hallucination check (L3)
    if not is_direct:
        lines.append("── HALLUCINATION CHECK (L3) ──")
        if hallucin_report:
            lines.append("  Plan steps referencing objects NOT in task:")
            lines.extend(hallucin_report)
        else:
            lines.append("  ✅ All non-implicit step objects are grounded in task text")
        lines.append("")

    # Section 4 — summary
    lines.append("── SUMMARY ──")
    if violations:
        lines.append(f"  Violations found: {len(violations)}")
        for v in violations:
            lines.append(f"  • {v}")
        lines.append("")
        lines.append("  Verdict: NEEDS_REVISION")
    else:
        lines.append("  No violations found.")
        lines.append("")
        lines.append("  Verdict: VALID")

    return "\n".join(lines)


def refine_plan(plan: list[dict], task: str, obs: str, evaluation: str, planner=None) -> list[dict]:
    """
    Rules-based plan refiner — no LLM required.

    Reads the structured evaluation report produced by evaluate_plan() and
    applies targeted surgical fixes:

      Fix P1  — insert Find <obj> before each Pick that lacks one
      Fix P2  — insert Place <prev_receptacle> before a double-Pick
      Fix P3  — insert Find+Pick <something> before an orphan Place
      Fix P4  — insert MoveTo <receptacle> before Place whose arrived ≠ obj
      Fix P5  — insert Find <obj> before Open/Close/TurnOn/TurnOff without Find
      Fix L1  — append Find+Close for every unclosed container
      Fix L2  — append steps that cover task nouns missing from the plan
      Fix L3  — remove steps whose object is not mentioned in task text
                (skips implicit-action steps: Find, Open, Close)

    The fixes are applied in a single forward pass over a working copy of
    the plan so each insertion is immediately visible to subsequent checks.
    """
    import re as _re

    _STOP_WORDS = {
        "a", "an", "the", "in", "on", "at", "to", "of", "and", "or",
        "from", "into", "onto", "up", "put", "get", "bring", "take",
        "place", "grab", "pick", "find", "move", "go", "turn", "it",
        "its", "this", "that", "then", "could", "you", "me", "my",
        "please", "need", "want", "is", "are", "was", "be", "do",
        "throw", "wash", "heat", "make", "set", "read",
        "there", "next", "front", "back", "can", "would",
        "near", "beside", "left", "right", "side", "dirty", "clean",
        "some", "all", "any", "for", "with", "out", "off",
    }
    _IMPLICIT_ACTIONS = {"Find", "Pick"}

    # Normalize multi-word phrases → single CamelCase tokens before extraction
    if task.lower() != "direct commands":
        task = _normalize_task(task)

    _CONTAINERS = {
        "fridge", "cabinet", "drawer", "microwave", "box", "chest",
        "safe", "dishwasher", "bin", "bag", "closet", "wardrobe",
    }
    _ROOMS_AND_FURNITURE_NORMS = {
        "kitchen", "livingroom", "bedroom", "bathroom", "hallway", "diningroom",
        "diningtable", "coffeetable", "sofa", "couch", "bed", "desk", "dresser",
        "bookshelf", "bookcase", "tvstand", "nightstand", "sidetable", "endtable",
        "armchair", "chair", "countertop", "sink", "bathtub", "toilet", "shower",
        "fridge", "refrigerator", "oven", "stove", "microwave", "dishwasher",
        "washer", "dryer", "cabinet", "drawer", "shelf",
    }

    def _norm(s: str) -> str:
        return s.lower().replace(" ", "").replace("_", "")

    def _camel_words(s: str) -> set[str]:
        parts = _re.findall(r"[A-Z][a-z]*|[a-z]+|[A-Z]+(?=[A-Z]|$)", s)
        return {p.lower() for p in parts} if parts else {s.lower()}

    def _task_nouns(text: str) -> set[str]:
        tokens = _re.findall(r"[a-zA-Z]+", text.lower())
        return {t for t in tokens if t not in _STOP_WORDS and len(t) > 2}

    def _in_task(obj: str, task_token_set: set[str]) -> bool:
        """
        True if the plan object is grounded in the task.
        Check 1 — word split: 'Apple'→{'apple'}, 'Fridge'→{'fridge'}
        Check 2 — normalized join: 'DiningTable'→'diningtable' matches
                  compound tokens produced by _normalize_task.
        """
        if _camel_words(obj) & task_token_set:
            return True
        return _norm(obj) in task_token_set

    def _in_objects(needle: str, objects: list[str]) -> bool:
        n = needle.lower()
        for o in objects:
            if not o:
                continue
            if n in _norm(o):
                return True
            if n in _camel_words(o):
                return True
        return False

    # Reverse map: "livingroom"→"LivingRoom", "diningtable"→"DiningTable", etc.
    _NORM_TO_CAMEL: dict[str, str] = {_norm(v): v for _, v in _COMPOUND_MAP}
    # Simple room names not in compound map
    _NORM_TO_CAMEL.update({
        "kitchen": "Kitchen", "bedroom": "Bedroom",
        "bathroom": "Bathroom", "hallway": "Hallway",
    })

    def _to_camel(word: str) -> str:
        """Convert a (possibly normalized compound) token to its CamelCase form."""
        return _NORM_TO_CAMEL.get(_norm(word), word.capitalize())

    # Room-like tokens that should map to MoveTo (not Find)
    _ROOM_NORMS: set[str] = {
        _norm(v) for _, v in _COMPOUND_MAP
        if any(r in _norm(v) for r in ("room", "hall", "way"))
    } | {"kitchen", "bedroom", "bathroom", "hallway"}

    def _s(action: str, obj: str) -> dict:
        return {"action": action, "object": obj}

    # ── work on a mutable copy ───────────────────────────────────────
    refined: list[dict] = [dict(s) for s in plan]

    # ── Fix L3 first: remove hallucinated objects ────────────────────
    if task.lower() != "direct commands":
        task_nouns = _task_nouns(task)
        refined = [
            s for s in refined
            if s.get("action") in _IMPLICIT_ACTIONS
            or not s.get("object")
            or _in_task(s.get("object", ""), task_nouns)
        ]

    # ── Forward pass: fix P1–P5 ──────────────────────────────────────
    holding: str | None = None
    found:   str | None = None
    arrived: str | None = None
    open_containers: list[str] = []
    last_receptacle: str | None = None   # remember last arrived for P2 fallback

    i = 0
    MAX_ITER = len(refined) * 4 + 20   # safety cap — each step can cause at most a few insertions
    iters = 0
    while i < len(refined) and iters < MAX_ITER:
        iters += 1
        step   = refined[i]
        action = step.get("action", "")
        obj    = step.get("object", "")

        if action == "Pick":
            # P1: need Find before Pick — insert and skip past it
            if found is None:
                refined.insert(i, _s("Find", obj or "Object"))
                found = obj or "Object"   # update state immediately to avoid re-trigger
                i += 2                    # skip inserted Find + this Pick
                holding = found
                found   = None
                continue
            # P2: already holding → insert Place then re-evaluate Pick
            if holding is not None:
                recep = last_receptacle or "CounterTop"
                refined.insert(i, _s("Place", recep))
                last_receptacle = arrived
                holding = None
                arrived = None
                # do NOT advance i — re-evaluate Pick at i+1 naturally via loop
                i += 1   # skip the Place we just inserted, pick up at Pick
                continue
            holding = found
            found   = None

        elif action == "Place":
            # P3: nothing held — insert Pick for last found object, or remove orphan
            if holding is None:
                if found is not None:
                    refined.insert(i, _s("Pick", found))
                    holding = found
                    found   = None
                    i += 2   # skip Pick + Place
                    last_receptacle = arrived
                    holding = None
                    arrived = None
                    continue
                # truly orphan — remove and move on
                refined.pop(i)
                continue
            # P4: no MoveTo for receptacle — insert it
            if arrived is None or (
                obj and _norm(obj) not in _norm(arrived)
                and _norm(arrived) not in _norm(obj)
            ):
                target = obj or "CounterTop"
                refined.insert(i, _s("MoveTo", target))
                arrived = target
                i += 2   # skip MoveTo + Place
                last_receptacle = arrived
                holding = None
                arrived = None
                continue
            last_receptacle = arrived
            holding  = None
            arrived  = None

        elif action in ("TurnOn", "TurnOff"):
            # P5: need Find before TurnOn/TurnOff
            if found is None:
                refined.insert(i, _s("Find", obj or "Object"))
                found = obj or "Object"
                i += 2   # skip Find + TurnOn/TurnOff
                continue

        elif action in ("Open", "Close"):
            if action == "Open":
                open_containers.append(obj)
            else:
                norm_obj = _norm(obj)
                open_containers = [c for c in open_containers if _norm(c) != norm_obj]

        elif action == "Find":
            # P6: Find on room/furniture → replace with MoveTo
            if _norm(obj) in _ROOMS_AND_FURNITURE_NORMS:
                refined[i] = _s("MoveTo", obj)
                arrived = obj
            else:
                found = obj

        elif action == "MoveTo":
            arrived = obj

        i += 1

    # ── Fix L1: close unclosed containers ────────────────────────────
    for c in open_containers:
        refined.append(_s("Close", c))

    # ── Fix L2: cover missing task nouns ─────────────────────────────
    if task.lower() != "direct commands":
        plan_objects = [s.get("object", "") for s in refined]
        task_nouns   = _task_nouns(task)
        for noun in sorted(task_nouns):  # stable order
            if _in_objects(noun, plan_objects):
                continue
            camel = _to_camel(noun)
            # Rooms → prepend MoveTo (must navigate before anything else)
            if _norm(noun) in _ROOM_NORMS:
                refined.insert(0, _s("MoveTo", camel))
            else:
                # Object/furniture → append Find so it appears in coverage
                refined.append(_s("Find", camel))

    # ── Deduplicate consecutive identical MoveTo steps ────────────────
    deduped: list[dict] = []
    for s in refined:
        if (deduped
                and s["action"] == "MoveTo"
                and deduped[-1]["action"] == "MoveTo"
                and deduped[-1]["object"] == s["object"]):
            continue
        deduped.append(s)

    return deduped


def do_step():
    """Execute next step from plan. Trigger replan on failure."""
    if not st.session_state.plan:
        return

    step   = st.session_state.plan.pop(0)
    # Resolve LLM-generated object names to real AI2-THOR objectTypes
    resolved_obj    = resolve_obj(step.get("object", ""))
    resolved_target = resolve_obj(step.get("target", ""))
    try:
        result = client.step(
            step.get("action", "Wait"),
            resolved_obj,
            resolved_target,
        )
    except Exception as _e:
        # ZMQ timeout or simulator crash — stop gracefully
        st.session_state.log.append({
            "type": "failed",
            "reason": f"Simulator error: {_e}",
        })
        st.session_state.running      = False
        st.session_state.auto_running = False
        client.reconnect()
        return
    # Update step with resolved names for accurate logging
    step = {**step, "object": resolved_obj, "target": resolved_target}

    st.session_state.obs             = result.get("obs", st.session_state.obs)
    st.session_state.robot_state     = result.get("robot_state", result.get("obs", ""))
    st.session_state.env_state       = result.get("env_state", "")
    st.session_state.visible_objects = result.get("visible_objects_meta", result.get("visible_objects", []))
    if "found"   in result: st.session_state["robot_found"]   = result["found"]
    if "arrived" in result: st.session_state["robot_arrived"] = result["arrived"]

    st.session_state.log.append({
        "type":    "step",
        "step":    step,
        "obs":     result.get("obs", ""),
        "success": result.get("success", False),
        "reward":  result.get("reward", 0.0),
        "msg":     result.get("msg", ""),
        "done":    result.get("done", False),
    })

    if result.get("done"):
        st.session_state.running      = False
        st.session_state.auto_running = False
        st.session_state.plan         = []
        st.session_state.log.append({"type": "done"})
        return

    if result.get("success", False):
        st.session_state.completed.append(step)
    else:
        rc = st.session_state.replan_count
        if rc >= st.session_state.max_replan:
            st.session_state.running      = False
            st.session_state.auto_running = False
            st.session_state.log.append({"type": "failed", "reason": "max replans reached"})
            return

        st.session_state.replan_count += 1
        planner = st.session_state.planner_obj
        new_plan, replan_metrics = [], None
        if planner is not None:
            try:
                new_plan, replan_metrics = planner.replan(
                    st.session_state.task_label,
                    st.session_state.completed,
                    step,
                    result.get("msg", "step failed"),
                    result.get("obs", ""),
                    st.session_state.visible_objects,
                )
            except Exception as _re:
                st.session_state.log.append({
                    "type": "failed",
                    "reason": f"Replan error: {_re}",
                })
                st.session_state.running      = False
                st.session_state.auto_running = False
                return

        st.session_state.plan = new_plan
        if new_plan is not None:
            st.session_state.log.append({
                "type":    "replan",
                "steps":   new_plan.copy(),
                "n":       st.session_state.replan_count,
                "metrics": replan_metrics,
            })

    if not st.session_state.plan and st.session_state.running:
        st.session_state.running      = False
        st.session_state.auto_running = False
        st.session_state.log.append({"type": "done"})


# ══════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("🤖 Daily Assistant")
    st.caption("AI2-THOR · Local Ollama · Streamlit")

    # ── Simulator connection ──────────────────────────────────────────
    st.subheader("Simulator")

    col_h, col_p = st.columns([3, 1])
    with col_h:
        sim_host = st.text_input(
            "Host",
            value="localhost",
            key="cfg_sim_host",
            label_visibility="collapsed",
            placeholder="localhost or 192.168.x.x",
        )
    with col_p:
        sim_port = st.number_input(
            "Port",
            value=5555,
            min_value=1,
            max_value=65535,
            step=1,
            key="cfg_sim_port",
            label_visibility="collapsed",
        )

    # ── Simulator type ────────────────────────────────────────────────
    # thor_server.py is a unified server — handles both iTHOR and ProcTHOR
    # on the SAME port. We only need to change simulator_type in reset().
    sim_type = st.radio(
        "Simulator type",
        ["Thor", "RoboThor", "ProcThor"],
        horizontal=True,
        key="cfg_sim_type",
        help="iTHOR: fixed FloorPlan scenes · RoboTHOR: locobot in apartment scenes · ProcTHOR: procedural houses\n(All use the same thor_server.py)",
    )

    # Re-create client if host/port changed; mark needs_reset when sim_type switches
    _new_key = f"{sim_host}:{sim_port}"
    _prev_sim_type = st.session_state.get("_prev_sim_type_port")
    _type_switched = _prev_sim_type is not None and sim_type != _prev_sim_type
    st.session_state["_prev_sim_type_port"] = sim_type

    if st.session_state.get("_client_key") != _new_key:
        # Host/port changed — close old ZMQ context cleanly before rebuilding
        old_client = st.session_state.get("client")
        if old_client is not None:
            try:
                old_client.close()
            except Exception:
                pass
        st.session_state.client = ThorClient(
            server_url=f"tcp://{sim_host}:{int(sim_port)}",
            simulator_type=sim_type.lower(),
        )
        st.session_state["_client_key"] = _new_key
        client = st.session_state.client
        _type_switched = True   # also trigger reset for new connection

    if _type_switched:
        # Update client's simulator_type so future reset() calls use the right type
        client.simulator_type = sim_type.lower()
        st.session_state["_needs_sim_reset"] = sim_type
        st.session_state.pop("_last_manual_scene", None)

    # Status + test button
    col_status, col_test = st.columns([2, 1])
    with col_status:
        if client.connected:
            st.success("Connected", icon="🟢")
        else:
            st.error("Offline", icon="🔴")

    with col_test:
        if st.button("Ping", use_container_width=True, help="Test connection"):
            import time as _t
            t0 = _t.perf_counter()
            client.reconnect()

            latency_ms = (_t.perf_counter() - t0) * 1000
            if client.connected:
                st.toast(f"✅ Connected  ({latency_ms:.0f} ms)", icon="🟢")
            else:
                st.toast("❌ No response (check terminal)", icon="🔴")
            st.rerun()

    if not client.connected:
        st.caption(f"Run on `{sim_host}:{sim_port}`:")
        st.code("python thor_server.py", language="bash")

    # ── When sim type just changed, clear scene key so picker re-loads ──
    if st.session_state.pop("_needs_sim_reset", None):
        st.session_state.pop("_last_proc_key", None)
        st.session_state.pop("_last_manual_scene", None)
        st.session_state.pop("_last_robo_scene", None)
        st.rerun()

    st.divider()

    # ── Task selection ──
    st.subheader("Task")

    # Persist mode in session_state so switching Browse ↔ Type freely
    # doesn't reset other widgets on rerun
    if "input_mode" not in st.session_state:
        st.session_state["input_mode"] = "Type freely"   # default

    mode = st.radio(
        "input_mode",
        ["Type freely", "Browse"],
        horizontal=True,
        label_visibility="collapsed",
        key="input_mode",
    )

    if mode == "Type freely":
        # Load prompt examples
        _examples_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompt_examples")
        _examples: list[str] = []
        if os.path.exists(_examples_path):
            with open(_examples_path) as _f:
                _examples = [l.strip() for l in _f if l.strip()]

        def _on_example_select():
            sel = st.session_state.get("_example_select", "")
            if sel and not sel.startswith("—"):
                st.session_state["task_custom_text"] = sel
                st.session_state["_example_select"] = "— examples —"

        with st.form("task_form", border=False, clear_on_submit=False):
            custom = st.text_area(
                "Task description",
                placeholder="make coffee, wash hands, watch TV...",
                key="task_custom_text",
                height=68,
                label_visibility="collapsed",
            )
            _form_start = st.form_submit_button(
                "▶ Start",
                use_container_width=True,
                type="primary",
                disabled=not client.connected,
            )
        if _form_start and custom.strip():
            st.session_state["_form_start_triggered"] = True

        if _examples:
            st.selectbox(
                "examples",
                ["— examples —"] + _examples,
                key="_example_select",
                label_visibility="collapsed",
                on_change=_on_example_select,
            )

        # Show hint inline — no st.stop(), layout stays stable
        task_ready = bool(custom.strip())
        if not task_ready:
            st.caption("⬆ Type a task then press **▶ Start** or **Ctrl+Enter**.")

        task_label = custom.strip() or "_"  # placeholder keeps downstream code happy

        room_for_task = st.selectbox(
            "Room",
            ["Kitchen", "Living room", "Bedroom", "Bathroom"],
            key="task_room_select",
        )
        lo, hi = PLAN_RANGES[room_for_task]
        task_info = {
            "scene": f"FloorPlan{lo}",
            "desc":  custom.strip(),
        }
    else:
        task_ready    = True
        cat           = st.selectbox("Category", list(CATEGORIES.keys()), key="browse_cat")
        task_label    = st.selectbox("Task", CATEGORIES[cat], key="browse_task")
        task_info     = get_task_info(task_label)
        st.caption(f"_{task_info['desc']}_")

    st.divider()

    # ── Scene picker ──
    _col_scene_hdr, _col_scene_reset = st.columns([4, 1])
    _col_scene_hdr.subheader("Scene")
    reset = _col_scene_reset.button("↺", use_container_width=True, help="Reset all state and reload scene")

    # Persist scene_mode — default to Manual pick
    if "scene_mode" not in st.session_state:
        st.session_state["scene_mode"] = "Manual pick"

    scene_mode = st.radio(
        "scene_mode",
        ["Manual pick", "Auto (from task)"],
        horizontal=True,
        label_visibility="collapsed",
        key="scene_mode",
    )

    if sim_type == "RoboThor":
        # ── RoboTHOR: split + block + variation ───────────────────────
        # Scene format: FloorPlan_Train{1-12}_{1-5}  /  FloorPlan_Val{1-3}_{1-5}
        col_rs, col_rb, col_rv = st.columns(3)
        with col_rs:
            robo_split = st.selectbox("Split", ["Train", "Val"], key="cfg_robo_split")
        with col_rb:
            robo_max_block = 12 if robo_split == "Train" else 3
            robo_block = st.number_input(
                "Block",
                min_value=1,
                max_value=robo_max_block,
                value=1,
                step=1,
                key="cfg_robo_block",
                help=f"1–{robo_max_block}",
            )
        with col_rv:
            robo_var = st.number_input(
                "Var",
                min_value=1,
                max_value=5,
                value=1,
                step=1,
                key="cfg_robo_var",
                help="Variation 1–5",
            )

        final_scene  = f"FloorPlan_{robo_split}{int(robo_block)}_{int(robo_var)}"
        _robo_key    = final_scene

        if client.connected and st.session_state.get("_last_robo_scene") != _robo_key:
            st.session_state["_last_robo_scene"] = _robo_key
            with st.spinner(f"🤖 Loading `{final_scene}`..."):
                resp = client.reset(final_scene, simulator_type="robotthor")
            if resp.get("status") == "ok":
                st.session_state.obs             = resp.get("obs", "")
                st.session_state.visible_objects = resp.get("visible_objects_meta", resp.get("visible_objects", []))
                st.session_state.task_scene      = final_scene
                st.session_state["_last_frame"]  = None
                st.rerun()
            else:
                st.warning(f"Could not load `{final_scene}`: {resp.get('msg', '')}")
        else:
            loaded = st.session_state.get("task_scene") == final_scene
            st.caption(f"{'✅' if loaded else '🤖'} Scene: `{final_scene}`")

    elif sim_type == "ProcThor":
        # ── ProcTHOR: split + fixed house index ───────────────────────
        col_split, col_idx = st.columns([2, 3])
        with col_split:
            proc_split = st.selectbox(
                "Split",
                ["train", "val", "test"],
                key="cfg_proc_split",
            )
        with col_idx:
            proc_idx = st.number_input(
                "House index",
                min_value=0,
                value=1,
                step=1,
                key="cfg_proc_house_idx",
                help="Fixed house index in the selected split. Same index → same house every time.",
            )

        _proc_key = f"{proc_split}:{proc_idx}"
        if client.connected and st.session_state.get("_last_proc_key") != _proc_key:
            st.session_state["_last_proc_key"] = _proc_key
            with st.spinner(f"🏠 Loading house #{proc_idx} ({proc_split})..."):
                resp = client.reset(simulator_type="procthor", split=proc_split, house_index=int(proc_idx))
            if resp.get("status") == "ok":
                st.session_state.obs             = resp.get("obs", "")
                st.session_state.visible_objects = resp.get("visible_objects_meta", resp.get("visible_objects", []))
                st.session_state.task_scene      = resp.get("scene", f"procthor_{proc_split}_{proc_idx}")
                st.session_state["_last_frame"]  = None
                st.rerun()
            else:
                st.warning(f"Could not load house #{proc_idx}: {resp.get('msg', '')}")
        else:
            loaded = st.session_state.get("_last_proc_key") == _proc_key
            st.caption(f"{'✅' if loaded else '🏠'} House #{proc_idx} ({proc_split})")

        final_scene = st.session_state.get("task_scene", f"procthor_{proc_split}_{proc_idx}")

    elif scene_mode == "Manual pick":
        room_type = st.selectbox(
            "Room type",
            list(PLAN_RANGES.keys()),
            index=0,
            key="cfg_scene_room",
        )
        lo, hi = PLAN_RANGES[room_type]
        if "cfg_scene_num" not in st.session_state:
            st.session_state["cfg_scene_num"] = lo
        plan_num = st.slider(
            f"FloorPlan number ({lo}–{hi})",
            min_value=lo,
            max_value=hi,
            step=1,
            key="cfg_scene_num",
        )

        # Quick preset buttons
        preset_cols = st.columns(4)
        presets = [lo, lo + 5, lo + 15, hi]
        for col, n in zip(preset_cols, presets):
            if col.button(str(n), use_container_width=True, key=f"preset_{n}"):
                plan_num = n

        final_scene = f"FloorPlan{plan_num}"

        # ── Auto-load when scene changes ──
        if client.connected and st.session_state.get("_last_manual_scene") != final_scene:
            st.session_state["_last_manual_scene"] = final_scene
            with st.spinner(f"🗺 Loading `{final_scene}`..."):
                resp = client.reset(final_scene)
            if resp.get("status") == "ok":
                st.session_state.obs             = resp.get("obs", "")
                st.session_state.visible_objects = resp.get("visible_objects_meta", resp.get("visible_objects", []))
                st.session_state.task_scene      = final_scene
                st.session_state["_last_frame"]  = None
                st.rerun()
            else:
                st.warning(f"Could not load `{final_scene}`: {resp.get('msg', '')}")
        else:
            loaded = st.session_state.get("task_scene") == final_scene
            st.caption(f"{'✅' if loaded else '🗺'} Scene: `{final_scene}`")
    else:
        st.session_state.pop("_last_manual_scene", None)
        final_scene = task_info["scene"]
        st.caption(f"Auto scene: `{final_scene}`")

    st.divider()

    # ── Config ──
    st.subheader("Config")

    # All widgets in the config section use explicit key= so their values
    # are stored in session_state and survive reruns from other widgets.

    # ── Planning method ──
    method_names    = list(REGISTRY.keys())
    selected_method = st.selectbox(
        "🧩 Planning method",
        method_names,
        index=0,
        key="cfg_method",
        help="Algorithm used to generate the action plan.",
    )
    st.caption(f"_{REGISTRY[selected_method].description}_")

    # Method-specific options
    react_max_steps    = 15
    refine_iterations  = 2
    router_backend     = "openai"
    router_model       = ""
    router_openai_key  = ""
    router_gemini_key  = ""

    if selected_method == "ReAct":
        react_max_steps = st.slider("Max steps (ReAct)", 5, 20, 15, key="cfg_react_steps")
    elif selected_method == "Self-Refine":
        refine_iterations = st.slider("Refine iterations", 1, 4, 2, key="cfg_refine_iter")
    elif selected_method == "LLM Router":
        router_backend = st.radio(
            "Verifier backend",
            ["openai", "gemini"],
            horizontal=True,
            key="cfg_router_backend",
            help="Which external API verifies/fixes the local plan.",
        )
        default_vm = "gpt-4o-mini" if router_backend == "openai" else "gemini-2.5-flash"
        router_model = st.text_input(
            "Verifier model (optional)",
            placeholder=default_vm,
            key="cfg_router_model",
            help=f"Leave empty to use default: {default_vm}",
        )
        if router_backend == "openai":
            router_openai_key = st.text_input(
                "OpenAI API key",
                type="password",
                key="cfg_router_oai_key",
                help="Or set OPENAI_API_KEY env variable.",
            )
        else:
            router_gemini_key = st.text_input(
                "Gemini API key",
                type="password",
                key="cfg_router_gemini_key",
                help="Or set GEMINI_API_KEY env variable.",
            )
        st.caption("💡 Local model generates; external API only verifies (cheaper).")

    # ── LLM Provider ──────────────────────────────────────────────────
    st.markdown("**LLM Provider**")

    selected_provider = st.selectbox(
        "Provider",
        ["gemini", "ollama", "openai"],
        index=0,
        key="cfg_provider",
        help="ollama = local server · openai = ChatGPT API · gemini = Google Gemini API",
        label_visibility="collapsed",
    )

    # Host URL — only shown for Ollama
    if selected_provider == "ollama":
        ollama_host = st.text_input(
            "🌐 Ollama host URL",
            value=DEFAULT_HOST,
            placeholder="http://localhost:11434",
            key="cfg_ollama_host",
            help="URL of the Ollama server (local or remote).",
        )
    else:
        ollama_host = st.session_state.get("cfg_ollama_host", DEFAULT_HOST)

    # Model selector — dynamic list per provider
    model_list     = PROVIDER_MODELS[selected_provider]
    selected_model = st.selectbox(
        "🧠 Model",
        model_list,
        index=0,
        key="cfg_model",
        help=f"Model for the {selected_provider} provider.",
    )
    # Custom model override — persists independently
    custom_model = st.text_input(
        "Custom model name (optional)",
        placeholder=f"e.g. {model_list[0]}",
        key="cfg_custom_model",
        help="Override the dropdown — type any model name supported by the provider.",
        label_visibility="visible",
    )
    if custom_model.strip():
        selected_model = custom_model.strip()

    # API key — shown for non-Ollama providers
    planner_api_key = ""
    _env_key_name   = None
    if selected_provider == "openai":
        _env_key_name   = "OPENAI_API_KEY"
        _default_key    = _env_vals.get(_env_key_name, os.environ.get(_env_key_name, ""))
        if "cfg_oai_key" not in st.session_state:
            st.session_state["cfg_oai_key"] = _default_key
        planner_api_key = st.text_input(
            "🔑 OpenAI API key",
            type="password",
            key="cfg_oai_key",
            help="Starts with sk-... · Saved to apps/.env",
        )
    elif selected_provider == "gemini":
        _env_key_name   = "GEMINI_API_KEY"
        _default_key    = _env_vals.get(_env_key_name, os.environ.get(_env_key_name, ""))
        if "cfg_gemini_key" not in st.session_state:
            st.session_state["cfg_gemini_key"] = _default_key
        planner_api_key = st.text_input(
            "🔑 Gemini API key",
            type="password",
            key="cfg_gemini_key",
            help="Saved to apps/.env",
        )

    if _env_key_name and planner_api_key:
        if st.button("💾 Save key", key="btn_save_api_key", use_container_width=True):
            _save_env({_env_key_name: planner_api_key})
            os.environ[_env_key_name] = planner_api_key
            _env_vals[_env_key_name]  = planner_api_key
            st.toast(f"{_env_key_name} saved to .env", icon="✅")

    st.session_state["planner_model"]    = selected_model
    st.session_state["planner_provider"] = selected_provider

    max_replan = st.slider("Max replans", 1, 5, 3, key="cfg_max_replan")
    st.session_state.max_replan = max_replan

    step_delay = st.slider("Auto-step delay (s)", 0.3, 3.0, 0.8, 0.1, key="cfg_step_delay")

    auto_exec = st.toggle("Auto-execute all steps", value=False, key="cfg_auto_exec")

    st.divider()

    # ── Start / Reset buttons ──
    _form_triggered = st.session_state.get("_form_start_triggered", False)
    if _form_triggered:
        del st.session_state["_form_start_triggered"]

    if mode == "Type freely":
        # Start button is the form_submit_button above; no duplicate here
        start = _form_triggered
    else:
        start = st.button(
            "▶ Start",
            type="primary",
            use_container_width=True,
            disabled=not client.connected or not task_ready,
            help="Generate action plan and begin execution",
        )

    if start:
        # reset plan/log state
        for k in ["plan", "log", "completed"]:
            st.session_state[k] = []
        st.session_state.replan_count  = 0
        st.session_state.running       = True
        st.session_state.auto_running  = auto_exec
        st.session_state.task_label    = task_label
        st.session_state.task_scene    = final_scene

        # Instantiate planner with all kwargs
        kwargs = {}
        if selected_method == "ReAct":
            kwargs["max_steps"] = react_max_steps
        elif selected_method == "Self-Refine":
            kwargs["max_iterations"] = refine_iterations
        elif selected_method == "LLM Router":
            kwargs["verifier_backend"]   = router_backend
            kwargs["verifier_model"]     = router_model
            kwargs["openai_api_key"]     = router_openai_key
            kwargs["gemini_api_key"]     = router_gemini_key
        planner = pyplanner.get(
            selected_method,
            host     = ollama_host,
            model    = selected_model,
            provider = selected_provider,
            api_key  = planner_api_key,
            **kwargs,
        )
        st.session_state.planner_obj = planner

        # Check if the target scene/house is already loaded
        _proc_split_now = st.session_state.get("cfg_proc_split", "test")
        _proc_idx_now   = int(st.session_state.get("cfg_proc_house_idx", 1))
        already_loaded = bool(st.session_state.obs) and (
            (sim_type == "ProcThor"  and st.session_state.get("_last_proc_key") == f"{_proc_split_now}:{_proc_idx_now}")
            or (sim_type == "RoboThor" and st.session_state.get("task_scene") == final_scene)
            or (sim_type == "Thor"     and st.session_state.get("_last_manual_scene") == final_scene)
        )

        if already_loaded:
            obs_for_plan     = st.session_state.obs
            visible_for_plan = st.session_state.visible_objects
        else:
            with st.spinner(f"Loading scene `{final_scene}`..."):
                if sim_type == "ProcThor":
                    resp = client.reset(simulator_type="procthor", split=_proc_split_now, house_index=_proc_idx_now)
                elif sim_type == "RoboThor":
                    resp = client.reset(final_scene, simulator_type="robotthor")
                else:
                    resp = client.reset(final_scene)
            if resp.get("status") != "ok":
                st.error(f"Simulator error: {resp.get('msg', 'unknown')}")
                st.session_state.running = False
                st.stop()
            obs_for_plan     = resp.get("obs", "")
            visible_for_plan = resp.get("visible_objects_meta", resp.get("visible_objects", []))
            st.session_state.obs             = obs_for_plan
            st.session_state.visible_objects = visible_for_plan

        with st.spinner(f"🧠 [{selected_method} · {selected_provider} · {selected_model}] Generating plan..."):
            try:
                plan, metrics = planner.generate_plan(
                    task_info["desc"],
                    obs_for_plan,
                    [o["name"] if isinstance(o, dict) else o for o in visible_for_plan],
                )
            except Exception as _e:
                st.session_state.running = False
                st.error(f"❌ Plan generation failed: {_e}", icon="🚨")
                st.caption("Check LLM host URL, model name, and API key.")
                st.stop()

        if not plan:
            st.session_state.running = False
            st.warning("⚠ LLM returned empty plan. Try different model or rephrase task.")
            st.stop()

        st.session_state.plan             = plan
        st.session_state.last_metrics     = metrics
        st.session_state["plan_evaluation"] = None
        st.session_state.bench_history.append(metrics)
        st.session_state.log.append({"type": "plan", "steps": plan.copy(), "n": 0, "metrics": metrics})
        st.rerun()

    if reset:
        for k, v in DEFAULTS.items():
            if k != "client":
                st.session_state[k] = v
        # Clear scene-load caches so the scene reloads after reset
        for _ck in ["_last_manual_scene", "_last_proc_key", "_last_robo_scene"]:
            st.session_state.pop(_ck, None)
        # Reload scene immediately
        if client.connected:
            _proc_split_r = st.session_state.get("cfg_proc_split", "test")
            _proc_idx_r   = int(st.session_state.get("cfg_proc_house_idx", 1))
            _sim_r        = st.session_state.get("cfg_sim_type", "ProcThor")
            with st.spinner("Reloading scene..."):
                if _sim_r == "ProcThor":
                    _resp = client.reset(simulator_type="procthor", split=_proc_split_r, house_index=_proc_idx_r)
                    st.session_state["_last_proc_key"] = f"{_proc_split_r}:{_proc_idx_r}"
                elif _sim_r == "RoboThor":
                    _resp = client.reset(final_scene, simulator_type="robotthor")
                else:
                    _resp = client.reset(final_scene)
            if _resp.get("status") == "ok":
                st.session_state.obs             = _resp.get("obs", "")
                st.session_state.visible_objects = _resp.get("visible_objects_meta", _resp.get("visible_objects", []))
        st.rerun()

    st.divider()

    # ── Live stats ──
    st.subheader("Stats")
    c1, c2, c3 = st.columns(3)
    c1.metric("Done",    len(st.session_state.completed))
    c2.metric("Left",    len(st.session_state.plan))
    c3.metric("Replans", st.session_state.replan_count)

    if st.session_state.task_label:
        st.caption(f"Task: **{st.session_state.task_label}**")
        st.caption(f"Scene: `{st.session_state.task_scene}`")


# ══════════════════════════════════════════════════════════════════════
# MAIN AREA — 3 columns
# ══════════════════════════════════════════════════════════════════════
st.header("🤖 Daily Assistant Robot")
if st.session_state.task_label:
    _tinfo = get_task_info(st.session_state.task_label)
    _desc  = _tinfo.get("desc", st.session_state.task_label)
    st.caption(
        f"**{st.session_state.task_label}**"
        f"  ·  {_desc}"
        f"  ·  scene `{st.session_state.task_scene}`"
    )

col_cam, col_plan, col_log = st.columns([1, 1, 1], gap="large")


# ══════════════════════════════════════════════════════════════════════
# COLUMN 1 — Camera + observation
# ══════════════════════════════════════════════════════════════════════
with col_cam:
    # ── Direct command input ──────────────────────────────────────
    st.subheader("⌨️ Direct Commands")
    st.caption("One command per line: `Action Object [→ Target]`  ·  Ctrl+Enter to execute")

    from pyplanner.base import ROBOT_ACTIONS as _ROBOT_ACTIONS
    with st.expander("Valid actions", expanded=False):
        st.markdown(" · ".join(f"`{a}`" for a in _ROBOT_ACTIONS))

    
    with st.form(key="direct_cmd_form", clear_on_submit=False):
        cmd_text = st.text_area(
            "commands",
            placeholder="MoveTo Kitchen\nFind Apple\nPick\nMoveTo DiningTable\nPlace DiningTable",
            height=110,
            label_visibility="collapsed",
        )
        _btn_exec, _btn_eval, _btn_refine = st.columns(3)
        with _btn_exec:
            cmd_submitted = st.form_submit_button(
                "▶ Execute  (Ctrl+Enter)",
                type="primary",
                use_container_width=True,
                disabled=not client.connected,
            )
        with _btn_eval:
            cmd_evaluate = st.form_submit_button(
                "🔍 Evaluate",
                use_container_width=True,
            )
        with _btn_refine:
            cmd_refine = st.form_submit_button(
                "✏️ Refine",
                use_container_width=True,
            )

    if cmd_submitted and cmd_text.strip():
        import re as _re
        from pyplanner.base import ROBOT_ACTIONS

        # ── Object name resolver (uses module-level resolve_obj) ──
        def _resolve_obj(name: str) -> tuple[str, str | None]:
            """Resolve name and return (resolved, warning_or_None)."""
            if not name:
                return name, None
            resolved = resolve_obj(name)
            if resolved == name and name:
                visible = st.session_state.get("visible_objects", [])
                if visible:
                    avail = ", ".join(f"`{o['name'] if isinstance(o, dict) else o}`" for o in visible[:8])
                    return name, f"`{name}` not found in scene. Visible: {avail}"
            return resolved, None

        # ── Parse lines ───────────────────────────────────────────
        parsed_steps = []
        errors       = []
        warnings     = []

        for raw_line in cmd_text.strip().splitlines():
            line = raw_line.strip()
            if not line:
                continue

            # Split on → or -> for target
            parts  = line.replace("->", "→").split("→", 1)
            target = parts[1].strip() if len(parts) == 2 else ""
            left   = parts[0].strip().split(None, 1)
            if not left:
                continue
            action = left[0].strip()
            obj    = left[1].strip() if len(left) > 1 else ""

            # Validate action
            if action not in ROBOT_ACTIONS:
                # Try case-insensitive fix e.g. "moveto" → "MoveTo"
                fixed = next((a for a in ROBOT_ACTIONS if a.lower() == action.lower()), None)
                if fixed:
                    action = fixed
                else:
                    errors.append(
                        f"Unknown action `{action}`. "
                        f"Valid: {', '.join(sorted(ROBOT_ACTIONS))}"
                    )
                    continue

            # Resolve object and target to real scene names
            obj,    obj_warn    = _resolve_obj(obj)
            target, target_warn = _resolve_obj(target)
            if obj_warn:    warnings.append(obj_warn)
            if target_warn: warnings.append(target_warn)

            parsed_steps.append({
                "action": action,
                "object": obj,
                "target": target,
                "reason": "direct input",
            })

        # Show errors (block execution) and warnings (inform only)
        for e in errors:
            st.error(e)
        for w in warnings:
            st.warning(w)

        if parsed_steps and not errors:
            # Log as a "direct" plan entry so it appears in the execution log
            st.session_state.log.append({
                "type":  "plan",
                "steps": parsed_steps.copy(),
                "n":     0,
                "label": "Direct commands",
            })

            # Execute all steps immediately — bypass plan/replan system entirely
            results_ph = st.empty()
            for step in parsed_steps:
                action = step["action"]
                obj    = step["object"]
                target = step["target"]
                label  = f"{action} {obj}" + (f" → {target}" if target else "")
                results_ph.info(f"⏳ {label}")
                resp = client.step(action, obj, target)

                st.session_state.log.append({
                    "type":    "step",
                    "step":    step,
                    "obs":     resp.get("obs", ""),
                    "success": resp.get("success", False),
                    "reward":  resp.get("reward", 0.0),
                    "msg":     resp.get("msg", resp.get("message", "")),
                    "done":    resp.get("done", False),
                })

                if resp.get("obs"):
                    st.session_state.obs         = resp["obs"]
                    st.session_state.robot_state = resp.get("robot_state", resp["obs"])
                    st.session_state.env_state   = resp.get("env_state", "")
                if resp.get("visible_objects_meta") or resp.get("visible_objects"):
                    st.session_state.visible_objects = resp.get("visible_objects_meta", resp.get("visible_objects", []))
                st.session_state["_last_frame"] = None

            st.session_state.log.append({"type": "done"})
            results_ph.empty()
            st.rerun()

    def _parse_cmd_text(text: str) -> tuple[list[dict], list[str]]:
        """Parse raw command text into steps + error list (shared by eval & refine)."""
        from pyplanner.base import ROBOT_ACTIONS as _RA
        steps, errors = [], []
        for _raw in text.strip().splitlines():
            _line = _raw.strip()
            if not _line:
                continue
            _parts = _line.replace("->", "→").split("→", 1)
            _left  = _parts[0].strip().split(None, 1)
            if not _left:
                continue
            _act = _left[0].strip()
            _obj = _left[1].strip() if len(_left) > 1 else ""
            _fixed = next((a for a in _RA if a.lower() == _act.lower()), None)
            if _fixed:
                _act = _fixed
            elif _act not in _RA:
                errors.append(f"Unknown action `{_act}`")
                continue
            steps.append({"action": _act, "object": _obj})
        return steps, errors

    _cmd_task = st.session_state.get("task_custom_text", "").strip() or "Direct commands"

    if cmd_evaluate and cmd_text.strip():
        _eval_steps, _eval_errors = _parse_cmd_text(cmd_text)
        for _e in _eval_errors:
            st.error(_e)
        if _eval_steps and not _eval_errors:
            _eval_result = evaluate_plan(
                _eval_steps,
                _cmd_task,
                st.session_state.get("robot_state", st.session_state.obs),
            )
            st.session_state["_direct_cmd_eval"]   = _eval_result
            st.session_state["_direct_cmd_steps"]  = _eval_steps

    if cmd_refine and cmd_text.strip():
        _ref_steps, _ref_errors = _parse_cmd_text(cmd_text)
        for _e in _ref_errors:
            st.error(_e)
        if _ref_steps and not _ref_errors:
            # Run evaluate first if not already done
            _existing_eval = st.session_state.get("_direct_cmd_eval", "")
            if not _existing_eval:
                _existing_eval = evaluate_plan(
                    _ref_steps,
                    _cmd_task,
                    st.session_state.get("robot_state", st.session_state.obs),
                )
                st.session_state["_direct_cmd_eval"] = _existing_eval
            _refined_steps = refine_plan(
                _ref_steps,
                _cmd_task,
                st.session_state.get("robot_state", st.session_state.obs),
                _existing_eval,
            )
            if _refined_steps:
                # Re-evaluate the refined plan and display
                _refined_eval = evaluate_plan(
                    _refined_steps,
                    _cmd_task,
                    st.session_state.get("robot_state", st.session_state.obs),
                )
                st.session_state["_direct_cmd_eval"]          = _refined_eval
                st.session_state["_direct_cmd_steps"]         = _refined_steps
                st.session_state["_direct_cmd_refined_lines"] = "\n".join(
                    f"{s['action']} {s['object']}".strip()
                    for s in _refined_steps
                )
            else:
                st.warning("Refine produced an empty plan — keeping original.")

    _direct_eval         = st.session_state.get("_direct_cmd_eval")
    _direct_refined_txt  = st.session_state.get("_direct_cmd_refined_lines")
    if _direct_eval:
        with st.expander("📝 Direct command evaluation", expanded=True):
            st.code(_direct_eval, language=None)
        if _direct_refined_txt:
            st.caption("✏️ Refined plan — copy into the command box above to execute:")
            st.code(_direct_refined_txt, language=None)
        if st.button("✕ Clear", key="btn_clear_direct_eval"):
            for _k in ("_direct_cmd_eval", "_direct_cmd_steps", "_direct_cmd_refined_lines"):
                st.session_state.pop(_k, None)
            st.rerun()

    st.divider()

    # ── Camera ────────────────────────────────────────────────────
    st.subheader("🎥 Robot View")
    frame_ph = st.empty()

    # Use cached frame from session_state so it survives rerun after do_step()
    # Fresh frame is only fetched on explicit Refresh or first load
    cached_frame = st.session_state.get("_last_frame")
    if cached_frame:
        frame_ph.image(cached_frame, use_column_width=True)
    elif client.connected:
        live = client.get_frame()
        if live:
            st.session_state["_last_frame"] = live
            frame_ph.image(live, use_column_width=True)
        else:
            frame_ph.info("No camera feed.\nStart a task to see robot view.")
    else:
        frame_ph.info("No camera feed.\nStart a task to see robot view.")

    if st.session_state.running:
        if st.button("🔄 Refresh frame", use_container_width=True):
            f = client.get_frame()
            if f:
                st.session_state["_last_frame"] = f
                frame_ph.image(f, use_column_width=True)

    # ── Navigation D-pad ─────────────────────────────────────────────
    if client.connected:
        st.caption("🕹 Navigate")

        def _do_nav(action: str):
            """Execute a free nav action and refresh frame + obs."""
            try:
                resp = client.navigate_free(action)
                if resp.get("status") == "ok":
                    f = client.get_frame()
                    if f:
                        st.session_state["_last_frame"] = f
                    st.session_state.obs             = resp.get("obs", st.session_state.obs)
                    st.session_state.visible_objects = resp.get("visible_objects_meta", st.session_state.visible_objects)
            except Exception:
                pass

        # Row 1: Look up + Move forward
        r1a, r1b, r1c = st.columns([1, 1, 1])
        with r1a:
            if st.button("↖", use_container_width=True, help="Look Up",
                         key="nav_lookup"):
                _do_nav("LookUp"); st.rerun()
        with r1b:
            if st.button("▲", use_container_width=True, help="Move Forward",
                         key="nav_fwd"):
                _do_nav("MoveAhead"); st.rerun()
        with r1c:
            if st.button("↗", use_container_width=True, help="Look Down",
                         key="nav_lookdown"):
                _do_nav("LookDown"); st.rerun()

        # Row 2: Rotate left + placeholder + Rotate right
        r2a, r2b, r2c = st.columns([1, 1, 1])
        with r2a:
            if st.button("◀", use_container_width=True, help="Rotate Left",
                         key="nav_rotl"):
                _do_nav("RotateLeft"); st.rerun()
        with r2b:
            st.button("·", use_container_width=True, disabled=True, key="nav_center")
        with r2c:
            if st.button("▶", use_container_width=True, help="Rotate Right",
                         key="nav_rotr"):
                _do_nav("RotateRight"); st.rerun()

        # Row 3: Move back
        r3a, r3b, r3c = st.columns([1, 1, 1])
        with r3b:
            if st.button("▼", use_container_width=True, help="Move Back",
                         key="nav_back"):
                _do_nav("MoveBack"); st.rerun()

        st.markdown(
            "<div style='font-size:11px;color:var(--color-text-tertiary);"
            "text-align:center;margin-top:2px'>"
            "↖↗ = look up/down &nbsp;·&nbsp; ◀▶ = rotate &nbsp;·&nbsp; ▲▼ = move"
            "</div>",
            unsafe_allow_html=True,
        )

    robot_state = st.session_state.get("robot_state", st.session_state.obs)
    env_state   = st.session_state.get("env_state", "")
    if robot_state:
        st.caption("Robot state")
        # st.markdown(
        #     f'<div class="obs-box">{robot_state}</div>',
        #     unsafe_allow_html=True,
        # )
        # st.markdown(
        #     f'<pre class="obs-box">{robot_state}</pre>',
        #     unsafe_allow_html=True,
        # )
        formatted = html.escape(robot_state).replace("\n", "<br>")
        st.markdown(
            f'<div class="obs-box">{formatted}</div>',
            unsafe_allow_html=True,
        )
    if env_state:
        st.caption("Environment state")
        st.markdown(
            f'<div class="obs-box">{env_state}</div>',
            unsafe_allow_html=True,
        )

    if st.session_state.visible_objects:
        with st.expander(
            f"Visible objects ({len(st.session_state.visible_objects)})",
            expanded=True,   # auto-expand after scene load so user sees objects
        ):
            # Show as a grid of small chips for readability
            chips = []
            for o in st.session_state.visible_objects:
                name = o["name"] if isinstance(o, dict) else o
                extra_parts = []
                if isinstance(o, dict):
                    if o.get("pickupable"):
                        extra_parts.append("<span style='margin-left:4px' title='Pickupable (graspable)'>🤏</span>")
                    if o.get("receptacle"):
                        extra_parts.append("<span style='margin-left:4px' title='Receptacle (container)'>📥</span>")

                    if o.get("openable"):
                        state_str = ""
                        if o.get("isOpen") is not None:
                            state_str = " (is open)" if o.get("isOpen") else " (is closed)"
                        extra_parts.append(f"<span style='margin-left:4px' title='Openable{state_str}'>🚪</span>")

                    if o.get("toggleable"):
                        state_str = ""
                        if o.get("isToggled") is not None:
                            state_str = " (is on)" if o.get("isToggled") else " (is off)"
                        color = "inherit" if o.get("isToggled") else "#888"
                        extra_parts.append(f"<span style='margin-left:4px; color:{color};' title='Toggleable{state_str}'>💡</span>")
                extra = "".join(extra_parts)
                chips.append(
                    f'<span style="display:inline-block;padding:2px 8px;margin:2px;'
                    f'border-radius:12px;font-size:12px;'
                    f'background:var(--color-background-secondary);'
                    f'border:1px solid var(--color-border-tertiary)">'
                    f'{name}{extra}</span>'
                )
            st.markdown(" ".join(chips), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# COLUMN 2 — Action plan queue + execute controls
# ══════════════════════════════════════════════════════════════════════
with col_plan:
    st.subheader("📋 Action Plan")

    if not st.session_state.plan and not st.session_state.running:
        st.info("Configure a task in the sidebar and press **▶ Start**.")

    for i, step in enumerate(st.session_state.plan):
        label  = fmt_step(step)
        reason = step.get("reason", "")
        css    = "step-active" if i == 0 else "step-pending"
        marker = "▶" if i == 0 else f"{i + 1}."
        st.markdown(
            f'<div class="{css}">{marker} {label}'
            f'<span class="reason-tag">{reason}</span></div>',
            unsafe_allow_html=True,
        )

    st.write("")

    # Execute controls
    if st.session_state.running and st.session_state.plan:
        if auto_exec:
            if st.session_state.auto_running:
                do_step()
                time.sleep(step_delay)
                f = client.get_frame()
                if f:
                    st.session_state["_last_frame"] = f
                st.rerun()
            else:
                if st.button("▶▶ Resume auto", type="primary", use_container_width=True):
                    st.session_state.auto_running = True
                    st.rerun()
        else:
            if st.button("⏩ Execute next step", type="primary", use_container_width=True):
                do_step()
                f = client.get_frame()
                if f:
                    st.session_state["_last_frame"] = f
                st.rerun()

    elif not st.session_state.plan and st.session_state.running:
        st.session_state.running = False
        st.rerun()

    # ── Evaluate / Refine ─────────────────────────────────────────
    _current_plan = st.session_state.plan
    _planner      = st.session_state.get("planner_obj")
    if _current_plan and _planner:
        st.write("")
        _ev_col, _rf_col = st.columns(2)

        if _ev_col.button("🔍 Evaluate plan", use_container_width=True,
                          help="Ask LLM to assess each step's preconditions and sequence"):
            with st.spinner("Evaluating plan..."):
                try:
                    _eval_text = evaluate_plan(
                        _current_plan,
                        st.session_state.task_label,
                        st.session_state.get("robot_state", st.session_state.obs),
                        _planner,
                    )
                    st.session_state["plan_evaluation"] = _eval_text
                except Exception as _ee:
                    st.error(f"Evaluation failed: {_ee}")

        _evaluation = st.session_state.get("plan_evaluation")
        if _evaluation:
            with st.expander("📝 Plan evaluation", expanded=True):
                # st.markdown(_evaluation)
                formatted = html.escape(_evaluation).replace("\n", "<br>")
                st.markdown(
                    f'<div class="obs-box">{formatted}</div>',
                    unsafe_allow_html=True,
                )

            if _rf_col.button("✏️ Refine plan", use_container_width=True,
                              help="Ask LLM to produce an improved plan based on evaluation"):
                with st.spinner("Refining plan..."):
                    try:
                        _refined = refine_plan(
                            _current_plan,
                            st.session_state.task_label,
                            st.session_state.get("robot_state", st.session_state.obs),
                            _evaluation,
                            _planner,
                        )
                        if _refined:
                            st.session_state.plan = _refined
                            st.session_state["plan_evaluation"] = None
                            st.session_state.log.append({
                                "type":  "plan",
                                "steps": _refined.copy(),
                                "n":     st.session_state.replan_count,
                                "label": "Refined plan",
                            })
                            st.rerun()
                        else:
                            st.warning("Refined plan was empty — keeping original.")
                    except Exception as _re:
                        st.error(f"Refinement failed: {_re}")


# ══════════════════════════════════════════════════════════════════════
# COLUMN 3 — Execution log
# ══════════════════════════════════════════════════════════════════════
with col_log:
    st.subheader("📜 Execution Log")

    if not st.session_state.log:
        st.info("Log appears here as steps execute.")

    for entry in reversed(st.session_state.log):

        if entry["type"] == "plan":
            n     = entry["n"]
            label = entry.get("label") or ("Initial plan" if n == 0 else f"Replan #{n}")
            with st.expander(
                f"📋 {label} — {len(entry['steps'])} steps",
                expanded=(n == 0),
            ):
                for s in entry["steps"]:
                    rsn = s.get("reason", "")
                    st.markdown(
                        f"- `{s.get('action', '')}` {s.get('object', '')}"
                        + (f" → {s['target']}" if s.get("target") else "")
                        + (f"  *{rsn}*" if rsn else "")
                    )

        elif entry["type"] == "step":
            step  = entry["step"]
            label = fmt_step(step)
            msg   = entry.get("msg", "")
            r     = entry.get("reward", 0.0)
            if entry["success"]:
                st.markdown(
                    f'<div class="step-done">✅ {label}'
                    f'<span class="reason-tag">reward {r:.2f}</span></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="step-fail">❌ {label}'
                    f'<span class="reason-tag">{msg}</span></div>',
                    unsafe_allow_html=True,
                )
            with st.expander("observation", expanded=False):
                st.markdown(
                    f'<div class="obs-box">{entry.get("obs", "")}</div>',
                    unsafe_allow_html=True,
                )

        elif entry["type"] == "replan":
            replan_m = entry.get("metrics")
            label    = f"⚠️ Replan #{entry['n']} — {len(entry['steps'])} new steps"
            if replan_m:
                label += f"  ·  {replan_m.latency_s:.1f}s"
            st.warning(label, icon="🔄")

        elif entry["type"] == "done":
            st.success("🎉 Task completed!", icon="✅")

        elif entry["type"] == "failed":
            st.error(f"❌ Task failed: {entry.get('reason', '')}", icon="🚨")


# ══════════════════════════════════════════════════════════════════════
# METRICS PANEL — last plan metrics + benchmark history
# ══════════════════════════════════════════════════════════════════════
st.divider()
st.subheader("📊 Planning Metrics")

last_m = st.session_state.last_metrics

if last_m is None:
    st.info("Run a task to see planning metrics here.")
else:
    # ── Last run summary ──
    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    mc1.metric("Method",       last_m.method)
    mc2.metric("Model",        last_m.model.split(":")[0])
    mc3.metric("Latency",      f"{last_m.latency_s:.1f}s")
    mc4.metric("LLM calls",    last_m.llm_calls)
    mc5.metric("Steps",        last_m.num_steps)

    mc6, mc7, mc8, mc9, _ = st.columns(5)
    mc6.metric("Input tok",    last_m.input_tokens)
    mc7.metric("Output tok",   last_m.output_tokens)
    mc8.metric("Parse OK",     "✅" if last_m.parse_ok else "❌")
    if last_m.notes:
        mc9.metric("Notes",    last_m.notes)

    # Method-specific extras
    extra = last_m.extra or {}
    if "reasoning" in extra and extra["reasoning"]:
        with st.expander("💭 CoT / Few-Shot Reasoning", expanded=False):
            st.markdown(extra["reasoning"])
    if "thoughts" in extra and extra["thoughts"]:
        with st.expander(f"💭 ReAct Thoughts ({len(extra['thoughts'])})", expanded=False):
            for i, t in enumerate(extra["thoughts"]):
                st.markdown(f"**Step {i+1}:** {t}")
    if "subgoals" in extra and extra["subgoals"]:
        with st.expander("🎯 Hierarchical Sub-goals", expanded=False):
            for i, sg in enumerate(extra["subgoals"]):
                st.markdown(f"{i+1}. {sg}")
    if "retrieved_tasks" in extra and extra["retrieved_tasks"]:
        with st.expander(f"🔎 Retrieved Examples ({len(extra['retrieved_tasks'])})", expanded=False):
            for i, t in enumerate(extra["retrieved_tasks"], 1):
                st.markdown(f"{i}. _{t}_")
    if "critiques" in extra and extra["critiques"]:
        with st.expander(f"🔍 Self-Refine Critiques ({len(extra['critiques'])})", expanded=False):
            for i, c in enumerate(extra["critiques"]):
                st.markdown(f"**Iteration {i+1}:** {c}")
    if "verifier_note" in extra:
        changed = extra.get("steps_changed", False)
        icon    = "✏️" if changed else "✅"
        with st.expander(f"{icon} LLM Router — Verifier note", expanded=True):
            st.caption(extra["verifier_note"])
            if changed:
                col_a, col_b = st.columns(2)
                with col_a:
                    st.caption("Original (local)")
                    for s in extra.get("local_steps", []):
                        st.markdown(f"- `{s.get('action')}` {s.get('object')}")
                with col_b:
                    st.caption("After verification")
                    for s in (st.session_state.log[-1].get("steps") or []) if st.session_state.log else []:
                        st.markdown(f"- `{s.get('action')}` {s.get('object')}")

# ── Benchmark comparison table ──
bench = st.session_state.bench_history
if len(bench) >= 2:
    st.subheader("🏆 Benchmark Comparison")

    col_clear, _ = st.columns([1, 4])
    if col_clear.button("🗑 Clear history", use_container_width=True):
        st.session_state.bench_history = []
        st.rerun()

    import pandas as pd
    rows = []
    for i, m in enumerate(bench):
        total_tok = m.input_tokens + m.output_tokens
        tok_per_step = round(total_tok / m.num_steps, 0) if m.num_steps else 0
        rows.append({
            "#":              i + 1,
            "Method":         m.method,
            "Model":          m.model,
            "Latency (s)":    round(m.latency_s, 2),
            "LLM calls":      m.llm_calls,
            "Steps":          m.num_steps,
            "Total tokens":   total_tok,
            "Tok/step":       tok_per_step,
            "Parse OK":       "✅" if m.parse_ok else "❌",
            "Notes":          m.notes,
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Mini bar charts
    if len(df) > 1:
        chart_cols = st.columns(4)
        with chart_cols[0]:
            st.caption("Latency (s)")
            st.bar_chart(df.set_index("#")["Latency (s)"], height=160)
        with chart_cols[1]:
            st.caption("LLM calls")
            st.bar_chart(df.set_index("#")["LLM calls"], height=160)
        with chart_cols[2]:
            st.caption("Steps generated")
            st.bar_chart(df.set_index("#")["Steps"], height=160)
        with chart_cols[3]:
            st.caption("Tokens per step")
            st.bar_chart(df.set_index("#")["Tok/step"], height=160)