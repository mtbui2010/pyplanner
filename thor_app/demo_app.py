"""
demo_app.py — PyPlanner interactive demo for Streamlit Cloud

Runs WITHOUT AI2-THOR. Uses a simulated environment that returns realistic
observations, so recruiters can try all 7 planning methods live.

Deploy: streamlit run demo_app.py
"""

import json
import os
import sys
import time
import random
import streamlit as st

# ── Resolve pyplanner ─────────────────────────────────────────────────
try:
    import pyplanner
except ModuleNotFoundError:
    _here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(_here, "pyplanner"))
    import pyplanner

from pyplanner import REGISTRY, PROVIDER_MODELS, DEFAULT_BACKEND

# ════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="PyPlanner Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.step-card   { padding:10px 14px; border-radius:10px; margin:4px 0; font-size:14px; }
.step-done   { background:var(--color-background-success); border-left:3px solid var(--color-border-success); }
.step-active { background:var(--color-background-info);    border-left:3px solid var(--color-border-info); font-weight:500; }
.step-pending{ background:var(--color-background-secondary);border-left:3px solid var(--color-border-secondary); }
.obs-box     { background:var(--color-background-secondary); border-radius:8px;
               padding:10px 14px; font-family:monospace; font-size:12px;
               line-height:1.6; white-space:pre-wrap; }
.metric-chip { display:inline-block; padding:3px 10px; border-radius:20px; font-size:12px;
               background:var(--color-background-info); color:var(--color-text-info);
               margin:2px; font-weight:500; }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
# SIMULATED ENVIRONMENT
# Realistic AI2-THOR-style responses without running the simulator
# ════════════════════════════════════════════════════════════════════

SCENE_OBJECTS = {
    "Kitchen": {
        "objects": ["CoffeeMachine", "Mug", "CounterTop", "Fridge", "Apple",
                    "Sink", "Faucet", "StoveBurner", "Pot", "Microwave",
                    "Toaster", "Bread", "Egg", "Pan", "Knife"],
        "obs_template": "Kitchen. Agent at ({x:.1f}, {z:.1f}), facing {r:.0f}°.\n"
                        "Holding: {held}.\nNearby: {nearby}.",
    },
    "Living Room": {
        "objects": ["Television", "Sofa", "CoffeeTable", "Lamp", "Book",
                    "RemoteControl", "Bookshelf", "Pillow", "Window", "Curtain"],
        "obs_template": "Living room. Agent at ({x:.1f}, {z:.1f}), facing {r:.0f}°.\n"
                        "Holding: {held}.\nNearby: {nearby}.",
    },
    "Bedroom": {
        "objects": ["Bed", "Dresser", "AlarmClock", "Pillow", "LightSwitch",
                    "Nightstand", "Clothes", "Mirror", "Shelf"],
        "obs_template": "Bedroom. Agent at ({x:.1f}, {z:.1f}), facing {r:.0f}°.\n"
                        "Holding: {held}.\nNearby: {nearby}.",
    },
    "Bathroom": {
        "objects": ["Sink", "Faucet", "Soap", "Toothbrush", "Towel",
                    "ShowerFaucet", "Toilet", "Mirror", "TowelRack", "Cabinet"],
        "obs_template": "Bathroom. Agent at ({x:.1f}, {z:.1f}), facing {r:.0f}°.\n"
                        "Holding: {held}.\nNearby: {nearby}.",
    },
}

DEMO_TASKS = {
    "☕ Make coffee": {
        "room": "Kitchen",
        "desc": "Make a cup of coffee using the coffee machine",
        "key_objects": ["Mug", "CoffeeMachine"],
    },
    "🍳 Cook an egg": {
        "room": "Kitchen",
        "desc": "Pick up an egg and place it in a pan on the stove",
        "key_objects": ["Egg", "Pan", "StoveBurner"],
    },
    "🚿 Wash an apple": {
        "room": "Kitchen",
        "desc": "Pick up an apple and wash it in the sink",
        "key_objects": ["Apple", "Sink", "Faucet"],
    },
    "📺 Watch TV": {
        "room": "Living Room",
        "desc": "Turn on the television and sit on the sofa",
        "key_objects": ["Television", "Sofa"],
    },
    "📚 Read a book": {
        "room": "Living Room",
        "desc": "Pick up a book and sit on the sofa to read",
        "key_objects": ["Book", "Sofa"],
    },
    "😴 Go to sleep": {
        "room": "Bedroom",
        "desc": "Turn off the light and lie on the bed",
        "key_objects": ["LightSwitch", "Bed"],
    },
    "🪥 Brush teeth": {
        "room": "Bathroom",
        "desc": "Pick up toothbrush, use sink, brush teeth",
        "key_objects": ["Toothbrush", "Sink", "Faucet"],
    },
    "🧴 Wash hands": {
        "room": "Bathroom",
        "desc": "Turn on sink faucet and wash hands with soap",
        "key_objects": ["Faucet", "Soap", "Sink"],
    },
}

def _sim_obs(room: str, step_n: int, held: str = "nothing") -> str:
    x = round(random.uniform(0, 3), 1)
    z = round(random.uniform(0, 3), 1)
    r = random.choice([0, 90, 180, 270])
    objs = SCENE_OBJECTS[room]["objects"]
    nearby = ", ".join(random.sample(objs, min(4, len(objs))))
    return SCENE_OBJECTS[room]["obs_template"].format(
        x=x, z=z, r=r, held=held, nearby=nearby
    )


def _sim_step(action: str, obj: str, target: str, room: str, step_n: int) -> dict:
    """Simulate a step execution — realistic success/fail based on action type."""
    time.sleep(0.05)  # small delay for realism

    objs_lower = {o.lower().replace("_","") for o in SCENE_OBJECTS[room]["objects"]}
    obj_key    = obj.lower().replace("_","").replace(" ","")

    # Fail if object doesn't exist in scene
    obj_exists = any(obj_key in o for o in objs_lower) or not obj

    success = obj_exists
    reward  = 1.0 if success else 0.0
    held    = obj if action == "Grab" and success else "nothing"

    return {
        "success": success,
        "reward":  reward,
        "done":    step_n >= 5 and success,
        "obs":     _sim_obs(room, step_n, held=held if action == "Grab" else "nothing"),
        "msg":     "" if success else f"'{obj}' not found in {room.lower()} scene",
        "visible_objects": SCENE_OBJECTS[room]["objects"],
    }


# ════════════════════════════════════════════════════════════════════
# SESSION STATE
# ════════════════════════════════════════════════════════════════════
DEFAULTS = {
    "plan": [], "log": [], "completed": [],
    "running": False, "obs": "", "visible_objects": [],
    "task_label": "", "room": "",
    "last_metrics": None, "bench_history": [],
    "planner_obj": None, "step_n": 0,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


def do_step_sim():
    if not st.session_state.plan:
        return
    step   = st.session_state.plan.pop(0)
    room   = st.session_state.room
    result = _sim_step(
        step.get("action","Wait"),
        step.get("object",""),
        step.get("target",""),
        room,
        st.session_state.step_n,
    )
    st.session_state.step_n += 1
    st.session_state.obs             = result["obs"]
    st.session_state.visible_objects = result["visible_objects"]
    st.session_state.log.append({
        "type":    "step", "step": step,
        "obs":     result["obs"],
        "success": result["success"],
        "reward":  result["reward"],
        "msg":     result["msg"],
        "done":    result["done"],
    })
    if result["done"]:
        st.session_state.running = False
        st.session_state.plan    = []
        st.session_state.log.append({"type":"done"})
        return
    if result["success"]:
        st.session_state.completed.append(step)
    if not st.session_state.plan:
        st.session_state.running = False
        st.session_state.log.append({"type":"done"})


# ════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("🤖 PyPlanner Demo")
    st.caption("Simulated AI2-THOR · No local setup needed")
    st.info("Real PyPlanner package — simulated environment", icon="ℹ️")
    st.divider()

    # ── Task ──
    st.subheader("Task")
    task_label = st.selectbox("Select task", list(DEMO_TASKS.keys()))
    task       = DEMO_TASKS[task_label]
    st.caption(f"_{task['desc']}_")
    st.caption(f"Room: **{task['room']}**")

    st.divider()

    # ── Planning method ──
    st.subheader("Planning method")
    method = st.selectbox(
        "Method",
        list(REGISTRY.keys()),
        help="Seven different LLM planning algorithms — all produce the same step format.",
    )
    st.caption(f"_{REGISTRY[method].description}_")

    method_kwargs = {}
    if method == "ReAct":
        method_kwargs["max_steps"] = st.slider("Max steps", 5, 15, 10)
    elif method == "Self-Refine":
        method_kwargs["max_iterations"] = st.slider("Refine iterations", 1, 3, 2)

    st.divider()

    # ── LLM Provider ──
    st.subheader("LLM Provider")
    provider = st.selectbox("Provider", ["ollama", "openai", "anthropic"])

    if provider == "ollama":
        host  = st.text_input("Ollama host", value="http://localhost:11434")
        model = st.selectbox("Model", PROVIDER_MODELS["ollama"])
        api_key = ""
        st.caption("Run `ollama serve` locally, or point to a remote server.")
    elif provider == "openai":
        host  = "https://api.openai.com"
        model = st.selectbox("Model", PROVIDER_MODELS["openai"])
        api_key = st.text_input("OpenAI API key", type="password",
                                 help="sk-... · or set OPENAI_API_KEY env var")
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY", "")
    else:
        host  = "https://api.anthropic.com"
        model = st.selectbox("Model", PROVIDER_MODELS["anthropic"])
        api_key = st.text_input("Anthropic API key", type="password",
                                 help="sk-ant-... · or set ANTHROPIC_API_KEY env var")
        if not api_key:
            api_key = os.getenv("ANTHROPIC_API_KEY", "")

    custom_model = st.text_input("Custom model name (optional)",
                                  placeholder=model,
                                  help="Override dropdown with any model name")
    if custom_model.strip():
        model = custom_model.strip()

    st.divider()

    # ── Start / Reset ──
    col_s, col_r = st.columns([3, 1])
    with col_s:
        start = st.button("▶ Generate Plan", type="primary", use_container_width=True)
    with col_r:
        reset = st.button("↺", use_container_width=True, help="Reset")

    if reset:
        for k, v in DEFAULTS.items():
            st.session_state[k] = v
        st.rerun()

    if start:
        for k in ["plan","log","completed"]:
            st.session_state[k] = []
        st.session_state.running    = True
        st.session_state.task_label = task_label
        st.session_state.room       = task["room"]
        st.session_state.step_n     = 0

        # Generate initial obs from simulated scene
        room    = task["room"]
        obs     = _sim_obs(room, 0)
        visible = SCENE_OBJECTS[room]["objects"]
        st.session_state.obs             = obs
        st.session_state.visible_objects = visible

        try:
            planner = pyplanner.get(
                method, host=host, model=model,
                provider=provider, api_key=api_key,
                **method_kwargs,
            )
            st.session_state.planner_obj = planner
        except Exception as e:
            st.error(f"Failed to init planner: {e}")
            st.session_state.running = False
            st.stop()

        with st.spinner(f"🧠 [{method} · {provider}/{model}] Generating plan..."):
            try:
                plan, metrics = planner.generate_plan(task["desc"], obs, visible)
            except Exception as e:
                st.error(f"LLM error: {e}")
                st.session_state.running = False
                st.stop()

        st.session_state.plan         = plan
        st.session_state.last_metrics = metrics
        st.session_state.bench_history.append(metrics)
        st.session_state.log.append({"type":"plan","steps":plan.copy(),"n":0,"metrics":metrics})
        st.rerun()

    st.divider()

    # Stats
    st.subheader("Stats")
    c1, c2 = st.columns(2)
    c1.metric("Steps done", len(st.session_state.completed))
    c2.metric("Remaining",  len(st.session_state.plan))
    if st.session_state.task_label:
        st.caption(f"**{st.session_state.task_label}** · {st.session_state.room}")


# ════════════════════════════════════════════════════════════════════
# MAIN AREA
# ════════════════════════════════════════════════════════════════════
st.title("🤖 PyPlanner — Live Demo")
st.caption(
    "Real [PyPlanner](https://github.com/your-repo) package · "
    "simulated AI2-THOR environment · "
    "seven planning algorithms · "
    "three LLM backends"
)

if not st.session_state.task_label:
    st.info("👈 Select a task and click **▶ Generate Plan** to start.")

    # Show the package API as a teaser
    with st.expander("See the PyPlanner API", expanded=True):
        st.code("""
import pyplanner

# Seven methods, three backends — same interface
planner = pyplanner.cot(provider="openai", model="gpt-4o-mini", api_key="sk-...")

steps, metrics = planner.generate_plan(
    task="make a cup of coffee",
    obs="Kitchen. Coffee machine and mug visible.",
    visible_objects=["CoffeeMachine", "Mug", "CounterTop"],
)

for s in steps:
    print(f"{s['action']:10} {s['object']:20} → {s['target']}")

# Navigate    Mug                  →
# Grab        Mug                  →
# Navigate    CoffeeMachine        →
# Place       Mug                  → CoffeeMachine
# TurnOn      CoffeeMachine        →

print(f"Generated in {metrics.latency_s:.1f}s · {metrics.total_tokens} tokens")
        """, language="python")
    st.stop()

col_plan, col_exec, col_log = st.columns([1, 1, 1], gap="large")

# ── Plan queue ──────────────────────────────────────────────────────
with col_plan:
    m = st.session_state.last_metrics
    if m:
        st.subheader("📋 Generated Plan")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Steps",   m.num_steps)
        c2.metric("Latency", f"{m.latency_s:.1f}s")
        c3.metric("Calls",   m.llm_calls)
        c4.metric("Tokens",  m.total_tokens)

        if hasattr(m, "extra") and m.extra:
            extra = m.extra
            if extra.get("reasoning"):
                with st.expander("💭 Chain-of-thought reasoning", expanded=False):
                    st.markdown(extra["reasoning"])
            if extra.get("thoughts"):
                with st.expander(f"💭 ReAct thoughts ({len(extra['thoughts'])})", expanded=False):
                    for i, t in enumerate(extra["thoughts"]):
                        st.markdown(f"**{i+1}.** {t}")
            if extra.get("subgoals"):
                with st.expander("🎯 Hierarchical sub-goals", expanded=False):
                    for i, sg in enumerate(extra["subgoals"]):
                        st.markdown(f"{i+1}. {sg}")
            if extra.get("critiques"):
                with st.expander(f"🔍 Self-Refine critiques ({len(extra['critiques'])})", expanded=False):
                    for i, c in enumerate(extra["critiques"]):
                        st.markdown(f"**Iter {i+1}:** {c}")
    else:
        st.subheader("📋 Plan")

    plan = st.session_state.plan
    done = st.session_state.completed

    for i, step in enumerate(done[-5:]):
        label = f"{step.get('action','')}  {step.get('object','')}"
        if step.get("target"):
            label += f"  →  {step['target']}"
        st.markdown(f'<div class="step-card step-done">✓ {label}</div>',
                    unsafe_allow_html=True)

    for i, step in enumerate(plan):
        label = f"{step.get('action','')}  {step.get('object','')}"
        if step.get("target"):
            label += f"  →  {step['target']}"
        css = "step-active" if i == 0 else "step-pending"
        marker = "▶" if i == 0 else f"{i+1}."
        reason = step.get("reason","")
        st.markdown(
            f'<div class="step-card {css}">{marker} {label}'
            + (f'<br><span style="font-size:11px;color:var(--color-text-secondary)">{reason}</span>' if reason else "")
            + "</div>",
            unsafe_allow_html=True,
        )

    st.write("")
    if st.session_state.running and plan:
        if st.button("⏩ Execute next step", type="primary", use_container_width=True):
            do_step_sim()
            st.rerun()
        if st.button("⏩⏩ Run all steps", use_container_width=True):
            while st.session_state.plan and st.session_state.running:
                do_step_sim()
            st.rerun()


# ── Scene / observation ─────────────────────────────────────────────
with col_exec:
    st.subheader("🌐 Environment State")

    if st.session_state.obs:
        st.markdown(
            f'<div class="obs-box">{st.session_state.obs}</div>',
            unsafe_allow_html=True,
        )
        st.write("")
        if st.session_state.visible_objects:
            with st.expander(
                f"Visible objects ({len(st.session_state.visible_objects)})",
                expanded=True,
            ):
                # Show objects as chips
                chips = " ".join(
                    f'<span class="metric-chip">{o}</span>'
                    for o in st.session_state.visible_objects
                )
                st.markdown(chips, unsafe_allow_html=True)
    else:
        st.info("Environment state appears here after plan generation.")

    # Completed steps summary
    if st.session_state.completed:
        st.write("")
        st.caption(f"**{len(st.session_state.completed)} steps executed**")
        reward = sum(
            e.get("reward", 0) for e in st.session_state.log if e.get("type") == "step"
        )
        st.metric("Total reward", f"{reward:.1f}")


# ── Log ─────────────────────────────────────────────────────────────
with col_log:
    st.subheader("📜 Execution Log")

    if not st.session_state.log:
        st.info("Execution log appears here as steps run.")

    for entry in reversed(st.session_state.log):
        if entry["type"] == "plan":
            m = entry.get("metrics")
            label = f"📋 Initial plan — {len(entry['steps'])} steps"
            if m:
                label += f" · {m.latency_s:.1f}s · {m.llm_calls} call(s)"
            with st.expander(label, expanded=True):
                for s in entry["steps"]:
                    st.markdown(
                        f"- `{s.get('action','')}` **{s.get('object','')}**"
                        + (f" → {s['target']}" if s.get("target") else "")
                        + (f"  *{s.get('reason','')}*" if s.get("reason") else "")
                    )

        elif entry["type"] == "step":
            step = entry["step"]
            label = f"{step.get('action','')}  {step.get('object','')}"
            if step.get("target"):
                label += f" → {step['target']}"
            r = entry.get("reward", 0.0)
            if entry["success"]:
                st.success(f"✅ {label}  (reward {r:.1f})")
            else:
                st.error(f"❌ {label}  — {entry.get('msg','')}")

        elif entry["type"] == "done":
            st.balloons()
            st.success("🎉 Task completed!", icon="✅")


# ════════════════════════════════════════════════════════════════════
# BENCHMARK PANEL
# ════════════════════════════════════════════════════════════════════
bench = st.session_state.bench_history
if len(bench) >= 2:
    st.divider()
    st.subheader("📊 Method Comparison")

    col_clear, _ = st.columns([1, 5])
    if col_clear.button("🗑 Clear history"):
        st.session_state.bench_history = []
        st.rerun()

    import pandas as pd
    rows = []
    for i, bm in enumerate(bench):
        rows.append({
            "#":           i + 1,
            "Method":      bm.method,
            "Model":       bm.model,
            "Latency (s)": round(bm.latency_s, 2),
            "LLM calls":   bm.llm_calls,
            "Steps":       bm.num_steps,
            "Tokens":      bm.total_tokens,
            "Tok/step":    round(bm.tokens_per_step, 0),
            "Parse OK":    "✅" if bm.parse_ok else "❌",
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    if len(df) > 1:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.caption("Latency (s)")
            st.bar_chart(df.set_index("#")["Latency (s)"], height=180)
        with c2:
            st.caption("LLM calls")
            st.bar_chart(df.set_index("#")["LLM calls"], height=180)
        with c3:
            st.caption("Tokens / step")
            st.bar_chart(df.set_index("#")["Tok/step"], height=180)


# ════════════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════════════
st.divider()
st.markdown(
    "<div style='text-align:center;color:var(--color-text-secondary);font-size:13px'>"
    "PyPlanner · 7 planning methods · 3 LLM backends · "
    "AI2-THOR evaluation pipeline · "
    "<a href='https://github.com/your-repo'>GitHub</a>"
    "</div>",
    unsafe_allow_html=True,
)