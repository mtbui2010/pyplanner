"""
demo_app.py — PyPlanner interactive demo
=========================================
Auto-detects the best available backend, in order:

  1. Remote ThorServer  — if THOR_SERVER_URL is set (ngrok / VPS / local)
  2. Local AI2-THOR     — if ai2thor is installed and Xvfb is available
  3. Rich simulation    — always works; LLM calls are real, environment is mocked

Deploy on Streamlit Cloud (option 3 always active):
  streamlit run demo_app.py

For live robot execution (option 1):
  # On your machine:
  python thor_app/thor_server.py
  ngrok tcp 5555
  # Set env var:
  THOR_SERVER_URL=tcp://0.tcp.ngrok.io:12345 streamlit run demo_app.py
"""

import os, sys, time, random, json
import streamlit as st

# Read Streamlit Cloud secrets into env vars (no-op if running locally)
try:
    for _k in ["OLLAMA_HOST", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "THOR_SERVER_URL"]:
        if _k in st.secrets and not os.environ.get(_k):
            os.environ[_k] = st.secrets[_k]
except Exception:
    pass

# ── Resolve pyplanner ─────────────────────────────────────────────
try:
    import pyplanner
except ModuleNotFoundError:
    _here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(_here, "pyplanner"))
    import pyplanner

from pyplanner import REGISTRY, PROVIDER_MODELS

# ════════════════════════════════════════════════════════════════════
# BACKEND DETECTION
# ════════════════════════════════════════════════════════════════════

def _detect_backend():
    """
    Returns (mode, client_or_None, description).
    mode: "remote" | "local_thor" | "simulation"
    """
    # Option 1: Remote ThorServer via env var (ngrok or VPS)
    url = os.environ.get("THOR_SERVER_URL", "")
    if url:
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "thor_app"))
            from thor_client import ThorClient
            # Parse: tcp://host:port  or  host:port  or  host
            host, port = "localhost", 5555
            raw = url.replace("tcp://", "").replace("zmq://", "")
            if ":" in raw:
                parts = raw.rsplit(":", 1)
                host, port = parts[0], int(parts[1])
            else:
                host = raw
            client = ThorClient(host=host, port=port)
            if client.connected:
                return "remote", client, f"🟢 Live AI2-THOR — {host}:{port}"
            else:
                # Log the failure so it's visible in sidebar
                return "simulation", None, f"⚠ ThorServer unreachable ({host}:{port}) — using simulation"
        except Exception as e:
            return "simulation", None, f"⚠ ThorServer error ({e}) — using simulation"

    # Option 2: Local AI2-THOR with virtual display
    try:
        import ai2thor  # noqa
        import subprocess
        xvfb = subprocess.run(["which", "Xvfb"], capture_output=True)
        if xvfb.returncode == 0:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "thor_app"))
            from thor_client import ThorClient
            # Try starting thor_server as subprocess with Xvfb
            import subprocess as sp
            env = os.environ.copy()
            env["DISPLAY"] = ":99"
            sp.Popen(["Xvfb", ":99", "-screen", "0", "1024x768x24"],
                     stdout=sp.DEVNULL, stderr=sp.DEVNULL)
            time.sleep(1.5)
            srv = sp.Popen(
                [sys.executable, os.path.join(os.path.dirname(__file__), "thor_app", "thor_server.py")],
                env=env, stdout=sp.DEVNULL, stderr=sp.DEVNULL,
            )
            time.sleep(4)
            client = ThorClient(host="localhost", port=5555)
            if client.connected:
                return "local_thor", client, "Local AI2-THOR (headless)"
    except Exception:
        pass

    # Option 3: Rich simulation (always works)
    return "simulation", None, "Simulation mode — LLM calls are real"


@st.cache_resource(show_spinner="Initialising environment...")
def get_backend():
    return _detect_backend()


# ════════════════════════════════════════════════════════════════════
# SIMULATION ENGINE  (used when no real simulator available)
# ════════════════════════════════════════════════════════════════════

SCENE_OBJECTS = {
    "Kitchen": [
        "CoffeeMachine","Mug","CounterTop","Fridge","Apple","Sink","Faucet",
        "StoveBurner","Pot","Microwave","Toaster","Bread","Egg","Pan","Knife","Cabinet",
    ],
    "Living Room": [
        "Television","Sofa","CoffeeTable","Lamp","Book","RemoteControl",
        "Bookshelf","Pillow","Window","Curtain",
    ],
    "Bedroom": [
        "Bed","Dresser","AlarmClock","Pillow","LightSwitch","Nightstand","Clothes","Mirror",
    ],
    "Bathroom": [
        "Sink","Faucet","Soap","Toothbrush","Towel","ShowerFaucet","Toilet","Cabinet","TowelRack",
    ],
}

DEMO_TASKS = {
    "☕ Make coffee":       {"room":"Kitchen",     "desc":"Make a cup of coffee using the coffee machine",    "key":["Mug","CoffeeMachine"]},
    "🍳 Cook an egg":       {"room":"Kitchen",     "desc":"Pick up an egg and place it in a pan on the stove","key":["Egg","Pan","StoveBurner"]},
    "🚿 Wash an apple":     {"room":"Kitchen",     "desc":"Pick up an apple and wash it in the sink",         "key":["Apple","Sink","Faucet"]},
    "📡 Heat in microwave": {"room":"Kitchen",     "desc":"Place food in microwave and turn it on",           "key":["Bread","Microwave"]},
    "📺 Watch TV":          {"room":"Living Room", "desc":"Turn on the television and sit on the sofa",       "key":["Television","Sofa"]},
    "📚 Read a book":       {"room":"Living Room", "desc":"Pick up a book and sit on the sofa to read",       "key":["Book","Sofa"]},
    "😴 Go to sleep":       {"room":"Bedroom",     "desc":"Turn off the light and lie on the bed",            "key":["LightSwitch","Bed"]},
    "🪥 Brush teeth":       {"room":"Bathroom",    "desc":"Pick up toothbrush, use sink, brush teeth",        "key":["Toothbrush","Sink","Faucet"]},
    "🧴 Wash hands":        {"room":"Bathroom",    "desc":"Turn on faucet and wash hands with soap",          "key":["Faucet","Soap","Sink"]},
}

ROBOT_ACTIONS = set(pyplanner.ROBOT_ACTIONS)

def _make_obs(room, step_n, held="nothing"):
    objs = SCENE_OBJECTS[room]
    x, z = round(random.uniform(0, 3), 1), round(random.uniform(0, 3), 1)
    r    = random.choice([0, 90, 180, 270])
    near = ", ".join(f"{o} ({random.uniform(0.5,2.5):.1f}m)" for o in random.sample(objs, min(4,len(objs))))
    return (f"Location: ({x:.1f}, {z:.1f}), facing {r}°\n"
            f"Holding: {held}\n"
            f"Nearby: {near}")


def _sim_step(action, obj, target, room, step_n):
    """Simulated step — succeeds for real objects in scene, fails otherwise."""
    time.sleep(0.06)
    objs_lower = {o.lower().replace("_","") for o in SCENE_OBJECTS[room]}
    obj_key    = obj.lower().replace("_","").replace(" ","")
    obj_exists = not obj or any(obj_key in o or o in obj_key for o in objs_lower)
    success    = action in ROBOT_ACTIONS and obj_exists
    held       = obj if action == "Grab" and success else "nothing"
    done       = step_n >= 4 and success and action in ("TurnOn","LieOn","Sit","Serve")
    return {
        "success":         success,
        "reward":          1.0 if success else 0.0,
        "done":            done,
        "obs":             _make_obs(room, step_n, held=held if action=="Grab" else "nothing"),
        "msg":             "" if success else f"'{obj}' not found in scene",
        "visible_objects": SCENE_OBJECTS[room],
    }


def execute_step(client, mode, step, room, step_n):
    """Unified step executor — routes to real sim or simulation."""
    action = step.get("action","Wait")
    obj    = step.get("object","")
    target = step.get("target","")

    if mode in ("remote","local_thor") and client:
        result = client.step(action, obj, target)
        if result.get("status") == "error":
            # Fallback to simulation on error
            result = _sim_step(action, obj, target, room, step_n)
        return result
    else:
        return _sim_step(action, obj, target, room, step_n)


def reset_env(client, mode, room, task_label):
    """Reset environment and return initial obs + visible objects."""
    ROOM_SCENE = {
        "Kitchen":"FloorPlan1","Living Room":"FloorPlan201",
        "Bedroom":"FloorPlan301","Bathroom":"FloorPlan401",
    }
    if mode in ("remote","local_thor") and client:
        resp = client.reset(ROOM_SCENE.get(room, "FloorPlan1"))
        if resp.get("status") == "ok":
            return resp.get("obs",""), resp.get("visible_objects", SCENE_OBJECTS[room])
    return _make_obs(room, 0), SCENE_OBJECTS[room]


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
.step-card    { padding:9px 14px; border-radius:9px; margin:3px 0; font-size:14px; }
.step-done    { background:var(--color-background-success); border-left:3px solid var(--color-border-success); }
.step-active  { background:var(--color-background-info);    border-left:3px solid var(--color-border-info); font-weight:500; }
.step-pending { background:var(--color-background-secondary);border-left:3px solid var(--color-border-secondary); }
.obs-box      { background:var(--color-background-secondary); border-radius:8px;
                padding:10px 14px; font-family:monospace; font-size:12px;
                line-height:1.7; white-space:pre-wrap; }
.badge        { display:inline-block; padding:2px 9px; border-radius:12px; font-size:11px;
                font-weight:500; margin:2px; }
.badge-ok     { background:var(--color-background-success); color:var(--color-text-success); }
.badge-warn   { background:var(--color-background-warning); color:var(--color-text-warning); }
.badge-info   { background:var(--color-background-info);    color:var(--color-text-info); }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
# BACKEND INIT
# ════════════════════════════════════════════════════════════════════
mode, client, backend_desc = get_backend()

# ════════════════════════════════════════════════════════════════════
# SESSION STATE
# ════════════════════════════════════════════════════════════════════
DEFAULTS = {
    "plan":[], "log":[], "completed":[], "running":False,
    "obs":"", "visible_objects":[], "task_label":"", "room":"",
    "last_metrics":None, "bench_history":[], "planner_obj":None, "step_n":0,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


def do_step():
    if not st.session_state.plan:
        return
    step   = st.session_state.plan.pop(0)
    result = execute_step(client, mode, step, st.session_state.room, st.session_state.step_n)
    st.session_state.step_n += 1
    st.session_state.obs             = result["obs"]
    st.session_state.visible_objects = result["visible_objects"]
    st.session_state.log.append({
        "type":"step", "step":step,
        "obs":result["obs"], "success":result["success"],
        "reward":result["reward"], "msg":result["msg"], "done":result["done"],
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
    st.title("🤖 PyPlanner")
    st.caption("Interactive demo · 7 planning methods · 3 LLM backends")

    # Backend status
    icons = {"remote":"🟢","local_thor":"🟡","simulation":"🔵"}
    st.markdown(
        f'<div class="badge badge-{"ok" if mode=="remote" else "info"}">'
        f'{icons[mode]} {backend_desc}</div>',
        unsafe_allow_html=True,
    )
    if mode == "simulation":
        st.caption("LLM calls are **real**. Environment is simulated.")
        with st.expander("Enable live AI2-THOR robot"):
            st.code("# Run on your machine:\npython thor_app/thor_server.py\nngrok tcp 5555\n\n# Then set env var:\nexport THOR_SERVER_URL=tcp://0.tcp.ngrok.io:PORT\nstreamlit run demo_app.py", language="bash")

    st.divider()

    # ── Task ──
    st.subheader("Task")
    task_label = st.selectbox("Select task", list(DEMO_TASKS.keys()))
    task       = DEMO_TASKS[task_label]
    st.caption(f"_{task['desc']}_")

    st.divider()

    # ── Method ──
    st.subheader("Planning method")
    method = st.selectbox("Method", list(REGISTRY.keys()), index=0)
    st.caption(f"_{REGISTRY[method].description}_")

    mkw = {}
    if method == "ReAct":
        mkw["max_steps"] = st.slider("Max steps (ReAct)", 5, 15, 10)
    elif method == "Self-Refine":
        mkw["max_iterations"] = st.slider("Refine iterations", 1, 3, 2)

    st.divider()

    # ── LLM Provider ──
    st.subheader("LLM Provider")
    provider = st.selectbox("Provider", ["ollama","openai","anthropic"])

    if provider == "ollama":
        host    = st.text_input("Ollama host", value=os.environ.get("OLLAMA_HOST","https://ollama.aistations.org"))
        model   = st.selectbox("Model", PROVIDER_MODELS["ollama"])
        api_key = ""
    elif provider == "openai":
        host    = ""
        model   = st.selectbox("Model", PROVIDER_MODELS["openai"])
        api_key = st.text_input("OpenAI API key", type="password") or os.getenv("OPENAI_API_KEY","")
    else:
        host    = ""
        model   = st.selectbox("Model", PROVIDER_MODELS["anthropic"])
        api_key = st.text_input("Anthropic API key", type="password") or os.getenv("ANTHROPIC_API_KEY","")

    custom = st.text_input("Custom model name (optional)", placeholder=model)
    if custom.strip():
        model = custom.strip()

    st.divider()

    col_s, col_r = st.columns([3,1])
    with col_s:
        start = st.button("▶ Generate Plan", type="primary", use_container_width=True)
    with col_r:
        if st.button("↺", help="Reset"):
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

        obs, visible = reset_env(client, mode, task["room"], task_label)
        st.session_state.obs             = obs
        st.session_state.visible_objects = visible

        try:
            planner = pyplanner.get(method, host=host or os.environ.get("OLLAMA_HOST","https://ollama.aistations.org"),
                                    model=model, provider=provider,
                                    api_key=api_key, **mkw)
            st.session_state.planner_obj = planner
        except Exception as e:
            st.error(f"Planner init failed: {e}")
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

    # Stats
    st.divider()
    c1, c2 = st.columns(2)
    c1.metric("Done",  len(st.session_state.completed))
    c2.metric("Left",  len(st.session_state.plan))


# ════════════════════════════════════════════════════════════════════
# MAIN AREA
# ════════════════════════════════════════════════════════════════════
st.title("🤖 PyPlanner — Live Demo")
st.caption(
    "**7 planning methods · 3 LLM backends · AI2-THOR evaluation pipeline** · "
    "[GitHub](https://github.com/your-repo) · "
    "[pyplanner package](https://github.com/your-repo/tree/main/pyplanner)"
)

# Backend badge
badge_color = {"remote":"badge-ok","local_thor":"badge-ok","simulation":"badge-info"}[mode]
badge_text  = {"remote":"🟢 Live AI2-THOR robot","local_thor":"🟡 Local AI2-THOR","simulation":"🔵 Simulation — LLM calls real"}[mode]
st.markdown(f'<span class="badge {badge_color}">{badge_text}</span>', unsafe_allow_html=True)
st.write("")

if not st.session_state.task_label:
    # Landing — show the API
    col_a, col_b = st.columns([1,1], gap="large")
    with col_a:
        st.subheader("Install")
        st.code("pip install -e ./pyplanner", language="bash")

        st.subheader("Usage")
        st.code("""\
import pyplanner

# Pick any of 7 methods
planner = pyplanner.cot(
    provider="openai",
    model="gpt-4o-mini",
    api_key="sk-...",
)

steps, metrics = planner.generate_plan(
    task="make a cup of coffee",
    obs="Kitchen. Coffee machine and mug visible.",
    visible_objects=["CoffeeMachine","Mug","CounterTop"],
)

for s in steps:
    print(f"{s['action']:10} {s['object']}")
# Navigate   Mug
# Grab       Mug
# Navigate   CoffeeMachine
# Place      Mug  → CoffeeMachine
# TurnOn     CoffeeMachine

print(metrics.to_dict())
# latency_s=2.3  llm_calls=1  total_tokens=580
""", language="python")

    with col_b:
        st.subheader("7 planning methods")
        methods_data = [
            ("Direct",       "1",   "Single prompt → JSON"),
            ("CoT",          "1",   "Reason first, then plan"),
            ("Few-Shot CoT", "1",   "Examples anchor output"),
            ("Self-Refine",  "1+2N","Generate → critique → fix"),
            ("ReAct",        "N",   "Thought + Action per step"),
            ("Hierarchical", "1+N", "Sub-goals → actions"),
            ("LLM Router",   "2",   "Local generate + API verify"),
        ]
        import pandas as pd
        st.dataframe(
            pd.DataFrame(methods_data, columns=["Method","LLM calls","Strategy"]),
            use_container_width=True, hide_index=True,
        )

        st.subheader("Evaluation pipeline")
        for step in [
            "record_reference.py  — execute steps in AI2-THOR, keep only success=True",
            "goal_checker.py      — verify task goal reached (not just steps done)",
            "evaluate_sim.py      — benchmark all methods, output CSV metrics",
        ]:
            st.markdown(f"**`{step.split('—')[0].strip()}`** — {step.split('—')[1].strip()}")

    st.info("👈 Select a task and click **▶ Generate Plan** to try it live.")
    st.stop()

# ── Main UI ──────────────────────────────────────────────────────────
col_plan, col_env, col_log = st.columns([1, 1, 1], gap="large")

with col_plan:
    m = st.session_state.last_metrics
    if m:
        st.subheader(f"📋 {m.method} Plan")
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Steps",   m.num_steps)
        c2.metric("Latency", f"{m.latency_s:.1f}s")
        c3.metric("Calls",   m.llm_calls)
        c4.metric("Tokens",  m.total_tokens)

        extra = m.extra or {}
        if extra.get("reasoning"):
            with st.expander("💭 Reasoning trace", expanded=False):
                st.markdown(extra["reasoning"])
        if extra.get("subgoals"):
            with st.expander("🎯 Sub-goals", expanded=False):
                for i, sg in enumerate(extra["subgoals"]):
                    st.markdown(f"{i+1}. {sg}")
        if extra.get("thoughts"):
            with st.expander(f"💭 ReAct thoughts ({len(extra['thoughts'])})", expanded=False):
                for i, t in enumerate(extra["thoughts"]):
                    st.markdown(f"**{i+1}.** {t}")
        if extra.get("critiques"):
            with st.expander(f"🔍 Self-Refine critiques ({len(extra['critiques'])})", expanded=False):
                for i, c_text in enumerate(extra["critiques"]):
                    st.markdown(f"**Iter {i+1}:** {c_text}")
    else:
        st.subheader("📋 Plan")

    done_steps = st.session_state.completed[-5:]
    for s in done_steps:
        lbl = f"{s.get('action','')}  {s.get('object','')}"
        if s.get("target"): lbl += f"  →  {s['target']}"
        st.markdown(f'<div class="step-card step-done">✓ {lbl}</div>', unsafe_allow_html=True)

    for i, s in enumerate(st.session_state.plan):
        lbl = f"{s.get('action','')}  {s.get('object','')}"
        if s.get("target"): lbl += f"  →  {s['target']}"
        css    = "step-active" if i == 0 else "step-pending"
        marker = "▶" if i == 0 else f"{i+1}."
        rsn    = s.get("reason","")
        rsn_html = f'<br><span style="font-size:11px;color:var(--color-text-secondary);font-style:italic">{rsn}</span>' if rsn else ""
        st.markdown(f'<div class="step-card {css}">{marker} {lbl}{rsn_html}</div>', unsafe_allow_html=True)

    st.write("")
    if st.session_state.running and st.session_state.plan:
        b1, b2 = st.columns(2)
        with b1:
            if st.button("⏩ Next step", type="primary", use_container_width=True):
                do_step(); st.rerun()
        with b2:
            if st.button("⏩⏩ Run all", use_container_width=True):
                while st.session_state.plan and st.session_state.running:
                    do_step()
                st.rerun()

with col_env:
    st.subheader("🌐 Environment")
    if st.session_state.obs:
        st.markdown(f'<div class="obs-box">{st.session_state.obs}</div>', unsafe_allow_html=True)
        if st.session_state.visible_objects:
            st.write("")
            with st.expander(f"Visible objects ({len(st.session_state.visible_objects)})", expanded=True):
                st.write(", ".join(f"`{o}`" for o in st.session_state.visible_objects))
        if mode == "simulation":
            st.caption("⚡ Simulated observation · object interactions are rule-based")
        else:
            st.caption(f"📡 Live from AI2-THOR ({backend_desc})")
        total_reward = sum(e.get("reward",0) for e in st.session_state.log if e.get("type")=="step")
        if total_reward:
            st.metric("Cumulative reward", f"{total_reward:.1f}")
    else:
        st.info("Environment state appears here after plan generation.")

with col_log:
    st.subheader("📜 Log")
    if not st.session_state.log:
        st.info("Execution log appears here as steps run.")
    for entry in reversed(st.session_state.log):
        if entry["type"] == "plan":
            bm = entry.get("metrics")
            with st.expander(
                f"📋 {bm.method if bm else 'Plan'} — {len(entry['steps'])} steps"
                + (f" · {bm.latency_s:.1f}s · {bm.llm_calls} call(s)" if bm else ""),
                expanded=True,
            ):
                for s in entry["steps"]:
                    st.markdown(
                        f"- `{s.get('action','')}` **{s.get('object','')}**"
                        + (f" → {s['target']}" if s.get("target") else "")
                        + (f"  *{s.get('reason','')}*" if s.get("reason") else "")
                    )
        elif entry["type"] == "step":
            s   = entry["step"]
            lbl = f"{s.get('action','')} {s.get('object','')}"
            if s.get("target"): lbl += f" → {s['target']}"
            if entry["success"]:
                st.success(f"✅ {lbl}  (+{entry.get('reward',0):.1f})")
            else:
                st.error(f"❌ {lbl}  — {entry.get('msg','')}")
        elif entry["type"] == "done":
            st.balloons()
            st.success("🎉 Task completed!", icon="✅")

# ── Benchmark ────────────────────────────────────────────────────────
bench = st.session_state.bench_history
if len(bench) >= 2:
    st.divider()
    st.subheader("📊 Method Comparison")
    cc, _ = st.columns([1,5])
    if cc.button("🗑 Clear"):
        st.session_state.bench_history = []
        st.rerun()
    import pandas as pd
    rows = [{"#":i+1,"Method":bm.method,"Model":bm.model.split(":")[0],
             "Latency (s)":round(bm.latency_s,2),"LLM calls":bm.llm_calls,
             "Steps":bm.num_steps,"Tokens":bm.total_tokens,
             "Tok/step":round(bm.tokens_per_step,0),"Parse":("✅" if bm.parse_ok else "❌")}
            for i, bm in enumerate(bench)]
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)
    if len(df) > 1:
        c1,c2,c3 = st.columns(3)
        with c1: st.caption("Latency (s)");    st.bar_chart(df.set_index("#")["Latency (s)"], height=170)
        with c2: st.caption("LLM calls");      st.bar_chart(df.set_index("#")["LLM calls"],   height=170)
        with c3: st.caption("Tokens / step");  st.bar_chart(df.set_index("#")["Tok/step"],    height=170)

st.divider()
st.markdown(
    "<div style='text-align:center;font-size:13px;color:var(--color-text-secondary)'>"
    "PyPlanner · 7 planning algorithms · Ollama / OpenAI / Anthropic · "
    "AI2-THOR evaluation pipeline · "
    "<a href='https://github.com/your-repo'>GitHub</a>"
    "</div>",
    unsafe_allow_html=True,
)