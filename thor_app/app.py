# app.py — Daily Assistant Robot (AI2-THOR + Ollama + Streamlit)
# Run: streamlit run app.py
# Requires: python thor_server.py running in another terminal
#
# Setup (run once):
#   pip install -e ../pyplanner
# Or if pip install is not possible, the path-fallback below handles it automatically.

import sys
import os
import time

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

from knowledge import TASKS, CATEGORIES, get_task_info
from thor_client import ThorClient
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
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

if st.session_state.client is None:
    st.session_state.client = ThorClient(host="localhost", port=5555)

client: ThorClient = st.session_state.client


# ══════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════
def fmt_step(step: dict) -> str:
    label = f"{step.get('action', '')}  {step.get('object', '')}"
    if step.get("target"):
        label += f"  →  {step['target']}"
    return label


def do_step():
    """Execute next step from plan. Trigger replan on failure."""
    if not st.session_state.plan:
        return

    step   = st.session_state.plan.pop(0)
    result = client.step(
        step.get("action", "Wait"),
        step.get("object", ""),
        step.get("target", ""),
    )

    st.session_state.obs             = result.get("obs", st.session_state.obs)
    st.session_state.visible_objects = result.get("visible_objects", [])

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
        if planner is not None:
            new_plan, replan_metrics = planner.replan(
                st.session_state.task_label,
                st.session_state.completed,
                step,
                result.get("msg", "step failed"),
                result.get("obs", ""),
                st.session_state.visible_objects,
            )
        else:
            new_plan, replan_metrics = [], None

        st.session_state.plan = new_plan
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

    # ── Connection status ──
    if client.connected:
        st.success("Simulator connected", icon="🟢")
    else:
        st.error("Simulator offline", icon="🔴")
        st.code("python thor_server.py", language="bash")
        if st.button("↺ Retry connection", use_container_width=True):
            client.reconnect()
            st.rerun()

    st.divider()

    # ── Task selection ──
    st.subheader("Task")

    mode = st.radio(
        "input_mode",
        ["Browse", "Type freely"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if mode == "Browse":
        cat        = st.selectbox("Category", list(CATEGORIES.keys()))
        task_label = st.selectbox("Task", CATEGORIES[cat])
        task_info  = get_task_info(task_label)
        st.caption(f"_{task_info['desc']}_")
    else:
        # custom = st.text_input(
        #     "Describe task",
        #     placeholder="make coffee, wash hands, watch TV...",
        # )
        # if not custom.strip():
        #     st.info("Type a task above.")
        #     st.stop()
        # task_label = match_task_from_text(custom)
        # task_info  = get_task_info(task_label)
        # st.caption(f"Matched: **{task_label}**")
        # st.caption(f"_{task_info['desc']}_")
        custom = st.text_input(
            "Describe task",
            placeholder="pick up the apple, make coffee, wash hands...",
        )
        if not custom.strip():
            st.info("Type a task above.")
            st.stop()

        # Không cần match — dùng thẳng input làm task description
        task_label = custom.strip()

        # Chỉ cần scene — chọn room type đơn giản
        room_for_task = st.selectbox(
            "Which room?",
            ["Kitchen", "Living room", "Bedroom", "Bathroom"],
        )
        lo, hi = PLAN_RANGES[room_for_task]
        task_info = {
            "scene": f"FloorPlan{lo}",
            "desc":  custom.strip(),
        }

    st.divider()

    # ── Scene picker ──
    st.subheader("Scene")

    scene_mode = st.radio(
        "scene_mode",
        ["Auto (from task)", "Manual pick"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if scene_mode == "Manual pick":
        room_type = st.selectbox(
            "Room type",
            list(PLAN_RANGES.keys()),
            index=0,
        )
        lo, hi = PLAN_RANGES[room_type]
        plan_num = st.slider(
            f"FloorPlan number ({lo}–{hi})",
            min_value=lo,
            max_value=hi,
            value=lo,
            step=1,
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
                st.session_state.visible_objects = resp.get("visible_objects", [])
                st.session_state.task_scene      = final_scene
                st.caption(f"✅ Scene `{final_scene}` loaded")
            else:
                st.warning(f"Could not load `{final_scene}`: {resp.get('msg', '')}")
        else:
            st.caption(f"Scene: `{final_scene}`")
    else:
        st.session_state.pop("_last_manual_scene", None)
        final_scene = task_info["scene"]
        st.caption(f"Auto scene: `{final_scene}`")

    st.divider()

    # ── Config ──
    st.subheader("Config")

    # ── Planning method ──
    method_names    = list(REGISTRY.keys())
    selected_method = st.selectbox(
        "🧩 Planning method",
        method_names,
        index=0,
        help="Algorithm used to generate the action plan.",
    )
    st.caption(f"_{REGISTRY[selected_method].description}_")

    # Method-specific options
    react_max_steps    = 15
    refine_iterations  = 2
    router_backend     = "openai"
    router_model       = ""
    router_openai_key  = ""
    router_claude_key  = ""

    if selected_method == "ReAct":
        react_max_steps = st.slider("Max steps (ReAct)", 5, 20, 15)
    elif selected_method == "Self-Refine":
        refine_iterations = st.slider("Refine iterations", 1, 4, 2)
    elif selected_method == "LLM Router":
        router_backend = st.radio(
            "Verifier backend",
            ["openai", "anthropic"],
            horizontal=True,
            help="Which external API verifies/fixes the local plan.",
        )
        default_vm = "gpt-4o-mini" if router_backend == "openai" else "claude-haiku-4-5-20251001"
        router_model = st.text_input(
            "Verifier model (optional)",
            placeholder=default_vm,
            help=f"Leave empty to use default: {default_vm}",
        )
        if router_backend == "openai":
            router_openai_key = st.text_input(
                "OpenAI API key",
                type="password",
                help="Or set OPENAI_API_KEY env variable.",
            )
        else:
            router_claude_key = st.text_input(
                "Anthropic API key",
                type="password",
                help="Or set ANTHROPIC_API_KEY env variable.",
            )
        st.caption("💡 Local model generates; external API only verifies (cheaper).")

    # ── LLM Provider ──────────────────────────────────────────────────
    st.markdown("**LLM Provider**")

    selected_provider = st.selectbox(
        "Provider",
        ["ollama", "openai", "anthropic"],
        index=0,
        help="ollama = local server · openai = ChatGPT API · anthropic = Claude API",
        label_visibility="collapsed",
    )

    # Host URL — only shown for Ollama
    if selected_provider == "ollama":
        ollama_host = st.text_input(
            "🌐 Ollama host URL",
            value=DEFAULT_HOST,
            placeholder="http://localhost:11434",
            help="URL of the Ollama server (local or remote).",
        )
    else:
        ollama_host = DEFAULT_HOST   # unused but keeps variable defined

    # Model selector — dynamic list per provider
    model_list     = PROVIDER_MODELS[selected_provider]
    selected_model = st.selectbox(
        "🧠 Model",
        model_list,
        index=0,
        help=f"Model for the {selected_provider} provider.",
    )
    # Custom model text override
    custom_model = st.text_input(
        "Custom model name (optional)",
        placeholder=f"e.g. {model_list[0]}",
        help="Override the dropdown — type any model name supported by the provider.",
        label_visibility="visible",
    )
    if custom_model.strip():
        selected_model = custom_model.strip()

    # API key — shown for non-Ollama providers
    planner_api_key = ""
    if selected_provider == "openai":
        planner_api_key = st.text_input(
            "🔑 OpenAI API key",
            type="password",
            help="Starts with sk-... · Or set OPENAI_API_KEY env variable.",
        )
    elif selected_provider == "anthropic":
        planner_api_key = st.text_input(
            "🔑 Anthropic API key",
            type="password",
            help="Starts with sk-ant-... · Or set ANTHROPIC_API_KEY env variable.",
        )

    st.session_state["planner_model"]    = selected_model
    st.session_state["planner_provider"] = selected_provider

    max_replan = st.slider("Max replans", 1, 5, 3)
    st.session_state.max_replan = max_replan

    step_delay = st.slider("Auto-step delay (s)", 0.3, 3.0, 0.8, 0.1)

    auto_exec = st.toggle("Auto-execute all steps", value=False)

    st.divider()

    # ── Start / Reset buttons ──
    col_start, col_reset = st.columns([3, 1])

    with col_start:
        start = st.button(
            "▶ Start",
            type="primary",
            use_container_width=True,
            disabled=not client.connected,
            help="Generate action plan and begin execution",
        )
    with col_reset:
        reset = st.button("↺", use_container_width=True, help="Reset all state")

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
            kwargs["anthropic_api_key"]  = router_claude_key
        planner = pyplanner.get(
            selected_method,
            host     = ollama_host,
            model    = selected_model,
            provider = selected_provider,
            api_key  = planner_api_key,
            **kwargs,
        )
        st.session_state.planner_obj = planner

        # If manual mode already loaded this scene, reuse existing obs
        already_loaded = (
            scene_mode == "Manual pick"
            and st.session_state.get("_last_manual_scene") == final_scene
            and st.session_state.obs
        )

        if already_loaded:
            obs_for_plan     = st.session_state.obs
            visible_for_plan = st.session_state.visible_objects
        else:
            with st.spinner(f"Loading scene `{final_scene}`..."):
                resp = client.reset(final_scene)
            if resp.get("status") != "ok":
                st.error(f"Simulator error: {resp.get('msg', 'unknown')}")
                st.session_state.running = False
                st.stop()
            obs_for_plan     = resp.get("obs", "")
            visible_for_plan = resp.get("visible_objects", [])
            st.session_state.obs             = obs_for_plan
            st.session_state.visible_objects = visible_for_plan

        with st.spinner(f"🧠 [{selected_method} · {selected_provider} · {selected_model}] Generating plan..."):
            plan, metrics = planner.generate_plan(
                task_info["desc"],
                obs_for_plan,
                visible_for_plan,
            )

        st.session_state.plan         = plan
        st.session_state.last_metrics = metrics
        st.session_state.bench_history.append(metrics)
        st.session_state.log.append({"type": "plan", "steps": plan.copy(), "n": 0, "metrics": metrics})
        st.rerun()

    if reset:
        for k, v in DEFAULTS.items():
            if k != "client":
                st.session_state[k] = v
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
    info = get_task_info(st.session_state.task_label)
    st.caption(
        f"**{st.session_state.task_label}**"
        f"  ·  {info['desc']}"
        f"  ·  scene `{st.session_state.task_scene}`"
    )

col_cam, col_plan, col_log = st.columns([1, 1, 1], gap="large")


# ══════════════════════════════════════════════════════════════════════
# COLUMN 1 — Camera + observation
# ══════════════════════════════════════════════════════════════════════
with col_cam:
    st.subheader("🎥 Robot View")
    frame_ph = st.empty()

    frame = client.get_frame() if client.connected else None
    if frame:
        frame_ph.image(frame, use_column_width=True)
    else:
        frame_ph.info("No camera feed.\nStart a task to see robot view.")

    if st.session_state.running:
        if st.button("🔄 Refresh frame", use_container_width=True):
            f = client.get_frame()
            if f:
                frame_ph.image(f, use_column_width=True)

    if st.session_state.obs:
        st.caption("Environment state")
        st.markdown(
            f'<div class="obs-box">{st.session_state.obs}</div>',
            unsafe_allow_html=True,
        )

    if st.session_state.visible_objects:
        with st.expander(
            f"Visible objects ({len(st.session_state.visible_objects)})",
            expanded=False,
        ):
            st.write(", ".join(st.session_state.visible_objects))


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
                    frame_ph.image(f, use_column_width=True)
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
                    frame_ph.image(f, use_column_width=True)
                st.rerun()

    elif not st.session_state.plan and st.session_state.running:
        st.session_state.running = False
        st.rerun()


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
            label = "Initial plan" if n == 0 else f"Replan #{n}"
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