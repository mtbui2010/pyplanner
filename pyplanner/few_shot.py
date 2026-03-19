# planners/few_shot.py
# Method 4 — Few-Shot Chain-of-Thought
#
# Provides 3 hand-crafted example (task → reasoning → plan) pairs in the prompt
# so the model can mimic the pattern rather than invent it.
#
# Advantages over plain CoT:
# - The examples anchor object names and action vocabulary to known-good patterns
# - Less likely to generate hallucinated actions or wrong action order
# - Still a single LLM call — same latency profile as CoT
#
# The examples cover: kitchen (grab+use appliance), bathroom (multi-step routine),
# and living room (navigate+interact). This gives the model diverse context.

import re
import time

from pyplanner.base import (
    ACTIONS_STR, JSON_EXAMPLE, STEP_SCHEMA,
    BasePlanner, PlanMetrics, parse_steps,
)

FEW_SHOT_EXAMPLES = '''
=== EXAMPLE 1 ===
Task: Make a cup of coffee
Visible objects: coffee_machine, mug, counter_top, fridge

Reasoning:
- Goal: activate the coffee machine and have coffee in a mug
- Objects needed: coffee_machine, mug
- Order: navigate to mug → grab mug → place mug under machine → navigate to machine → turn on machine
- Risk: mug must be placed before turning on machine

Plan:
{"steps": [
  {"action": "Navigate", "object": "mug",            "target": "",             "reason": "Move to the mug"},
  {"action": "Grab",     "object": "mug",            "target": "",             "reason": "Pick up the mug"},
  {"action": "Navigate", "object": "coffee_machine", "target": "",             "reason": "Move to the coffee machine"},
  {"action": "Place",    "object": "mug",            "target": "coffee_machine","reason": "Put mug under the dispenser"},
  {"action": "TurnOn",   "object": "coffee_machine", "target": "",             "reason": "Start brewing coffee"}
]}

=== EXAMPLE 2 ===
Task: Brush teeth
Visible objects: toothbrush, toothpaste, sink, faucet

Reasoning:
- Goal: complete the tooth-brushing routine
- Objects needed: toothbrush, sink
- Order: navigate toothbrush → grab → navigate sink → turn on faucet → wash toothbrush → turn off faucet
- Risk: must reach sink before turning it on

Plan:
{"steps": [
  {"action": "Navigate", "object": "toothbrush", "target": "",    "reason": "Move to toothbrush"},
  {"action": "Grab",     "object": "toothbrush", "target": "",    "reason": "Pick up the toothbrush"},
  {"action": "Navigate", "object": "sink",        "target": "",    "reason": "Move to the sink"},
  {"action": "TurnOn",   "object": "faucet",      "target": "",    "reason": "Start water flow"},
  {"action": "Wash",     "object": "toothbrush",  "target": "",    "reason": "Wet and clean the brush"},
  {"action": "TurnOff",  "object": "faucet",      "target": "",    "reason": "Stop the water"}
]}

=== EXAMPLE 3 ===
Task: Watch TV
Visible objects: television, remote_control, sofa

Reasoning:
- Goal: turn on TV and sit to watch
- Objects needed: television (or remote), sofa
- Order: navigate TV → turn on → navigate sofa → sit
- Risk: sitting before turning on TV is inefficient but acceptable; turning on TV first is cleaner

Plan:
{"steps": [
  {"action": "Navigate", "object": "television",     "target": "", "reason": "Move to the TV"},
  {"action": "TurnOn",   "object": "television",     "target": "", "reason": "Switch the TV on"},
  {"action": "Navigate", "object": "sofa",           "target": "", "reason": "Move to the sofa"},
  {"action": "Sit",      "object": "sofa",           "target": "", "reason": "Sit down to watch"}
]}
'''

SYSTEM_PROMPT = f"""You are a household assistant robot planner.
Study the examples below, then generate a plan for the new task using the SAME format.

Available robot actions:
{ACTIONS_STR}

{STEP_SCHEMA}

{FEW_SHOT_EXAMPLES}

For the new task, output EXACTLY:
Reasoning:
<your reasoning>

Plan:
{JSON_EXAMPLE}

Do not output anything else."""


def _extract(raw: str):
    reasoning = ""
    rm = re.search(r"Reasoning:\s*(.*?)(?=Plan:|$)", raw, re.DOTALL | re.IGNORECASE)
    if rm:
        reasoning = rm.group(1).strip()
    plan_raw = raw
    pm = re.search(r"Plan:\s*(\{.*)", raw, re.DOTALL | re.IGNORECASE)
    if pm:
        plan_raw = pm.group(1).strip()
    return reasoning, plan_raw


class FewShotPlanner(BasePlanner):
    name        = "Few-Shot CoT"
    description = (
        "3 hand-crafted examples anchor the model's output style. "
        "Single call like CoT but more consistent object/action naming."
    )

    def generate_plan(self, task, obs, visible_objects):
        t0 = time.perf_counter()
        user_msg = (
            self._context_str(task, obs, visible_objects)
            + "\n\nNow generate the Reasoning and Plan for this task:"
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ]
        try:
            raw, in_tok, out_tok = self._chat(messages, temperature=0.2)
            reasoning, plan_raw  = _extract(raw)
            steps = parse_steps(plan_raw)
            ok    = bool(steps)
        except Exception as e:
            reasoning, steps = str(e), []
            in_tok, out_tok, ok = 0, 0, False

        metrics = PlanMetrics(
            method        = self.name,
            model         = self.model,
            backend       = self.provider,
            latency_s     = time.perf_counter() - t0,
            llm_calls     = 1,
            input_tokens  = in_tok,
            output_tokens = out_tok,
            num_steps     = len(steps),
            parse_ok      = ok,
            extra         = {"reasoning": reasoning},
        )
        print(f"[{self.name}] {len(steps)} steps in {metrics.latency_s:.1f}s")
        return steps, metrics

    def replan(self, task, completed, failed_step, failure_reason, obs, visible_objects):
        t0 = time.perf_counter()
        user_msg = (
            self._replan_context(task, completed, failed_step, failure_reason, obs, visible_objects)
            + "\n\nNow generate the Reasoning and remaining Plan:"
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ]
        try:
            raw, in_tok, out_tok = self._chat(messages, temperature=0.2)
            reasoning, plan_raw  = _extract(raw)
            steps = parse_steps(plan_raw)
            ok    = bool(steps)
        except Exception as e:
            reasoning, steps = str(e), []
            in_tok, out_tok, ok = 0, 0, False

        metrics = PlanMetrics(
            method        = self.name,
            model         = self.model,
            backend       = self.provider,
            latency_s     = time.perf_counter() - t0,
            llm_calls     = 1,
            input_tokens  = in_tok,
            output_tokens = out_tok,
            num_steps     = len(steps),
            parse_ok      = ok,
            extra         = {"reasoning": reasoning},
        )
        return steps, metrics
