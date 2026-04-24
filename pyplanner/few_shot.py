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
from pyplanner.few_shot_examples import FEW_SHOT_EXAMPLES

from pyplanner.base import (
    ACTIONS_STR, JSON_EXAMPLE, STEP_SCHEMA,
    BasePlanner, PlanMetrics, parse_steps,
)


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
