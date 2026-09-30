#!/usr/bin/env python3
"""Runs the eval cases through local Ollama and checks each prompt against its system prompt's rules."""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

EVALS_DIR = Path(__file__).resolve().parent
ROOT = EVALS_DIR.parent
sys.path.insert(0, str(EVALS_DIR))

from cases import CASES  # noqa: E402
from checks import check_krea2, check_minimax, stated_duration  # noqa: E402


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "models" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_case(case: dict, modules: dict, model_override: str) -> dict:
    """Generates one prompt the way the Function does after captioning, then checks it."""
    mod = modules[case["function"]]
    v = mod.Pipe().valves
    model = model_override or v.TEXT_MODEL
    text, caption, duration = case.get("text", ""), case.get("caption", ""), case.get("duration")
    system = mod.SYSTEM_PROMPT if case["function"] == "krea2" else mod.SYSTEM_PROMPTS[case["mode"]]
    idea = mod.build_idea(text, caption, duration)

    start = time.monotonic()
    raw = mod.ollama_generate(v.OLLAMA_BASE_URL, model, system, idea, v.REQUEST_TIMEOUT_SECONDS, v.TEXT_NUM_CTX)
    seconds = time.monotonic() - start
    output = mod.tidy_output(raw)
    if case["function"] == "krea2":
        problems = check_krea2(output)
    else:
        output = mod.frame_not_image(output)
        problems = check_minimax(output, case["mode"], duration or stated_duration(text))
    return {"id": case["id"], "model": model, "seconds": round(seconds, 1), "output": output, "problems": problems}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="writer model to test (default: the Functions' TEXT_MODEL)")
    parser.add_argument("--case", help="only run cases whose id contains this, e.g. krea2 or fl2va")
    parser.add_argument("--repeat", type=int, default=1, help="runs per case; output varies run to run")
    parser.add_argument("--verbose", action="store_true", help="print every output, not just failures")
    args = parser.parse_args()

    modules = {"krea2": load("krea2"), "minimax": load("minimax_h3")}
    cases = [c for c in CASES if not args.case or args.case in c["id"]]
    if not cases:
        sys.exit(f"no case id contains {args.case!r}")

    results = []
    for case in cases:
        for _ in range(max(1, args.repeat)):
            try:
                result = run_case(case, modules, args.model)
            except modules["krea2"].PIPELINE_ERRORS as exc:
                sys.exit(f"{case['id']}: {modules['krea2'].pipeline_error(exc, False)}")
            results.append(result)
            status = "FAIL" if result["problems"] else "PASS"
            print(f"{status}  {result['id']}  ({result['seconds']}s)", flush=True)
            for problem in result["problems"]:
                print(f"      - {problem}")
            if result["problems"] or args.verbose:
                print("      " + result["output"].replace("\n", "\n      ") + "\n")

    out_dir = EVALS_DIR / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{time.strftime('%Y%m%d-%H%M%S')}.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    passed = sum(not r["problems"] for r in results)
    print(f"\n{passed}/{len(results)} passed; outputs saved to {out_path.relative_to(ROOT)}")
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
