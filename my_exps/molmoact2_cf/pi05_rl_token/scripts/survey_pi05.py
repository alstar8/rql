"""What do we already have for fine-tuning pi0.5, and can the RL token be read from it?

Read-only survey. Answers, in one pass, the questions that decide the plan:

  demos       is there an assembled demonstration set for Pick, and in what format
  generator   can MolmoSpaces produce more on its own
  openpi      is there a PyTorch pi0.5 here, with training configs
  checkpoints what pi0.5 weights exist locally or in the HF cache
  hook        does the pi0.5 implementation expose the prefix embeddings that
              equation (1) of RL Token reads

    python scripts/survey_pi05.py

Touches nothing. In particular it never goes near `test.py`, which holds the job.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

B1K = Path("/home/jovyan/users/staroverov/B1K")
AIRI = B1K / "B1K_AIRI"
CF = AIRI / "submodules/rql/my_exps/molmoact2_cf"

DATASET_ROOTS = [
    Path("/weka/prior/datasets/robomolmo"),
    B1K / "datasets",
    B1K / "mlspaces/datasets",
    AIRI / "submodules/molmospaces/mbdata",
    Path.home() / ".cache/lerobot",
    Path("/home/jovyan/datasets"),
    Path("/home/jovyan/users/staroverov/datasets"),
]

HF_CACHE = [Path.home() / ".cache/huggingface/hub", Path("/home/jovyan/.cache/huggingface/hub")]


def head(title: str) -> None:
    print(f"\n{'=' * 78}\n== {title}\n{'=' * 78}")


def run(cmd: str, limit: int = 40) -> str:
    try:
        out = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300).stdout
    except subprocess.TimeoutExpired:
        return "(timed out)"
    lines = out.splitlines()
    return "\n".join(lines[:limit]) + ("\n  ..." if len(lines) > limit else "")


def show_dir(path: Path, depth: int = 1, limit: int = 25) -> None:
    if not path.exists():
        print(f"  {path}  -- absent")
        return
    size = run(f"du -sh {path} 2>/dev/null | cut -f1", limit=1).strip()
    print(f"  {path}  ({size or '?'})")
    print(run(f"find {path} -maxdepth {depth} -mindepth 1 | head -{limit}", limit=limit))


def demo_sources() -> None:
    head("1. where the demo corpus came from")
    for name in ("launch_rlt_pretrain_demo1k.sh", "encode_offline_demo_tokens.py", "droid_ipec_loader.py"):
        path = CF / name
        if not path.exists():
            print(f"  {name}: absent")
            continue
        print(f"\n--- {name}: paths it names")
        print(run(f"grep -nE '/(weka|home|workspace|data)[A-Za-z0-9_/.-]*' {path} | head -25", limit=25))

    head("2. dataset roots")
    for root in DATASET_ROOTS:
        show_dir(root)

    head("3. anything that looks like a trajectory store")
    print(run(
        "find /home/jovyan/users/staroverov/B1K /weka/prior/datasets 2>/dev/null "
        "-maxdepth 6 \\( -name '*.h5' -o -name '*.hdf5' -o -name 'meta' -o -name '*.parquet' \\) "
        "| head -30", limit=30))


def molmospaces_generator() -> None:
    head("4. can MolmoSpaces generate demonstrations itself")
    ms = AIRI / "submodules/molmospaces"
    print(run(f"ls {ms}/molmo_spaces/policy/solvers/object_manipulation/ 2>/dev/null", limit=20))
    print("\n--- data generation entry points")
    print(run(f"ls {ms}/scripts/ {ms}/bin/ 2>/dev/null | head -30", limit=30))


def openpi_and_torch() -> None:
    head("5. openpi / pi0.5 on this machine")
    print(run(
        "find /home/jovyan/users/staroverov/B1K -maxdepth 4 -type d "
        "\\( -name 'openpi*' -o -name '*pi0*' -o -name 'lerobot*' \\) 2>/dev/null | head -20", limit=20))

    for candidate in (AIRI / "submodules/openpi", B1K / "openpi"):
        if not candidate.exists():
            continue
        print(f"\n--- {candidate}")
        print(run(f"ls {candidate}", limit=20))
        print("  pytorch implementation:")
        print(run(f"find {candidate} -maxdepth 4 -type d -name '*pytorch*' -o -maxdepth 4 -name '*pytorch*.py' | head -10", limit=10))
        print("  training configs mentioning pi05:")
        print(run(f"grep -rln 'pi05\\|pi0_5\\|pi0.5' {candidate}/src 2>/dev/null | head -15", limit=15))

    print("\n--- torch/jax in the venvs")
    for venv in (AIRI / "submodules/molmospaces/.venv", AIRI / "submodules/molmoact2/.venv"):
        pip = venv / "bin/pip"
        if pip.exists():
            print(f"  {venv.parent.name}: " + run(
                f"{pip} list 2>/dev/null | grep -iE '^(torch|jax|jaxlib|flax|openpi|lerobot|transformers) ' | tr '\\n' ' '",
                limit=1))


def checkpoints() -> None:
    head("6. pi0.5 checkpoints already here")
    for cache in HF_CACHE:
        if cache.exists():
            print(f"  {cache}:")
            print(run(f"ls {cache} | grep -iE 'pi0|pi-0|physical|paligemma|lerobot' | head -20", limit=20))
    print("\n  local checkpoint-looking directories:")
    print(run(
        "find /home/jovyan/users/staroverov/B1K -maxdepth 5 -type d "
        "\\( -iname '*pi05*' -o -iname '*pi0_5*' -o -iname '*checkpoint*' \\) 2>/dev/null | head -25", limit=25))


def rl_token_hook() -> None:
    head("7. can the RL token be read from this pi0.5 implementation")
    print("  RL Token eq. (1) needs the VLA's final-layer token embeddings for (state, language).")
    print("  For MolmoAct2 we hooked the inner VLM and reused the action-generation prefill.")
    print("  For pi0.5 the same place is the VLM prefix the action expert attends to.\n")
    for candidate in (AIRI / "submodules/openpi", B1K / "openpi"):
        if not candidate.exists():
            continue
        print(run(
            f"grep -rn 'output_hidden_states\\|hidden_states\\|last_hidden_state\\|def embed_prefix\\|kv_cache' "
            f"{candidate}/src 2>/dev/null | head -20", limit=20))


def main() -> None:
    print(f"host {os.uname().nodename}   python {sys.version.split()[0]}")
    print(run("nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader", limit=10))
    demo_sources()
    molmospaces_generator()
    openpi_and_torch()
    checkpoints()
    rl_token_hook()
    print("\ndone -- nothing was modified")


if __name__ == "__main__":
    main()
