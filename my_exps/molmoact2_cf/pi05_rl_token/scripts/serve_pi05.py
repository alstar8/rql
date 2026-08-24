"""Serve a pi0.5 checkpoint to the MolmoSpaces pi_policy eval client.

    python scripts/serve_pi05.py --checkpoint <dir> --port 8080

Speaks openpi's websocket protocol, so PiPolicyEvalConfig connects unchanged. Predicted
joint deltas become the absolute targets the simulator executes; run the eval with
chunk_size=1 so that conversion stays exact.

Holds a GPU while it runs. Stop it when the benchmark finishes.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from openpi.policies import policy_config as _policy_config  # noqa: E402
from openpi.serving import websocket_policy_server  # noqa: E402
from openpi.training import config as _config  # noqa: E402

from pi05.serve import DeltaToAbsolutePolicy  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, help="checkpoint directory containing params/")
    ap.add_argument("--config", default="pi05_droid_finetune", help="openpi training config name")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument(
        "--absolute",
        action="store_true",
        help="serve raw model output, for a model trained on absolute targets",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    train_config = _config.get_config(args.config)
    policy = _policy_config.create_trained_policy(train_config, args.checkpoint)

    if args.absolute:
        logging.info("serving raw model output (absolute action space)")
        served = policy
    else:
        logging.info("serving delta predictions converted to absolute joint targets")
        served = DeltaToAbsolutePolicy(policy)

    logging.info("checkpoint: %s", args.checkpoint)
    logging.info("listening on port %d", args.port)
    websocket_policy_server.WebsocketPolicyServer(
        served,
        host="0.0.0.0",
        port=args.port,
        metadata={"policy_name": "pi05_molmospaces_pick", "checkpoint": str(args.checkpoint)},
    ).serve_forever()


if __name__ == "__main__":
    main()
