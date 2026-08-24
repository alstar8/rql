"""Serve pi0.5 to the MolmoSpaces evaluator while recording its token sequences.

Same websocket contract as scripts/serve_pi05.py, so the evaluator connects unchanged,
with two additions: the tokens behind every action are written to disk for phase-1
training, and the delta actions are converted to the absolute joint targets the simulator
executes.

    python scripts/serve_pi05_tokens.py --checkpoint <dir> --port 8080 \
        --record-tokens <dir> --tag house21

Holds a GPU while it runs, and flushes any partial shard on shutdown so the last
sequences are not lost.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from openpi.policies import policy_config as _policy_config  # noqa: E402
from openpi.serving import websocket_policy_server  # noqa: E402
from openpi.training import config as _config  # noqa: E402

from pi05.token_recorder import TokenRecordingPolicy  # noqa: E402
from pi05.vla_server import FrozenPi05  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default="pi05_droid_finetune")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--record-tokens", type=Path, default=None)
    ap.add_argument("--tag", default="tokens", help="goes into the shard filename")
    ap.add_argument("--shard-size", type=int, default=256)
    ap.add_argument("--max-sequences", type=int, default=None,
                    help="stop recording after this many sequences, to bound disk use")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    policy = _policy_config.create_trained_policy(
        _config.get_config(args.config), args.checkpoint, pytorch_device="cuda"
    )
    frozen = FrozenPi05(policy)
    served = TokenRecordingPolicy(
        frozen,
        args.record_tokens,
        shard_size=args.shard_size,
        tag=args.tag,
        max_sequences=args.max_sequences,
    )

    def shutdown(_signum, _frame):
        total = served.flush()
        logging.info("flushed on shutdown, %d sequences recorded", total)
        sys.exit(0)

    # The evaluator kills the server when it finishes, so the last partial shard would be
    # lost without this.
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    logging.info("checkpoint %s", args.checkpoint)
    logging.info("recording %s", args.record_tokens or "disabled")
    logging.info("listening on port %d", args.port)
    websocket_policy_server.WebsocketPolicyServer(
        served, host="0.0.0.0", port=args.port,
        metadata={"policy_name": "pi05_rlt_frozen", "tag": args.tag},
    ).serve_forever()


if __name__ == "__main__":
    main()
