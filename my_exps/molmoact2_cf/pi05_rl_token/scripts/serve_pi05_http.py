"""Frozen pi0.5 over HTTP: one server for evaluation and for RL training alike.

    POST /act  {"external_cam", "wrist_cam", "instruction", "state", "want_tokens"}
            -> {"actions": (16, 8), "converts_delta_to_absolute": false,
                "token_features": (968, 2048), "token_attention_mask": (968,)}

    POST /reload {"checkpoint"} -> {"status": "ok", "copied": N, "reloads": K}

    GET  /act -> status, including converts_delta_to_absolute

WHAT IS ON THE WIRE
-------------------
Actions leave here exactly as the model produces them: joint DELTAS. The conversion into
the absolute joint targets MolmoSpaces executes belongs to the client, in
`pi05/model.py`, because only the client knows which arm state each action of a chunk
should be resolved against. Keeping it there also means the RL actor can be handed either
space deliberately (`rl_action_space`) rather than whichever one the server happened to
produce.

`converts_delta_to_absolute` is reported so a client can refuse to run against a server
that converts. Adding the arm state twice drives the arm toward a folded pose, which
looks like a bad policy rather than a misconfiguration -- that mistake is the whole
reason the published pi0.5 baseline on this benchmark reads 0/10.

`--convert` restores the old server-side behaviour for reproducing earlier runs. Nothing
current uses it.

Tokens are ~7.9 MB per call, so they are only computed and sent when the client asks.
A client that omits `want_tokens` gets them, which keeps rlt/vla.py working unchanged.

Built on the standard library rather than FastAPI, which is not installed in any
environment here; json_numpy is used because the client patches it, so the wire format
has to match.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import json_numpy
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

json_numpy.patch()

log = logging.getLogger(__name__)

ARM_DOF = 7


def make_handler(frozen, checkpoint: str, convert: bool, recorder=None):
    stats = {"calls": 0, "seconds": 0.0, "checkpoint": checkpoint, "reloads": 0}
    # Inference is not reentrant: the token capture keeps per-call state, so concurrent
    # requests would mix one call's tokens into another's response. Reload takes the
    # same lock so a swap cannot tear an in-flight forward.
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):  # quiet; the run logs are noisy enough
            return

        def _send(self, payload: dict, status: int = 200) -> None:
            body = json_numpy.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            self._send(
                {
                    "status": "ok",
                    "checkpoint": stats["checkpoint"],
                    "reloads": stats["reloads"],
                    "converts_delta_to_absolute": convert,
                    "calls": stats["calls"],
                    "mean_seconds": round(stats["seconds"] / max(1, stats["calls"]), 4),
                    "tokens_recorded": recorder.written if recorder is not None else 0,
                }
            )

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length)
            route = self.path.split("?", 1)[0].rstrip("/") or "/"
            if route == "/reload":
                payload = json_numpy.loads(raw.decode("utf-8"))
                ckpt = str(payload.get("checkpoint") or "")
                if not ckpt:
                    self._send({"error": "checkpoint is required"}, status=400)
                    return
                try:
                    with lock:
                        copied = frozen.reload_expert(ckpt)
                        stats["checkpoint"] = ckpt
                        stats["reloads"] += 1
                except Exception as exc:  # noqa: BLE001
                    log.exception("reload failed")
                    self._send({"error": str(exc)}, status=500)
                    return
                log.info("serving expert from %s (%d tensors)", ckpt, copied)
                self._send(
                    {
                        "status": "ok",
                        "checkpoint": ckpt,
                        "copied": copied,
                        "reloads": stats["reloads"],
                    }
                )
                return

            payload = json_numpy.loads(raw.decode("utf-8"))
            started = time.perf_counter()
            try:
                state = np.asarray(payload["state"], dtype=np.float32).reshape(-1)
                if state.shape != (8,):
                    raise ValueError(f"state must be (8,), got {state.shape}")
                # Absent means yes, so a client written against the older contract still
                # receives what it expects.
                want_tokens = bool(payload.get("want_tokens", True))

                observation = {
                    "observation/exterior_image_1_left": np.asarray(
                        payload["external_cam"], dtype=np.uint8
                    ),
                    "observation/wrist_image_left": np.asarray(
                        payload["wrist_cam"], dtype=np.uint8
                    ),
                    "observation/joint_position": state[:ARM_DOF],
                    "observation/gripper_position": state[ARM_DOF:8],
                    "prompt": str(payload["instruction"]),
                }
                with lock:
                    out = frozen.predict(observation, want_tokens=want_tokens)
                    # Recorded here rather than client-side: this is the one place the
                    # tokens already exist, so a corpus can never disagree with the
                    # actions the same call returned.
                    if recorder is not None and want_tokens:
                        recorder.append(out["token_features"], out["token_attention_mask"])

                actions = np.array(out["actions"], dtype=np.float32, copy=True)
                if convert:
                    actions[..., :ARM_DOF] += state[:ARM_DOF]
                out = {**out, "actions": actions, "converts_delta_to_absolute": convert}
            except Exception as exc:  # noqa: BLE001
                log.exception("inference failed")
                self._send({"error": str(exc)}, status=500)
                return
            stats["calls"] += 1
            stats["seconds"] += time.perf_counter() - started
            self._send(out)

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default="pi05_droid_finetune")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument(
        "--convert",
        action="store_true",
        help="add the arm state server-side, as the pre-refactor server did. Only for "
             "reproducing earlier runs; current clients convert themselves and will "
             "refuse to talk to a server started with this.",
    )
    ap.add_argument(
        "--record-tokens",
        type=Path,
        default=None,
        help="write the prefix token sequences here as phase-1 training data",
    )
    ap.add_argument("--record-tag", default="tokens", help="shard name prefix, e.g. the scene")
    ap.add_argument(
        "--max-sequences",
        type=int,
        default=0,
        help="stop recording after this many sequences (0 = no cap). At 4 MB each, "
             "5000 sequences is ~20 GB per scene.",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    os.environ["TORCHINDUCTOR_CUDAGRAPHS"] = "0"
    from pi05.eval import disable_policy_compile

    disable_policy_compile()

    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    from pi05.vla_server import FrozenPi05

    policy = _policy_config.create_trained_policy(
        _config.get_config(args.config), args.checkpoint, pytorch_device="cuda"
    )
    frozen = FrozenPi05(policy)

    if args.convert:
        log.warning("serving ABSOLUTE joint targets: the arm state is added here")
    else:
        log.info("serving joint DELTAS; the client converts (pi05/model.py)")

    recorder = None
    if args.record_tokens is not None:
        from pi05.token_recorder import TokenShardWriter

        recorder = TokenShardWriter(
            args.record_tokens,
            tag=args.record_tag,
            max_sequences=args.max_sequences or None,
            provenance={
                "checkpoint": args.checkpoint,
                "openpi_config": args.config,
                "converts_delta_to_absolute": args.convert,
                "collected_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "note": "actions are joint deltas; the client owns chunk_size and "
                        "conversion, and records them in its own run config.json",
            },
        )

    server = ThreadingHTTPServer(
        (args.host, args.port), make_handler(frozen, args.checkpoint, args.convert, recorder)
    )
    log.info("frozen pi0.5 serving on http://%s:%d/act", args.host, args.port)

    # SIGTERM's default action ends the process outright, which would skip the flush and
    # lose up to a shard's worth of sequences. The eval driver stops the server with
    # SIGTERM, so it has to unwind cleanly instead.
    def _stop(signum, _frame):
        log.info("signal %d: shutting down", signum)
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    try:
        server.serve_forever()
    finally:
        # A partial shard is otherwise lost, and with it up to 255 sequences.
        if recorder is not None:
            log.info("flushed %d token sequences", recorder.flush())


if __name__ == "__main__":
    main()
