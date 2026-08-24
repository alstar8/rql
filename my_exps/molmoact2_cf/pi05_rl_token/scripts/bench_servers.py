"""How much throughput does a second, third, fourth VLA server on one card buy?

Each server serializes inference behind a lock, so concurrency needs several
processes. Whether they overlap depends on the GPU: a 5.4B forward at batch 1
is latency-bound, and latency-bound work usually overlaps well -- but "usually"
is not a measurement, and 13 GB fitting into 80 GB says nothing about it.

    python scripts/bench_servers.py --ports 8000,8010,8011,8012 --seconds 20

Reports requests per second and latency for 1..N servers used concurrently, so
the marginal value of each extra server is visible.
"""

from __future__ import annotations

import argparse
import statistics
import threading
import time

import json_numpy
import numpy as np
import requests

json_numpy.patch()


def hammer(port: int, stop: threading.Event, latencies: list[float], host: str) -> None:
    session = requests.Session()
    url = f"http://{host}:{port}/act"
    image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    state = np.zeros(8, np.float32)
    payload = {
        "external_cam": image,
        "wrist_cam": image,
        "instruction": "pick up the cup",
        "state": state,
    }
    while not stop.is_set():
        t0 = time.perf_counter()
        try:
            response = session.post(url, json=payload, timeout=120)
            response.raise_for_status()
            response.content  # noqa: B018  -- force the body to be read
        except Exception:  # noqa: BLE001 -- a failed request is a slow request here
            continue
        latencies.append(time.perf_counter() - t0)


def measure(ports: list[int], seconds: float, host: str) -> tuple[float, float]:
    stop = threading.Event()
    latencies: list[list[float]] = [[] for _ in ports]
    threads = [
        threading.Thread(target=hammer, args=(port, stop, latencies[i], host), daemon=True)
        for i, port in enumerate(ports)
    ]
    start = time.perf_counter()
    for thread in threads:
        thread.start()
    time.sleep(seconds)
    stop.set()
    for thread in threads:
        thread.join(timeout=180)
    elapsed = time.perf_counter() - start

    flat = [value for group in latencies for value in group]
    return len(flat) / elapsed, statistics.median(flat) if flat else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ports", required=True, help="servers to use, all on the same card")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--seconds", type=float, default=20.0)
    args = ap.parse_args()

    ports = [int(p) for p in args.ports.split(",")]
    print(f"{'servers':>8} {'req/s':>8} {'median ms':>10} {'scaling':>9}")
    first = None
    for count in range(1, len(ports) + 1):
        rate, latency = measure(ports[:count], args.seconds, args.host)
        first = first or rate
        print(f"{count:>8} {rate:>8.2f} {latency * 1000:>10.0f} {rate / first:>8.2f}x")


if __name__ == "__main__":
    main()
