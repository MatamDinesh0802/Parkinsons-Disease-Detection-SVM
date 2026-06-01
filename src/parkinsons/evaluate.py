"""Re-run evaluation against the held-out split and print the metrics table.

Useful for CI-style checks even without GitHub Actions: `python -m src.parkinsons.evaluate`.
"""
from __future__ import annotations

import json

from .config import METRICS_PATH


def main() -> None:
    if not METRICS_PATH.exists():
        raise SystemExit("No metrics.json found — run `python -m src.parkinsons.train` first.")
    payload = json.loads(METRICS_PATH.read_text())
    print(f"Best model: {payload['best_model']}")
    print(f"Train size: {payload['n_train']}  Test size: {payload['n_test']}\n")
    header = f"{'model':22s}  {'acc':>6s} {'prec':>6s} {'rec':>6s} {'f1':>6s} {'auc':>6s}"
    print(header)
    print("-" * len(header))
    for name, m in payload["models"].items():
        print(f"{name:22s}  {m['accuracy']:.4f} {m['precision']:.4f} "
              f"{m['recall']:.4f} {m['f1']:.4f} {m['roc_auc']:.4f}")


if __name__ == "__main__":
    main()
