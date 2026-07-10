#!/usr/bin/env python
"""Download the released PPE weights and verify their sha256.

Standard setup step after cloning:

  python scripts/get_weights.py

Places the asset at weights/ppe-detect-y8n-sfchd.pt (gitignored). Fails hard
if the checksum does not match the recorded value.
"""

from __future__ import annotations

import hashlib
import ssl
import sys
import urllib.request
from pathlib import Path

import certifi


def sha256_file(path: Path) -> str:
    # stdlib-only on purpose: this script must run before/without the package
    # (e.g. in the Docker builder stage, which has no OpenCV system libs)
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


WEIGHTS_URL = (
    "https://github.com/abhipabhi/ppe-detect/releases/download/"
    "weights-v1/ppe-detect-y8n-sfchd.pt"
)
WEIGHTS_SHA256 = "ed20fd001d5dfef61c738928c799201d3323f2c5148e3d0dfa0c521263d28870"
DEST = Path(__file__).resolve().parents[1] / "weights" / "ppe-detect-y8n-sfchd.pt"


def main() -> int:
    if DEST.exists() and sha256_file(DEST) == WEIGHTS_SHA256:
        print(f"weights already present and verified: {DEST}")
        return 0

    DEST.parent.mkdir(parents=True, exist_ok=True)
    tmp = DEST.with_suffix(".pt.part")
    print(f"downloading {WEIGHTS_URL}")
    # certifi's CA bundle: system Pythons on macOS often ship without root certs
    context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(WEIGHTS_URL, context=context) as response, open(tmp, "wb") as out:
        while chunk := response.read(1 << 20):
            out.write(chunk)

    digest = sha256_file(tmp)
    if digest != WEIGHTS_SHA256:
        tmp.unlink(missing_ok=True)
        print(
            f"checksum mismatch!\n  expected {WEIGHTS_SHA256}\n  got      {digest}",
            file=sys.stderr,
        )
        return 1

    tmp.rename(DEST)
    print(f"weights verified and saved to {DEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
