"""E5: in-place pcaps patcher flow2pcap — fix ip.len.

The pcaps generated with `pk[IP].len = ln` (stats frame size
of the flow) declare 14 bytes more than the real IP → Suricata discards
all packets as truncated. Fix: rewrite total IP field
length (frame offset 16-17, big-endian) = caplen − 14 (the payload is already
is complete in the frame; the frame IS the actual length). IP/TCP Checksums
remain of the old length — irrelevant, the run goes with
checksum-checks=no (synthetic pcap, documented).

Use:
.venv/bin/python -m src.e5.patch_iplen pcaps_e5/e5_2017-07-06.pcap [...]
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

ETH = 14
HDR = 24   # pcap global header
REC = 16   # record header


def patch(path: str) -> tuple[int, int]:
    """Rewrites ip.len per packet. Returns (packets, patched_bytes)."""
    buf = bytearray(Path(path).read_bytes())
    off = HDR
    n = 0
    changed = 0
    while off + REC <= len(buf):
        incl = struct.unpack_from("<I", buf, off + 8)[0]
        frame = buf[off + REC:off + REC + incl]
        if len(frame) > ETH + 20 and frame[12:14] == b"\x08\x00":
            ip_len = incl - ETH
            cur = struct.unpack_from(">H", frame, 16)[0]
            if cur != ip_len:
                struct.pack_into(">H", frame, 16, ip_len)
                buf[off + REC:off + REC + incl] = frame
                changed += 1
        off += REC + incl
        n += 1
    Path(path).write_bytes(buf)
    return n, changed


def main() -> None:
    for p in sys.argv[1:]:
        n, c = patch(p)
        print(f"[patch] {Path(p).name}: {n:,} paquetes, {c:,} ip.len corregidos")


if __name__ == "__main__":
    main()
