"""E5: parcheador in-place de pcaps flow2pcap — repara ip.len.

Los pcaps generados con `pk[IP].len = ln` (tamaño de frame de las stats
del flujo) declaran 14 bytes de más que el IP real → Suricata descarta
todos los paquetes como truncados. Fix: reescribir el campo IP total
length (offset 16-17 del frame, big-endian) = caplen − 14 (el payload ya
está completo en el frame; el frame ES la longitud real). Checksums IP/TCP
quedan de la longitud vieja — irrelevante, la corrida va con
checksum-checks=no (pcap sintético, documentado).

Uso:
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
    """Reescribe ip.len por paquete. Devuelve (paquetes, bytes_parcheados)."""
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
