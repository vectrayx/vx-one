"""E5 v0.1 (§4.4): enhanced parquet → synthetic pcap for Suricata.

VectrayX F1 C++ does not exist; Phase 1 is Suricata 7 + ET Open, which
processes pcap (not streams). Documented reconstruction, lower level:
- REAL enhanced headers: 5-tuple, start timestamp, TCP flags
flow aggregates, flow stats packet sizes.
- NO payload: ET content rules (payload match) do not fire —
documented Phase 1 lower bound (the paper reports the metrics of
structural cascade: 5-tuple matching, gray zone, E2E latency).
- TCP: SYN → SYN/ACK → ACK + one data packet per direction with the flags
aggregates; close FIN/RST if the flow registers them. UDP: one packet per
address. The actual IATs in the stream are not played (only the order).
- Pcap per day with ordered real timestamps (Meerkat demands order).

Use:
python -m src.e5.flow2pcap --day 2017-07-05 --out pcaps_e5
"""

from __future__ import annotations

import argparse
import math
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
from scapy.all import IP, TCP, UDP, Ether, PcapWriter, Raw

ENHANCED = Path("data/raw/cicids2017/enhanced_flows/CICIDS_Flow.parquet")
# 2 formats documented in S1: 07/03/2017 08:55:58 and 7/5/2017 2:59
_DAYS = {"2017-07-03": "3/7/2017", "2017-07-04": "4/7/2017",
         "2017-07-05": "5/7/2017", "2017-07-06": "6/7/2017",
         "2017-07-07": "7/7/2017"}
_COLS = ["flow_id", "source_ip", "source_port", "destination_ip",
         "destination_port", "protocol", "Timestamp", "Flow Duration",
         "Total Fwd Packets", "Total Backward Packets",
         "Fwd Packet Length Mean", "Bwd Packet Length Mean",
         "SYN Flag Count", "FIN Flag Count", "RST Flag Count",
         "PSH Flag Count", "ACK Flag Count", "URG Flag Count",
         "attack_label"]


def _norm_day(s: str) -> str:
    """'07/03/2017' and '7/3/2017' → '7/3/2017' (day and month without zero)."""
    d, m, y = s.split("/")
    return f"{int(d)}/{int(m)}/{y}"


def build_day_pcap(day: str, out_dir: Path) -> tuple[Path, int, int]:
    """Pcap of ONE day. Returns (path, packets, streams)."""
    tbl = pq.read_table(ENHANCED, columns=_COLS)
    ts = tbl.column("Timestamp").to_numpy(zero_copy_only=False).astype(str)
    want = _DAYS[day]
    mask = np.array([_norm_day(t.split(" ")[0]) == want for t in ts])
    idx = np.flatnonzero(mask)
    sub = tbl.take(idx)
    n = len(idx)
    ts0 = np.array([datetime.strptime(t if t.count(":") == 2 else t + ":00",
                                      "%d/%m/%Y %H:%M:%S")
                    for t in ts[idx]])

    ip_s = sub.column("source_ip").to_pylist()
    ip_d = sub.column("destination_ip").to_pylist()
    port_s = sub.column("source_port").to_pylist()
    port_d = sub.column("destination_port").to_pylist()
    proto = sub.column("protocol").to_pylist()
    dur_s = np.asarray(sub.column("Flow Duration").to_pylist(),
                       dtype=np.float64) / 1e6  # µs → s
    fwd_p = np.asarray(sub.column("Total Fwd Packets").to_pylist(),
                       dtype=np.float64)
    bwd_p = np.asarray(sub.column("Total Backward Packets").to_pylist(),
                       dtype=np.float64)
    fwd_len = np.nan_to_num(np.asarray(
        sub.column("Fwd Packet Length Mean").to_pylist(), dtype=np.float64),
        nan=0.0)
    bwd_len = np.nan_to_num(np.asarray(
        sub.column("Bwd Packet Length Mean").to_pylist(), dtype=np.float64),
        nan=0.0)
    syn = np.asarray(sub.column("SYN Flag Count").to_pylist(), dtype=np.float64)
    fin = np.asarray(sub.column("FIN Flag Count").to_pylist(), dtype=np.float64)
    rst = np.asarray(sub.column("RST Flag Count").to_pylist(), dtype=np.float64)
    psh = np.asarray(sub.column("PSH Flag Count").to_pylist(), dtype=np.float64)
    ack = np.asarray(sub.column("ACK Flag Count").to_pylist(), dtype=np.float64)
    urg = np.asarray(sub.column("URG Flag Count").to_pylist(), dtype=np.float64)
    labels = sub.column("attack_label").to_pylist()

    out = out_dir / f"e5_{day}.pcap"
    w = PcapWriter(str(out), linktype=1, sync=1)  # Ethernet
    n_pk = 0

    def emit(i: int, t: float, sport: int, dport: int, flags: str,
             length: float, forward: bool) -> None:
        nonlocal n_pk
        ln = int(min(max(length, 54), 1454))
        pay = b"\x00" * max(ln - 54, 0)
        tcp = proto[i] == "tcp"
        l4 = (TCP(sport=sport, dport=dport, flags=flags) if tcp
              else UDP(sport=sport, dport=dport))
        pk = Ether() / IP(src=ip_s[i], dst=ip_d[i]) / l4
        if pay:
            pk = pk / Raw(pay)
        # IP.len is calculated by scapy when serializing — set it by hand (the size
        # frame of the stats) lied in 14 bytes and Suricata discarded
        # all packets as truncated (runs 1-3: 0 flows)
        pk.time = max(0.0, t)  # EPOCH float; PcapWriter.write usa pkt.time
        w.write(pk)
        n_pk += 1

    for i in range(n):
        t0 = ts0[i].timestamp()
        tcp = proto[i] == "tcp"
        sp, dp = int(port_s[i]), int(port_d[i])
        if tcp:
            emit(i, t0, sp, dp, "S", 54, True)                      # SYN
            emit(i, t0 + 0.001, dp, sp, "SA", 54, False)            # SYN/ACK
            emit(i, t0 + 0.002, sp, dp, "A", 54, True)              # ACK
            f_end = t0 + min(dur_s[i], 3600.0) / 2
            b_end = t0 + min(dur_s[i], 3600.0)
            if fwd_p[i] > 3:
                fl = "PA" + ("P" if psh[i] > 0 else "") + \
                     ("U" if urg[i] > 0 else "")
                emit(i, f_end, sp, dp, fl, fwd_len[i] + 54, True)
            if bwd_p[i] > 0:
                emit(i, b_end, dp, sp, "PA" if ack[i] > 0 else "A",
                     bwd_len[i] + 54, False)
            if rst[i] > 0:
                emit(i, b_end + 0.001, sp, dp, "R", 54, True)
            elif fin[i] > 0:
                emit(i, b_end + 0.001, sp, dp, "FA", 54, True)
        else:
            emit(i, t0, sp, dp, "", fwd_len[i] + 54, True)
            if bwd_p[i] > 0:
                emit(i, t0 + 0.01, dp, sp, "", bwd_len[i] + 54, False)
    w.close()
    return out, n_pk, n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", required=True, help="2017-07-03 .. 2017-07-07")
    ap.add_argument("--out", default="pcaps_e5")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pcap, n_pk, n_fl = build_day_pcap(args.day, out_dir)
    print(f"[flow2pcap] {pcap}: {n_fl} flujos → {n_pk} paquetes "
          f"({pcap.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
