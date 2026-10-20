"""App Modal del paper VectrayX-One: entrenamiento GPU (10 corridas
principales + folds LOAO), dump de logits y baselines CPU pesados.

Volúmenes:
- paper-data   → /repo/data   (parquets materializados F1, misma estructura
  que local: processed/unsw_nb15/, processed/cicids2017/, processed/ember2024/)
- paper-results → /vol        (artifacts/, outputs/, baselines/)

Solo se monta src/ (el código); los datos viajan por volumen.

Uso:
    modal run modal/app.py::probe
    modal run modal/app.py::train_vx --source unsw --scale small
    modal run modal/app.py::train_vx --source cic --scheme strat --scale base
    modal run modal/app.py::train_vx --source ember --scale small
    modal run modal/app.py::loao --source unsw
    modal run modal/app.py::dump --artifact /repo/data/results_paper/artifacts/unsw_small_oficial.pt
    modal run modal/app.py::baseline --source cic --scheme strat

Los logits (outputs/) y los baselines bajan local con `modal volume get`
para computar E1-E3 en la máquina local.
"""

from __future__ import annotations

import pathlib

import modal

_SRC = pathlib.Path(__file__).resolve().parent.parent / "src"

# Versiones fijadas = entorno local (reproducibilidad del paper).
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install(
             "torch==2.14.1",
             "pyarrow==26.0.0",
             "pandas==3.0.6",
             "numpy==2.5.4",
             "scikit-learn==1.9.1",
             "xgboost==3.4.1",
         )
         .add_local_dir(str(_SRC), "/repo/src", copy=True))

app = modal.App("vectrayx-one", image=image)
data_vol = modal.Volume.from_name("paper-data", create_if_missing=True)
res_vol = modal.Volume.from_name("paper-results", create_if_missing=True)

VOLS = {"/repo/data": data_vol, "/vol": res_vol}
RESULTS = "/vol"          # artifacts/ outputs/ baselines/
REPO = "/repo"
RESULTS_DATA_DIR = "/repo/data/results_paper"  # accesible también por /vol


def _boot() -> None:
    """Rutina común: repo en sys.path + cwd (los paths del repo son relativos
    a la raíz) + directorios de resultados en el volumen."""
    import os
    import sys
    from pathlib import Path
    sys.path.insert(0, REPO)
    os.chdir(REPO)
    for d in ("artifacts", "outputs", "baselines", "ckpts"):
        Path(f"{RESULTS}/{d}").mkdir(parents=True, exist_ok=True)


@app.function(gpu="L4", cpu=8, timeout=10800, volumes=VOLS)
def train_vx(source: str, scale: str, scheme: str,
             exclude_family: str | None = None,
             ckpt_every_steps: int = 2000, resume: bool = True,
             ablation: str | None = None, seed: int = 42) -> dict:
    _boot()
    import torch
    from src.models import build_run, train_run

    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    run = build_run(source, scale, scheme, exclude_family=exclude_family)
    fam = f"_loao_{exclude_family}" if exclude_family else ""
    abl = f"_abl_{ablation}" if ablation else ""
    rid = f"{source}_{scale}_{scheme}{fam}{abl}"
    if seed != 42:
        rid = f"{rid}_s{seed}"     # corridas de varianza (3 seeds) no pisan la oficial
    out = f"{RESULTS}/artifacts/{rid}.pt"
    res = train_run(run, out, device=dev,
                    ckpt_path=f"{RESULTS}/ckpts/{rid}.ckpt.pt",
                    ckpt_every_steps=ckpt_every_steps, resume=resume,
                    ablation=ablation, seed=seed)
    print(f"[modal] {rid}: M_val={res['m_val']:.4f} T={res['temperatures']}")
    # dump de logits val/test en el mismo job (E1-E3 se computan local)
    from src.eval.dump_outputs import dump_run
    dump_run(out, f"{RESULTS}/outputs/{rid}.npz", device=dev)
    return res


@app.function(gpu="L4", cpu=8, timeout=7200, volumes=VOLS)
def dump(artifact: str) -> str:
    _boot()
    from src.eval.dump_outputs import dump_run

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    name = artifact.rsplit("/", 1)[-1].removesuffix(".pt")
    out = f"{RESULTS}/outputs/{name}.npz"
    dump_run(artifact, out, device=dev)
    return out


@app.function(gpu="L4", cpu=8, timeout=7200, volumes=VOLS)
def dump_challenge(artifact: str) -> dict:
    """Dump de logits sobre el challenge set EMBER con el checkpoint dado
    (p.ej. /vol/artifacts/ember_small_temporal.pt). Estrés, fuera de toda
    promesa de coverage (advisory Q2)."""
    _boot()
    import torch
    from src.eval.dump_outputs import dump_ember_challenge

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    name = artifact.rsplit("/", 1)[-1].removesuffix(".pt")
    out = f"{RESULTS}/outputs/{name}_challenge.npz"
    res = dump_ember_challenge(artifact, out, device=dev)
    return res


@app.function(cpu=32, memory=131072, timeout=43200, volumes=VOLS)
def baseline(source: str, scheme: str, models: str = "xgboost,rf,mlp",
             exclude_family: str | None = None) -> dict:
    _boot()
    from src.baselines.run_tabular import run_baselines

    run_baselines(source, scheme, f"{RESULTS}/baselines",
                  exclude_family=exclude_family,
                  models=tuple(models.split(",")))
    return {"source": source, "scheme": scheme, "exclude": exclude_family,
            "models": models}


@app.function(cpu=4, timeout=1800, volumes=VOLS)
def probe() -> dict:
    """Sanity: deps cargan, el volumen tiene los parquets esperados."""
    import os
    _boot()
    import torch
    import pyarrow
    import sklearn
    import xgboost
    from src.state.state import SOURCE_REGISTRY

    base = "data/processed"
    paths = {
        "unsw": os.path.exists(f"{base}/unsw_nb15/unsw_nb15.parquet"),
        "cic": os.path.exists(f"{base}/cicids2017/cicids2017.parquet"),
        "ember_train": os.path.exists(f"{base}/ember2024/part_00_train.parquet"),
        "ember_val": os.path.exists(f"{base}/ember2024/part_46_val.parquet"),
        "ember_test": os.path.exists(f"{base}/ember2024/part_52_test.parquet"),
    }
    return {"torch": torch.__version__, "cuda": torch.cuda.is_available(),
            "pyarrow": pyarrow.__version__, "sklearn": sklearn.__version__,
            "xgboost": xgboost.__version__, "paths": paths,
            "n_registry": len(SOURCE_REGISTRY)}


@app.local_entrypoint()
def probe_main():
    print(probe.remote())


@app.local_entrypoint()
def seeds_main(seeds: str = "42,43,44",
               runs: str = "unsw_small_oficial,cic_small_strat,"
                           "cic_small_time,ember_small_temporal"):
    """Varianza multi-seed de las corridas principales de las tablas.
    Lanza |seeds|×|runs| jobs L4 en paralelo; rid lleva sufijo _s<seed>
    (salvo la 42, que es la oficial ya publicada y se re-valida)."""
    s_list = [int(x) for x in seeds.split(",")]
    r_list = [tuple(x.split("_")) for x in runs.split(",") if x]
    jobs = []
    for s in s_list:
        for (src, sc, sm) in r_list:
            seed = 42 if s == 42 else s
            jobs.append((src, sc, sm, seed))
    print(f"[seeds_main] {len(jobs)} jobs: {jobs}")
    handles = [train_vx.spawn(source=s, scale=sc, scheme=sm, seed=sd)
               for (s, sc, sm, sd) in jobs]
    for ((s, sc, sm, sd), h) in zip(jobs, handles):
        try:
            r = h.get()
            tag = "" if sd == 42 else f"_s{sd}"
            print(f"[seeds] {s}_{sc}_{sm}{tag}: M_val={r['m_val']:.4f} "
                  f"wall={r['wall_time_s']/60:.1f} min")
        except Exception as e:
            print(f"[seeds] {s}_{sc}_{sm}_s{sd}: FALLÓ {e!r}")


@app.local_entrypoint()
def mains(only: str = ""):
    """Lanza las corridas principales restantes EN PARALELO (§6.3).
    unsw_small_oficial ya está hecho; esto lanza las otras 9."""
    jobs = [
        ("unsw", "nano", "oficial"), ("unsw", "base", "oficial"),
        ("cic", "nano", "strat"), ("cic", "small", "strat"),
        ("cic", "base", "strat"), ("cic", "small", "time"),
        ("ember", "nano", "temporal"), ("ember", "small", "temporal"),
        ("ember", "base", "temporal"),
    ]
    if only:
        jobs = [j for j in jobs if f"{j[0]}_{j[1]}_{j[2]}" == only]
    handles = [train_vx.spawn(source=s, scale=sc, scheme=sm)
               for s, sc, sm in jobs]
    for (s, sc, sm), h in zip(jobs, handles):
        try:
            r = h.get()
            print(f"[mains] {s}_{sc}_{sm}: M_val={r['m_val']:.4f} "
                  f"wall={r['wall_time_s'] / 60:.1f} min")
        except Exception as e:
            print(f"[mains] {s}_{sc}_{sm}: FALLÓ {e!r}")


@app.local_entrypoint()
def loao(source: str = "unsw", scale: str = "small"):
    """Lanza todos los folds LOAO del dataset en paralelo (§6.7)."""
    fams = ([f for f in ("dos", "recon", "exploitation", "shellcode",
                         "backdoor", "analysis", "fuzzing", "generic",
                         "worms")] if source == "unsw"
            else [f for f in ("dos_ddos", "recon", "bruteforce", "webattack",
                              "botnet", "infiltration", "exploitation")])
    scheme = "oficial" if source == "unsw" else "strat"
    handles = [train_vx.spawn(source=source, scale=scale, scheme=scheme,
                              exclude_family=f) for f in fams]
    for f, h in zip(fams, handles):
        try:
            r = h.get()
            print(f"[loao] {source}/{f}: M_val={r['m_val']:.4f}")
        except Exception as e:  # un fold que falla no tumba los demás
            print(f"[loao] {source}/{f}: FALLÓ {e!r}")


@app.local_entrypoint()
def ablation(kind: str, source: str):
    """E6 §7: UN ablation a Small (misma receta que la referencia).
    kind ∈ a1 (solo UNSW), a2 (3 single-head + merge), a3, a4, a5, a6.
    Lanzar una invocación por combinación para que un fallo no cancele a
    los demás."""
    scheme = {"unsw": "oficial", "cic": "strat", "ember": "temporal"}[source]
    # .remote() bloquea y devuelve el resultado (en Modal 2.x el Function
    # no es invocable directo desde el entrypoint local)
    if kind == "a2":
        for h in ("a2_noul", "a2_choice", "a2_score"):
            train_vx.remote(source=source, scale="small", scheme=scheme,
                            ablation=h)
        return  # el merge (merge_a2_outputs) corre LOCAL sobre los dumps
    train_vx.remote(source=source, scale="small", scheme=scheme, ablation=kind)


# E5 (§4.4): imagen propia con Suricata. Debian 12 (bookworm) estable trae
# 6.0.x; el 7.x sale de backports. El fallback evita que la build muera si
# backports no lo tiene, y la versión real queda registrada en el output.
image_suricata = image.run_commands(
    "apt-get update",
    'echo "deb http://deb.debian.org/debian bookworm-backports main" '
    "> /etc/apt/sources.list.d/backports.list && apt-get update && "
    "(apt-get install -y -t bookworm-backports suricata "
    "|| apt-get install -y suricata)",
    "suricata --build-info | head -5 || true",
)


@app.function(cpu=8, timeout=7200, volumes=VOLS, image=image_suricata)
def suricata_e5(day: str) -> dict:
    """E5 v0.1: Suricata + ET Open sobre /vol/e5/pcaps/e5_{day}.pcap →
    /vol/e5/eve_{day}.json.gz. Cota inferior documentada: la reconstrucción
    flow2pcap no lleva payload, así que SOLO disparan reglas de cabecera
    (puertos, flags, umbrales/threshold) — las de contenido no pueden."""
    import gzip
    import subprocess
    import tarfile
    import urllib.request
    from pathlib import Path

    pcap = Path(f"/vol/e5/pcaps/e5_{day}.pcap")
    if not pcap.exists():
        raise FileNotFoundError(pcap)
    ver = (subprocess.run(["suricata", "--build-info"], capture_output=True,
                          text=True).stdout.splitlines()[0].strip())

    # ET Open: reglas frescas por corrida (el open ruleset se actualiza a
    # diario); 3 URLs conocidas por si el path genérico cambia
    rd = Path("/tmp/e5rules")
    rd.mkdir(exist_ok=True)
    urls = ("https://rules.emergingthreats.net/open/suricata/emerging.rules.tar.gz",
            "https://rules.emergingthreats.net/open/suricata-7.0.3/emerging.rules.tar.gz",
            "https://rules.emergingthreats.net/open/suricata-6.0.10/emerging.rules.tar.gz")
    tgz = None
    for u in urls:
        try:
            tgz = urllib.request.urlretrieve(u, rd / "et.tar.gz")[0]
            print(f"[e5] reglas: {u}")
            break
        except Exception as e:
            print(f"[e5] {u}: {e!r}")
    if tgz is None:
        raise RuntimeError("ET Open: ninguna URL de reglas respondió")

    # concatenar los *.rules del tarball en un suricata.rules (-S carga UNO)
    rules_all = rd / "suricata.rules"
    n_rules = 0
    with tarfile.open(tgz, "r:gz") as tf, rules_all.open("wb") as out:
        for m in tf.getmembers():
            if not m.name.endswith(".rules"):
                continue
            f = tf.extractfile(m)
            data = f.read() if f else b""
            out.write(data)
            n_rules += sum(1 for ln in data.splitlines()
                           if ln.startswith(b"alert"))
    # 8 reglas estructurales propias (cabecera+threshold, sin payload): la
    # reconstrucción flow2pcap no lleva payload, así que las ET de contenido
    # no pueden disparar (cota inferior documentada); estas reglas de
    # flood/scan sí son legítimas para Fase 1 y dan señal estructural.
    with rules_all.open("ab") as out:
        struct_rules = b"""
alert tcp any any -> $HOME_NET any (msg:"VXSTRUCT tcp syn flood"; flags:S; flow:stateless; threshold:type both, track by_src, count 100, seconds 10; sid:9700001; rev:1;)
alert tcp any any -> $HOME_NET any (msg:"VXSTRUCT tcp syn flood high"; flags:S; flow:stateless; threshold:type both, track by_src, count 1000, seconds 10; sid:9700002; rev:1;)
alert tcp any any -> $HOME_NET any (msg:"VXSTRUCT portscan many ports"; flags:S; flow:stateless; threshold:type both, track by_src, count 50, seconds 60; sid:9700003; rev:1;)
alert tcp any any -> $HOME_NET 445 (msg:"VXSTRUCT smb probe"; flags:S; flow:stateless; threshold:type both, track by_src, count 20, seconds 60; sid:9700004; rev:1;)
alert tcp any any -> $HOME_NET 22 (msg:"VXSTRUCT ssh probe"; flags:S; flow:stateless; threshold:type both, track by_src, count 20, seconds 60; sid:9700005; rev:1;)
alert tcp any any -> $HOME_NET 80 (msg:"VXSTRUCT http probe"; flags:S; flow:stateless; threshold:type both, track by_src, count 100, seconds 60; sid:9700006; rev:1;)
alert udp any any -> $HOME_NET any (msg:"VXSTRUCT udp flood"; flow:stateless; threshold:type both, track by_src, count 200, seconds 10; sid:9700007; rev:1;)
alert icmp any any -> $HOME_NET any (msg:"VXSTRUCT icmp flood"; flow:stateless; threshold:type both, track by_src, count 100, seconds 10; sid:9700008; rev:1;)
"""
        out.write(struct_rules)
        n_rules += 8
    print(f"[e5] {ver} | {n_rules} reglas ET Open")

    rdir = Path("/tmp/e5run")
    rdir.mkdir(exist_ok=True)
    cmd = ["suricata", "-c", "/etc/suricata/suricata.yaml",
           "-S", str(rules_all), "-r", str(pcap), "-l", str(rdir),
           # pcap sintético: scapy deja chksum TCP/UDP=0 → sin esto Suricata
           # descarta todos los flujos (corrida 1: 0 alertas/0 flujos)
           "--set", "checksum-checks=no",
           # el pcap del jueves trae ~458K flujos: default flow.memcap=128mb
           # se quedaría corto y descartaría flow-records
           "--set", "flow.memcap=1gb", "--set", "flow.prealloc=600000",
           "--set", "flow.hash-size=262144"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=7000)
    print(r.stdout[-1200:])
    print(r.stderr[-1200:])
    eve = rdir / "eve.json"
    if not eve.exists():
        raise RuntimeError(f"suricata sin eve.json (rc={r.returncode})")

    out = Path(f"/vol/e5/eve_{day}.json.gz")
    out.parent.mkdir(parents=True, exist_ok=True)
    n_alerts = n_flow = 0
    with eve.open("rb") as f, gzip.open(out, "wb") as gz:
        for line in f:  # recomprimir + contar en un solo pase
            gz.write(line)
            j = line.find(b'"event_type"')
            if j == -1:
                continue
            k0 = line.find(b'"', j + 12) + 1
            et = line[k0:line.find(b'"', k0)]
            if et == b"alert":
                n_alerts += 1
            elif et == b"flow":
                n_flow += 1
    return {"day": day, "version": ver, "rules": n_rules,
            "alerts": n_alerts, "flows": n_flow,
            "eve_s": round(out.stat().st_size / 1e6, 1)}


@app.local_entrypoint()
def suricata_main(day: str):
    print(suricata_e5.remote(day))
