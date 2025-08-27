#!/usr/bin/env python
"""
plot_temps_exact.py ─ Plotta Tbp e Tjr per ogni CSV,
con le stesse chiamate usate da optuna_run.py.

Esempi
------
$ python plot_temps_exact.py                       # directory & glob da config
$ python plot_temps_exact.py --show                # apre finestre interattive
$ python plot_temps_exact.py --dir data_sets/new   # directory custom
$ python plot_temps_exact.py --glob '*_coolant.csv'
"""

# --------------------------------------------------------------------- std lib
import argparse
from pathlib import Path

# --------------------------------------------------------------------- 3rd-party
import matplotlib.pyplot as plt

# --------------------------------------------------------------------- project
from src.config import CSV_DIR, CSV_GLOB, PLOT_PATH, PLOT_DPI
# le funzioni che abbiamo introdotto nel refactor multi-CSV
from src.data_utils import list_csv_files, load_single_csv   # ← stesse di optuna

# --------------------------------------------------------------------- helpers
def plot_dataset(csv_path: str, show: bool = False):
    """Carica e plott­a (t, Tbp, Tjr) exactly like optuna_run.py does."""
    cols = load_single_csv(csv_path)          # t-sorted, colonne 9 & 12
    t, Tbp, Tjr = cols["t"], cols["Tbp"], cols["Tjr"]

    fname_out = Path(csv_path).stem + "_tbp_tjr.png"
    out_path  = Path(PLOT_PATH) / fname_out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 4))
    plt.plot(t, Tbp, label="Tbp (°C)", linewidth=1.2)
    plt.plot(t, Tjr, label="Tjr (°C)", linewidth=1.2)
    plt.title(Path(csv_path).name)
    plt.xlabel("Tempo [s]")
    plt.ylabel("Temperatura [°C]")
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=PLOT_DPI)
    if show:
        plt.show()
    plt.close()
    print(f"✔ Salvato: {out_path}")

# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Plot Tbp & Tjr (exact optuna style)")
    ap.add_argument("--dir",  default=CSV_DIR,  help="Cartella CSV (def: config.CSV_DIR)")
    ap.add_argument("--glob", default=CSV_GLOB, help="Pattern glob (def: config.CSV_GLOB)")
    ap.add_argument("--show", action="store_true", help="Mostra figure a schermo")
    args = ap.parse_args()

    csv_files = list_csv_files() if args.dir == CSV_DIR and args.glob == CSV_GLOB \
                else sorted(Path(args.dir).glob(args.glob))
    if not csv_files:
        raise SystemExit("Nessun CSV trovato – controlla --dir / --glob")

    print(f"Trovati {len(csv_files)} file – inizio plotting…")
    for f in csv_files:
        plot_dataset(str(f), show=args.show)

if __name__ == "__main__":
    main()
