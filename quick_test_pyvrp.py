# Quick test + FRVCPY evaluation
# Usage:
#   python quick_test_pyvrp.py <instance.vrp> <frvcpy.json> [q_init=auto] [seconds=2] [seed=1]
#
# What it does:
#   - Loads VRPLIB .vrp, solves briefly with PyVRP.
#   - Extracts routes from best solution (clients only), wraps each with depot 0 on both ends.
#   - Passes each route to frvcpy.Solver with Q_INIT (default = JSON["max_q"]).
#   - Prints per-route inserted-station distance ("duration" in frvcpy when speed=1) and total.
#
# Requirements:
#   pip install pyvrp frvcpy
#
# Notes:
#   - This assumes the FRVCPY JSON was built from the *same* EVRP as the .vrp,
#     with coords ordered as [depot=0] + customers (ascending by original id) + stations.
#   - Under that assumption, PyVRP internal indices (0=depot, 1..n=clients) match the JSON indices.
#   - If your depot in EVRP isn't node id 1, or JSON order differs, adjust the mapping in _map_route().

import sys
import json
from pathlib import Path

def _make_ascii_copy_if_needed(p: Path) -> Path:
    txt = p.read_text(encoding="utf-8", errors="ignore")
    # Common non-ASCII replacements
    repl = {
        "\u2013": "-", "\u2014": "-",
        "\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"',
        "\u00a0": " ",
    }
    has_non_ascii = any(ord(ch) > 127 for ch in txt)
    if not has_non_ascii:
        return p
    for k, v in repl.items():
        txt = txt.replace(k, v)
    ascii_txt = txt.encode("ascii", "ignore").decode("ascii")
    outp = p.with_name(p.stem + "_ascii" + p.suffix)
    outp.write_text(ascii_txt, encoding="ascii")
    print(f"[info] Non-ASCII detected. Using ASCII-safe copy: {outp.name}")
    return outp

def _extract_routes(res):
    # Returns list of lists of ints (client indices), without depot.
    routes = []
    try:
        sol = res.best
        for route in sol.routes():
            seq = [int(node) for node in route]  # typically clients only
            routes.append(seq)
    except Exception as e:
        print("[warn] Could not extract routes via res.best.routes():", e)
    return routes

def _map_route(pyvrp_seq, depot_index_json=0):
    # Map PyVRP internal indices to FRVCPY JSON indices.
    # With matching construction, indices are identical: 0 = depot, 1..n = clients.
    # We still enforce depot at both ends.
    return [depot_index_json] + list(pyvrp_seq) + [depot_index_json]

def main():
    if len(sys.argv) < 3:
        print("Usage: python quick_test_pyvrp.py <instance.vrp> <frvcpy.json> [q_init=auto] [seconds=2] [seed=1]")
        sys.exit(1)

    vrp_path = Path(sys.argv[1])
    frv_json_path = Path(sys.argv[2])
    q_init_arg = sys.argv[3] if len(sys.argv) > 3 else "auto"
    secs = float(sys.argv[4]) if len(sys.argv) > 4 else 2.0
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 1

    # Prep VRP path (ASCII-safe if needed)
    vrp_to_read = _make_ascii_copy_if_needed(vrp_path)

    # Load FRVCPY JSON for defaults
    j = json.loads(frv_json_path.read_text(encoding="utf-8"))
    max_q = float(j.get("max_q", 100.0))
    depot_json = int(j.get("depot", 0))
    q_init = max_q if q_init_arg == "auto" else float(q_init_arg)

    try:
        from pyvrp.read import read
        from pyvrp import Model
        from pyvrp.stop import MaxRuntime
    except Exception:
        print("PyVRP not available. Install with: pip install pyvrp")
        raise

    try:
        from frvcpy import solver
    except Exception:
        print("frvcpy not available. Install with: pip install frvcpy")
        raise

    data = read(str(vrp_to_read))
    model = Model.from_data(data)
    res = model.solve(stop=MaxRuntime(secs), seed=seed)

    print("\n=== PyVRP Result ===")
    print("Total cost (PyVRP):", res.cost())

    routes_clients = _extract_routes(res)
    if not routes_clients:
        print("[error] No routes extracted from PyVRP solution.")
        sys.exit(2)

    print("\n=== FRVCPY Evaluation (with station insertion) ===")
    total_duration = 0.0
    for idx, cliseq in enumerate(routes_clients, 1):
        route_json_idx = _map_route(cliseq, depot_index_json=depot_json)
        frv = solver.Solver(str(frv_json_path), route_json_idx, q_init)
        duration, feas_route = frv.solve()  # duration ~ distance when speed=1.0
        total_duration += float(duration)
        print(f"Route {idx:02d}: duration={duration:.6f} | mapped_route={route_json_idx}")

    print(f"\n>>> Sum of inserted-station durations (distance @ speed=1): {total_duration:.6f}")

if __name__ == "__main__":
    main()
