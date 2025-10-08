#!/usr/bin/env python3
# EVRP (.evrp) -> frvcpy JSON converter
# - 讀取 EVRP：NAME / DIMENSION / CAPACITY / ENERGY_CAPACITY / ENERGY_CONSUMPTION / NODE_COORD_SECTION /
#              STATIONS_COORD_SECTION / DEPOT_SECTION
# - 產出 frvcpy JSON 欄位：coords, depot, customers, css, time_matrix, energy_matrix,
#                        max_q, consumption_rate, speed, breakpoints_by_type, euclidean, decimals
# - 距離採歐式距離；time_matrix = distance（速度=1），energy_matrix = distance * consumption_rate
#
# 用法：
#   python evrp_to_frvcpy.py INPUT.evrp OUTPUT.json
#
# 注意：
# - coords 的索引順序為：depot(索引0) → 客戶(按ID升冪) → 有座標的充電站
# - css 的 node_id、customers 皆使用輸出 JSON 的索引（非原始 EVRP ID）
# - 若 STATIONS 沒提供座標，該站不納入（會印出警告）
# - ENERGY_CONSUMPTION、ENERGY_CAPACITY 缺失時，預設 1.0 與 100.0（可自行改）

import sys
import math
import json
import re
from pathlib import Path

def read_lines(path):
    return [ln.strip() for ln in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()]

def get_header(lines, key, default=None):
    pat = re.compile(rf"^{key}\s*:\s*(.*)$", re.IGNORECASE)
    for ln in lines:
        m = pat.match(ln)
        if m:
            return m.group(1).strip()
    return default

def parse_evrp(path):
    lines = read_lines(path)

    name = get_header(lines, "NAME") or get_header(lines, "Name") or Path(path).stem
    dim  = get_header(lines, "DIMENSION")
    cap  = get_header(lines, "CAPACITY")
    ecap = get_header(lines, "ENERGY_CAPACITY")
    erate= get_header(lines, "ENERGY_CONSUMPTION")
    edge = get_header(lines, "EDGE_WEIGHT_FORMAT") or get_header(lines, "EDGE_WEIGHT_TYPE") or "EUC_2D"

    def as_int(s, d=None):
        try: return int(s)
        except: return d
    def as_float(s, d=None):
        try: return float(s)
        except: return d

    DIM   = as_int(dim, 0)
    CAP   = as_float(cap, 0.0)               #（PyVRP資料；這裡僅保留參考，不輸出到 frvcpy）
    ECAP  = as_float(ecap, 100.0)
    ERATE = as_float(erate, 1.0)

    coords_raw = {}          # id -> (x,y)
    station_ids = []         # station ids listed in STATIONS_COORD_SECTION (may be id-only)
    depot_id = 1

    sec = None
    i = 0
    while i < len(lines):
        ln = lines[i].strip()
        up = ln.upper()

        if up == "NODE_COORD_SECTION":
            sec = "NODE"; i += 1; continue
        if up in ("STATIONS_COORD_SECTION", "STATIONS_SECTION"):
            sec = "STATIONS"; i += 1; continue
        if up == "DEPOT_SECTION":
            sec = "DEPOT"; i += 1; continue
        if up == "EOF":
            break

        if sec == "NODE":
            # id x y
            toks = ln.split()
            if len(toks) >= 3 and toks[0].replace('.','',1).isdigit():
                try:
                    nid = int(float(toks[0]))
                    x = float(toks[1]); y = float(toks[2])
                    coords_raw[nid] = (x, y)
                except:
                    pass
            i += 1; continue

        if sec == "STATIONS":
            # usually a single integer per line: the station node id
            toks = ln.split()
            if toks and toks[0].lstrip("-").isdigit():
                station_ids.append(int(toks[0]))
            i += 1; continue

        if sec == "DEPOT":
            if ln.lstrip("-").isdigit():
                v = int(ln)
                if v == -1:
                    sec = None
                else:
                    depot_id = v
            i += 1; continue

        i += 1

    # 建立 coords（frvcpy：0-based）
    # 排序：depot -> 所有客戶(id != depot, 且 id <= DIM 或在 coords_raw 裡) -> 有座標的站
    if DIM is None or DIM == 0:
        # 若沒給 DIM，回退用 coords_raw 的最大 id（一般 EVRP 都會有）
        DIM = max(coords_raw) if coords_raw else 0

    customer_ids = [nid for nid in sorted(coords_raw) if nid != depot_id and nid <= DIM]
    stations_with_coords = [sid for sid in station_ids if sid in coords_raw]

    order = [depot_id] + customer_ids + stations_with_coords
    idx_of = {nid: i for i, nid in enumerate(order)}
    coords = [[coords_raw[nid][0], coords_raw[nid][1]] for nid in order]

    depot_idx = 0
    customers_idx = list(range(1, 1 + len(customer_ids)))
    css = [{"node_id": idx_of[sid], "cs_type": "fast"} for sid in stations_with_coords]

    # 矩陣
    def euclid(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    n = len(coords)
    time_matrix = [[0.0] * n for _ in range(n)]
    energy_matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            d = euclid(coords[i], coords[j]) if i != j else 0.0
            time_matrix[i][j] = d                 # speed = 1
            energy_matrix[i][j] = d * ERATE       # consumption = dist * rate

    # 簡單線性充電曲線（可改為更真實的分段）
    breakpoints_by_type = [{
        "cs_type": "fast",
        "time":   [0.0, float(ECAP)],
        "charge": [0.0, float(ECAP)],
    }]

    # 建構 frvcpy JSON
    obj = {
        "name": name,
        "euclidean": True,
        "decimals": 6,
        "coords": coords,            # [[x,y], ...]  0-based
        "depot": depot_idx,          # 0
        "customers": customers_idx,  # [1..#customers]
        "css": css,                  # charging stations (by JSON index)
        "max_q": float(ECAP),        # energy capacity
        "consumption_rate": float(ERATE),
        "speed": 1.0,
        "breakpoints_by_type": breakpoints_by_type,
        "time_matrix": time_matrix,
        "energy_matrix": energy_matrix,
    }

    # 一些提示
    notes = []
    if edge.upper() not in ("EUC_2D", "EUC_2D_CEIL"):
        notes.append(f"Edge format '{edge}' not EUC_2D; converter assumed Euclidean distances.")
    missing_station_coords = [sid for sid in station_ids if sid not in coords_raw]
    if missing_station_coords:
        notes.append(f"Skipped {len(missing_station_coords)} stations without coordinates: {missing_station_coords[:10]}...")

    return obj, {
        "name": name,
        "dimension": DIM,
        "depot_id": depot_id,
        "customers": len(customers_idx),
        "stations_included": len(css),
        "edge_format": edge,
        "notes": notes,
    }

def main():
    if len(sys.argv) != 3:
        print("Usage: python evrp_to_frvcpy.py INPUT.evrp OUTPUT.json")
        sys.exit(1)

    src, dst = sys.argv[1], sys.argv[2]
    obj, meta = parse_evrp(src)
    Path(dst).write_text(json.dumps(obj, indent=2), encoding="utf-8")

    print(f"Wrote: {dst}")
    print(f"Summary: name={meta['name']}, DIM={meta['dimension']}, depot_id={meta['depot_id']}, "
          f"customers={meta['customers']}, stations_included={meta['stations_included']}, "
          f"edge={meta['edge_format']}")
    if meta["notes"]:
        print("Notes:")
        for s in meta["notes"]:
            print(" -", s)

if __name__ == "__main__":
    main()
