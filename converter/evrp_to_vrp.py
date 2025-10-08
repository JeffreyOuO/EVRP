#!/usr/bin/env python3
# EVRP (.evrp) -> VRPLIB CVRP (.vrp) converter for PyVRP
# - 移除充電站（無論站點 ID 是否 <= DIMENSION，只要在 STATIONS_* 就排除）
# - 只輸出 depot + 客戶（NODE_COORD_SECTION / DEMAND_SECTION / DEPOT_SECTION）
# - 表頭用 "KEY : VALUE"
# - NODE_COORD_SECTION 用 5 位小數
# - 需求缺失時：depot=0、其他預設 0；若 DEMAND_SECTION 缺失，會在 NODE 區塊後面嘗試蒐羅 "id demand"
# - DIMENSION 自動以實際輸出行數寫入（避免不一致）
# - 預設 ASCII-safe 輸出；用 --no-ascii 可停用
#
# 用法：
#   python evrp_to_vrp.py INPUT.evrp OUTPUT.vrp [--no-ascii]

import sys
import re

# ------------- ASCII 安全輸出 -------------
ASCII_MAP = {
    "–": "-",  # en dash
    "—": "-",  # em dash
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
    "\u00A0": " ",  # nbsp
}
def to_ascii(s: str) -> str:
    for k, v in ASCII_MAP.items():
        s = s.replace(k, v)
    return s.encode("ascii", "ignore").decode("ascii")

def read_text(path):
    return open(path, "r", encoding="utf-8", errors="ignore").read()

def write_text(path, text, ascii_only=True):
    out = to_ascii(text) if ascii_only else text
    enc = "ascii" if ascii_only else "utf-8"
    with open(path, "w", encoding=enc) as f:
        f.write(out)

# ------------- 表頭讀取 -------------
def get_header(lines, key):
    # 允許 "KEY: X" 也允許 "KEY : X"；大小寫無關；前後空白忽略
    pat = re.compile(rf"^{key}\s*:\s*(.*)$", re.IGNORECASE)
    for ln in lines:
        m = pat.match(ln.strip())
        if m:
            return m.group(1).strip()
    return None

def as_int(x, default=None):
    try:
        # 允許 "6000 " 或 "6000.0"
        return int(float(x))
    except:
        return default

def as_float(x, default=None):
    try:
        return float(x)
    except:
        return default

def convert(src_path, dst_path, ascii_only=True):
    raw = read_text(src_path)
    lines = [ln.strip() for ln in raw.splitlines()]

    name       = get_header(lines, "NAME") or get_header(lines, "Name") or "EVRP_Converted"
    comment    = get_header(lines, "COMMENT") or ""
    dim_s      = get_header(lines, "DIMENSION")
    cap_s      = get_header(lines, "CAPACITY")
    edge_fmt   = get_header(lines, "EDGE_WEIGHT_TYPE") or get_header(lines, "EDGE_WEIGHT_FORMAT") or "EUC_2D"

    DIM = as_int(dim_s, 0)
    CAP = as_int(cap_s, None)  # VRPLIB 需要整數

    # ------------- 區塊解析 -------------
    sec = None
    node_coords = {}     # id -> (x,y)
    demands = {}         # id -> demand
    station_ids = set()  # 需排除
    depot_id = None

    def is_section_header(ln):
        up = ln.upper()
        return up in {
            "NODE_COORD_SECTION", "DEMAND_SECTION",
            "STATIONS_COORD_SECTION", "STATIONS_SECTION",
            "DEPOT_SECTION", "EOF",
        }

    i = 0
    while i < len(lines):
        ln = lines[i].strip()
        up = ln.upper()
        if up == "NODE_COORD_SECTION":
            sec = "NODE"; i += 1; continue
        elif up == "DEMAND_SECTION":
            sec = "DEMAND"; i += 1; continue
        elif up in ("STATIONS_COORD_SECTION", "STATIONS_SECTION"):
            sec = "STATIONS"; i += 1; continue
        elif up == "DEPOT_SECTION":
            sec = "DEPOT"; i += 1; continue
        elif up == "EOF":
            break

        if sec == "NODE":
            # id x y
            toks = ln.split()
            if len(toks) >= 3 and toks[0].replace('.','',1).isdigit():
                try:
                    nid = int(float(toks[0]))
                    x = float(toks[1]); y = float(toks[2])
                    node_coords[nid] = (x, y)
                except:
                    pass
            i += 1; continue

        if sec == "DEMAND":
            toks = ln.split()
            if len(toks) >= 2 and toks[0].lstrip("-").isdigit() and toks[1].lstrip("-").isdigit():
                nid = int(toks[0]); dem = as_int(toks[1], 0)
                demands[nid] = dem
            i += 1; continue

        if sec == "STATIONS":
            toks = ln.split()
            if toks and toks[0].lstrip("-").isdigit():
                station_ids.add(int(toks[0]))
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

    # 若沒有 DEMAND_SECTION，但在 NODE 區塊後面有 "id demand" 也嘗試蒐羅
    if not demands:
        scavenging = False
        for ln in lines:
            if ln.upper() == "NODE_COORD_SECTION":
                scavenging = True; continue
            if is_section_header(ln):
                scavenging = False
            if scavenging:
                toks = ln.split()
                if len(toks) == 2 and all(t.lstrip("-").isdigit() for t in toks):
                    nid = int(toks[0]); dem = as_int(toks[1], 0)
                    demands[nid] = dem

    # depot 預設 1
    if depot_id is None:
        depot_id = 1

    # ------------- 排除 CS，建立輸出節點集合 -------------
    # 先取「原始座標中」的 id；再扣掉站點；最後只保留有座標者
    # 注意：有些 EVRP 站點 id 會在 1..DIM 中；這裡一律排除 station_ids
    base_ids = sorted(node_coords.keys())
    keep_ids = [nid for nid in base_ids if nid not in station_ids]

    # 驗證 depot 在不在 keep_ids
    if depot_id not in keep_ids and depot_id in node_coords:
        # 如果 depot 被錯誤列為站點（極少見），強制保留它
        keep_ids = [depot_id] + [nid for nid in keep_ids if nid != depot_id]

    # ------------- 寫 VRPLIB -------------
    # DIMENSION 以實際輸出數量為準（避免行數不符）
    out_DIM = len(keep_ids)

    header = []
    header.append(f"NAME : {name}")
    header.append(f"COMMENT : Converted from EVRP - stations removed")
    header.append(f"TYPE : CVRP")
    header.append(f"DIMENSION : {out_DIM}")
    header.append(f"EDGE_WEIGHT_TYPE : {edge_fmt}")
    header.append(f"CAPACITY : {CAP if CAP is not None else 0}")

    body = []
    body.append("NODE_COORD_SECTION")
    for nid in keep_ids:
        x, y = node_coords[nid]
        body.append(f"{nid} {x:.5f} {y:.5f}")

    body.append("DEMAND_SECTION")
    for nid in keep_ids:
        dem = 0 if nid == depot_id else demands.get(nid, 0)
        body.append(f"{nid} {dem}")

    body.append("DEPOT_SECTION")
    body.append(str(depot_id))
    body.append("-1")
    body.append("EOF")

    write_text(dst_path, "\n".join(header + body), ascii_only=ascii_only)

def main():
    if len(sys.argv) < 3:
        print("Usage: python evrp_to_vrp.py INPUT.evrp OUTPUT.vrp [--no-ascii]")
        sys.exit(1)
    src, dst = sys.argv[1], sys.argv[2]
    ascii_only = True
    if len(sys.argv) >= 4 and sys.argv[3] == "--no-ascii":
        ascii_only = False
    convert(src, dst, ascii_only=ascii_only)
    print("Wrote", dst, "(ASCII-only:", ascii_only, ")")

if __name__ == "__main__":
    main()
