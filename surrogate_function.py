# evrp_min_detour_energy_matrix.py
import math
from typing import Dict, Tuple, Set
import matplotlib.pyplot as plt


def parse_evrp_nodes_and_stations(file_path: str):
    """
    讀取 EVRP 檔案：
      - 解析 ENERGY_CAPACITY 與 ENERGY_CONSUMPTION
      - NODE_COORD_SECTION: id x y (所有節點)
      - STATIONS_COORD_SECTION: id (充電站編號)
    回傳:
      all_nodes: {id: (x, y)}
      other_nodes: {id: (x, y)}（非充電站）
      station_nodes: {id: (x, y)}（充電站）
      energy_capacity: float
      energy_consumption: float
    """
    section = None
    all_nodes: Dict[int, Tuple[float, float]] = {}
    station_ids: Set[int] = set()
    energy_capacity = None
    energy_consumption = None

    with open(file_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            # --- 基本參數 ---
            if line.startswith("ENERGY_CAPACITY"):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        energy_capacity = float(parts[1])
                    except ValueError:
                        pass
                continue

            if line.startswith("ENERGY_CONSUMPTION"):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        energy_consumption = float(parts[1])
                    except ValueError:
                        pass
                continue

            # --- 區段切換 ---
            if line == "NODE_COORD_SECTION":
                section = "NODE"
                continue
            if line == "STATIONS_COORD_SECTION":
                section = "STATION"
                continue
            if line in {"DEMAND_SECTION", "DEPOT_SECTION", "EOF"}:
                section = None
                continue

            # --- 內容讀取 ---
            if section == "NODE":
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        idx = int(parts[0])
                        x = float(parts[1])
                        y = float(parts[2])
                        all_nodes[idx] = (x, y)
                    except ValueError:
                        pass
            elif section == "STATION":
                parts = line.split()
                try:
                    idx = int(parts[0])
                    station_ids.add(idx)
                except ValueError:
                    pass

    station_nodes = {i: all_nodes[i] for i in all_nodes if i in station_ids}
    other_nodes = {i: all_nodes[i] for i in all_nodes if i not in station_ids}

    return all_nodes, other_nodes, station_nodes, energy_capacity, energy_consumption


def _euclid(p: Tuple[float, float], q: Tuple[float, float]) -> float:
    return math.hypot(p[0] - q[0], p[1] - q[1])


def compute_min_detour_table(other_nodes: Dict[int, Tuple[float, float]],
                             station_nodes: Dict[int, Tuple[float, float]],
                             energy_consumption: float):
    """
    對所有 (a,b)，a!=b，a、b 為非充電站節點：
      找到使 d(a,s)+d(s,b) 最小的充電站 s。
    改為能量消耗 (distance * ENERGY_CONSUMPTION)
    回傳：
      table[a][b] = (s, energy_a_s, energy_s_b, total_energy)
    """
    station_ids = list(station_nodes.keys())
    table: Dict[int, Dict[int, Tuple[int, float, float, float]]] = {}

    for a, a_xy in other_nodes.items():
        table[a] = {}
        for b, b_xy in other_nodes.items():
            if a == b:
                continue

            best_s = None
            best_energy = float("inf")
            best_as = 0.0
            best_sb = 0.0

            for s in station_ids:
                s_xy = station_nodes[s]
                d_as = _euclid(a_xy, s_xy)
                d_sb = _euclid(s_xy, b_xy)
                e_as = d_as * energy_consumption
                e_sb = d_sb * energy_consumption
                total = e_as + e_sb
                if total < best_energy:
                    best_energy = total
                    best_s = s
                    best_as = e_as
                    best_sb = e_sb

            table[a][b] = (best_s, best_as, best_sb, best_energy)

    return table


def compute_energy_matrix_full(nodes: Dict[int, Tuple[float, float]],
                               energy_consumption: float):
    """
    建立「完整」能量矩陣：
      energy_matrix[u][v] = distance(u,v) * ENERGY_CONSUMPTION
    覆蓋傳入的 nodes 中所有節點的兩兩組合（含 depot、客戶、站點）。
    """
    energy_matrix: Dict[int, Dict[int, float]] = {}
    for u, u_xy in nodes.items():
        energy_matrix[u] = {}
        for v, v_xy in nodes.items():
            if u == v:
                energy_matrix[u][v] = 0.0
            else:
                dist = math.hypot(u_xy[0] - v_xy[0], u_xy[1] - v_xy[1])
                energy_matrix[u][v] = dist * energy_consumption
    return energy_matrix


def plot_nodes(all_nodes: Dict[int, Tuple[float, float]],
               station_nodes: Dict[int, Tuple[float, float]]):
    """
    畫出所有節點，並區分 station 與一般節點。
    - 一般節點: 藍色圓點
    - station: 紅色方形
    - 每個節點標示編號
    """
    plt.figure(figsize=(8, 6))

    other_nodes = {i: all_nodes[i] for i in all_nodes if i not in station_nodes}

    for i, (x, y) in other_nodes.items():
        plt.scatter(x, y, color='blue', marker='o')
        plt.text(x + 0.5, y + 0.5, str(i), fontsize=9, color='blue')

    for i, (x, y) in station_nodes.items():
        plt.scatter(x, y, color='red', marker='s')
        plt.text(x + 0.5, y + 0.5, str(i), fontsize=9, color='red')

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("EVRP Nodes (Blue = Customer, Red = Station)")
    plt.grid(True)
    plt.axis("equal")
    plt.show()


# 範例使用（不會自動執行）
# if __name__ == "__main__":
#     file_path = "/mnt/data/E-n22-k4.evrp"
#     all_nodes, other_nodes, station_nodes, cap, cons = parse_evrp_nodes_and_stations(file_path)
#     plot_nodes(all_nodes, station_nodes)
#
#     energy_matrix = compute_energy_matrix(other_nodes, cons)
#     print("Energy(a=1,b=2):", energy_matrix[1][2])
#
#     detour_table = compute_min_detour_table(other_nodes, station_nodes, cons)
#     print("Detour(a=1,b=2):", detour_table[1][2])
from typing import Dict, Tuple, List, Optional

def frvcp_insert_bc_with_detour(
    route: List[int],
    energy_capacity: float,
    energy_consumption: float,
    other_nodes: Dict[int, Tuple[float, float]],
    station_nodes: Dict[int, Tuple[float, float]],
    detour_table: Dict[int, Dict[int, Tuple[int, float, float, float]]],
    energy_matrix: Dict[int, Dict[int, float]],
    depot: int = 0,
) -> List[int]:
    """
    Backtracking-based Charging (BC) with detour_table:
    - route: 僅含 depot 與非充電站節點的路徑 (例如 [0, a, b, c, 0])
    - energy_capacity: Q
    - energy_consumption: h（已用在 energy_matrix / detour_table 計算）
    - other_nodes, station_nodes: 節點座標（此函式不重算距離，只用來界定集合）
    - detour_table[a][b] = (s, e_as, e_sb, e_sum): a→b 經由最佳站點 s 的能量資訊
    - energy_matrix[a][b] = 直走 a→b 所需能量
    - depot: 配送中心編號（預設 0）
    回傳：插入站點後的路徑（含 station 與 depot）
    """
 
    # 工作拷貝
    r: List[int] = route[:]   # 目前路徑（會在中途插站）
    n = len(r)

    # 追蹤自「上次充電（或出發）」以來的「剩餘能量」在各節點位置的值
    # rem_at[i]: 到達 r[i] 當下的剩餘能量
    # 初始在 depot 充滿
    rem_at: List[Optional[float]] = [None] * n
    rem_at[0] = energy_capacity

    i = 0
    # 以「邊」為單位走訪，遇到不足以直達下個節點時進入回溯插站
    while i < len(r) - 1:
        u, v = r[i], r[i + 1]
        # 若 v 是 station（理論上輸入 route 不含站，但插入後可能出現）
        # 到站就視為充滿。
        if u in station_nodes:
            rem_at[i] = energy_capacity
        # 計算直走 u->v 的能量需求
        # 新：energy_matrix 必須是 full matrix，直接查
        try:
            e_uv = energy_matrix[u][v]
        except KeyError:
            raise ValueError(f"energy_matrix 缺少邊 {u}->{v} 的能量。請用全節點矩陣（含 depot/站點/客戶）。")


        # 若目前在 u 的剩餘能量未知（發生在插站後的對齊），用前一點推導
        if rem_at[i] is None:
            # 正常流程不應到這裡；保守作法：用上一點剩餘能量扣 e_prev。
            # 但為避免越權推測，直接拋錯以提示矩陣/流程需一致。
            raise RuntimeError("內部狀態錯誤：rem_at[i] 未定。")

        # 可以直走
        if rem_at[i] >= e_uv:
            rem_at_i1 = rem_at[i] - e_uv
            # 若 v 是 station，抵達即滿電
            rem_at[i + 1] = energy_capacity if v in station_nodes else rem_at_i1
            i += 1
            continue

        # ----- 需要回溯插站（BC with detour_table）-----
        # 從當前 i 往回找可插站的位置（直到上一次充電點或最遠到 depot）
        # 定義「上一次充電點」：rem_at[j] 有定義且為「該點抵達時的真實剩餘能量」
        # 尋找最佳插入 (ip, s)：使插入後新增能量成本最小，且可達性滿足
        best_dm = float("inf")
        best_ip = None
        best_s = None

        j = i
        # 回溯期間用一個累加器重播走到 i 的能量，來計算「在 j 處的剩餘能量」
        # 但我們已經有 rem_at[j] 記錄了「當時抵達 r[j] 的剩餘能量」
        # 故以 rem_at[j] 判斷是否能到 detour_table 給的 s。
        while j >= 0:
            # 若 j 到 j+1 是原本的相鄰對（確保索引）
            if j + 1 >= len(r):
                j -= 1
                continue

            a, b = r[j], r[j + 1]

            # 僅在 a,b 為非充電站（或 depot 視為非站端點）時可用 detour_table
            if (a in other_nodes or a == depot) and (b in other_nodes or b == depot):
                # detour_table 必須有 a,b
                if a in detour_table and b in detour_table[a]:
                    s, e_as, e_sb, e_sum = detour_table[a][b]
                    # 可達性約束：
                    # 1) 在 a 的剩餘能量 rem_at[j] 必須 >= e_as，才能到 s
                    # 2) 充電後從 s → b 所需 e_sb 必須 <= energy_capacity
                    rem_a = rem_at[j]
                    if rem_a is not None and rem_a >= e_as and e_sb <= energy_capacity:
                        # 插入後的能量增量（相對於原本直接 a->b）
                        e_ab = energy_matrix[a][b]
                        dm = (e_as + e_sb) - e_ab
                        if dm < best_dm:
                            best_dm = dm
                            best_ip = j
                            best_s = s
                # 若 detour_table 沒有 a,b，則此位置不可插

            # 若在這個位置 j，a 是 station 或 depot，也可視為回溯的起點；繼續往回找
            j -= 1

        if best_ip is None or best_s is None:
            raise RuntimeError("找不到可行的插站位置（請檢查 detour_table/energy_matrix/能量參數）。")

        # 在 (r[best_ip], r[best_ip+1]) 之間插入 best_s
        insert_pos = best_ip + 1
        r.insert(insert_pos, best_s)

        # 調整 rem_at：在插入點前方維持不變；在 s 位置設為滿電後再扣到下一點
        rem_at.insert(insert_pos, energy_capacity)  # 抵達站點即滿電

        # 插入之後，重新計算從插入點起的 rem_at（僅局部即可）
        # 設置從 best_ip 開始重播（包含新插入的站點）
        start = best_ip
        # 保留 start 的 rem_at（抵達 r[start] 時的剩餘能量不變）
        for k in range(start, len(r) - 1):
            x, y = r[k], r[k + 1]
            # 若 x 是 station，抵達時已滿電
            if x in station_nodes:
                rem_at[k] = energy_capacity
            try:
                e_xy = energy_matrix[x][y]
            except KeyError:
                raise ValueError(f"energy_matrix 缺少邊 {x}->{y} 的能量。請用全節點矩陣（含 depot/站點/客戶）。")

            # 若 x 是站點，抵達時滿電
            if x in station_nodes:
                rem_at[k] = energy_capacity

            if rem_at[k] is None:
                raise RuntimeError("內部狀態錯誤：重播時 rem_at[k] 未定。")

            rem_at[k + 1] = energy_capacity if y in station_nodes else (rem_at[k] - e_xy)


        # 回到插入點繼續檢查（i 退回到 best_ip，讓下一輪檢查新插入邊）
        i = best_ip

    return r


if __name__ == "__main__":
    file_path = "instances/evrp_instances/E-n22-k4.evrp"  # 你可以改成自己的路徑
    all_nodes, other_nodes, station_nodes, cap, cons = parse_evrp_nodes_and_stations(file_path)
    energy_matrix = compute_energy_matrix_full(all_nodes, cons)
    detour_table = compute_min_detour_table(other_nodes, station_nodes, cons)
 

    route =[1, 14, 12, 5, 4, 7, 9, 1]
    new_route = frvcp_insert_bc_with_detour(
    route=route,
    energy_capacity=cap,
    energy_consumption=cons,
    other_nodes=other_nodes,
    station_nodes=station_nodes,
    detour_table=detour_table,
    energy_matrix=energy_matrix,
    depot=0,
    )
    print(new_route)