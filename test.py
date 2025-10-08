# run_hgs_with_frvcpy.py
from __future__ import annotations
from pathlib import Path
import random
from typing import List
from typing import Dict, Tuple
from pyvrp import read, RandomNumberGenerator, Solution, PenaltyManager
from pyvrp.search import LocalSearch, compute_neighbours, NODE_OPERATORS, ROUTE_OPERATORS
from pyvrp.crossover import selective_route_exchange  # 我們嘗試用 SREX；若 import 失敗，再降級
from pyvrp.stop import MaxIterations

from frvcpy_adapter import FrvcpyAdapter
from pyvrp.diversity import broken_pairs_distance as bpd  # 多樣性度量
import json
import matplotlib.pyplot as plt
from pyvrp.plotting import plot_solution

ALPHA_EV = 0.7      # EV 成本的權重（0~1），越大越重視插站後距離
K_NEIGHBORS = None    # 多樣性採樣鄰居數；None=對所有成員取平均，多樣性更穩
# ===== 你的檔案與參數 =====
VRP_INSTANCE = "instances/vrp_instances/E-n22-k4.vrp"            
FRVCPY_INSTANCE = "instances/frvcpy_instances/E-n22-k4_frvcpy.json"                        
SEED = 1
INIT_POP = 25
ITERS = 300

def _minmax_norm(vals):
    vmin, vmax = min(vals), max(vals)
    if vmax - vmin < 1e-12:
        return [0.5] * len(vals)   # 避免除零：全一樣時給中性 0.5
    return [(v - vmin) / (vmax - vmin) for v in vals]

def _diversity_scores(pop, k: int | None = K_NEIGHBORS):
    """
    回傳每個個體的多樣性貢獻（值越大=越多樣）。
    以 Broken Pairs Distance 計算與其他解的距離；預設對所有成員取平均，
    也可取前 k 大（k 最近鄰，實際上取距離大的前 k 名以鼓勵差異）。
    """
    n = len(pop)
    if n <= 1:
        return [0.0] * n

    # pairwise BPD
    dmat = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = bpd(pop[i], pop[j])
            dmat[i][j] = dmat[j][i] = d

    div = []
    for i in range(n):
        others = dmat[i][:i] + dmat[i][i + 1:]
        others.sort(reverse=True)  # 距離大者優先（越不相似越好）
        if k is None or k >= len(others):
            val = sum(others) / max(1, len(others))
        else:
            val = sum(others[:k]) / k
        div.append(val)
    return div

def _route_signature(sol: Solution) -> Tuple[Tuple[int, ...], ...]:
    # 把解序列化成可哈希的 key（每條路徑的客戶序列）
    return tuple(tuple(int(x) for x in r) for r in sol.routes())

class EVRanker:
    """
    只在 GA 的排名/淘汰時用：rank_cost = base + w_ev * ev_cost
    base 來自原本的 cost_eval（距離 + TW/容量懲罰），ev_cost 來自 FRVCPY。
    內建快取，避免重複 eval FRVCPY。
    """
    def __init__(self, base_eval, adapter: FrvcpyAdapter, w_ev: float = 1.0):
        self.base_eval = base_eval
        self.adapter = adapter
        self.w_ev = w_ev
        self._cache: Dict[Tuple[Tuple[int, ...], ...], float] = {}

    def ev_cost(self, sol: Solution) -> float:
        key = _route_signature(sol)
        if key not in self._cache:
            ev, _ = self.adapter.eval_solution(sol)  # 僅此處呼叫 FRVCPY
            self._cache[key] = ev
        return self._cache[key]

    def rank_cost(self, sol: Solution) -> float:
        base = self.base_eval.penalised_cost(sol)
        return base + self.w_ev * self.ev_cost(sol)

# 1) build_components：回傳 cost_eval，後面會用到
def build_components(vrp_path: str, seed: int = 1):
    from pyvrp import read, RandomNumberGenerator, PenaltyManager
    from pyvrp.search import LocalSearch, compute_neighbours, NODE_OPERATORS, ROUTE_OPERATORS

    data = read(vrp_path, "dimacs")
    rng = RandomNumberGenerator(seed=seed)

    neighbours = compute_neighbours(data)
    ls = LocalSearch(data, rng, neighbours)
    for op in NODE_OPERATORS:
        ls.add_node_operator(op(data))
    for op in ROUTE_OPERATORS:
        ls.add_route_operator(op(data))

    pm = PenaltyManager.init_from(data)
    cost_eval = pm.cost_evaluator()   # <── 關鍵：之後要傳給 ls.search()
    return data, rng, ls, pm, cost_eval

# 2) make_initial_population：把 ls 與 cost_eval 傳進來，用 ls.search(s, cost_eval)
def make_initial_population(data, rng, ls, cost_eval, n: int):
    from pyvrp import Solution
    from pyvrp.search import compute_neighbours, LocalSearch  # 如果你想用一次性 LS，可省略

    pop = []
    for _ in range(n):
        s = Solution.make_random(data, rng)
        pop.append(s)
    return pop

def pick_two(parents: List[Solution]) -> tuple[Solution, Solution]:
    a, b = random.sample(parents, 2)
    return a, b

def make_offspring_with_srex(p1, p2, data, cost_eval, rng, ls):
    """
    依官方 API：selective_route_exchange((p1, p2), data, cost_evaluator, rng) -> Solution
    再用 LS 修復/強化子代。
    """
    child = selective_route_exchange((p1, p2), data, cost_eval, rng)
    child = ls.search(child, cost_eval)
    return child

import matplotlib.pyplot as plt
from pyvrp.plotting import plot_solution

def _get_xy(data, idx: int):
    # 相容兩種取法：data.location(i) 或 data.locations[i]
    loc = None
    try:
        loc = data.location(idx)
    except Exception:
        pass
    if loc is None:
        try:
            loc = data.locations[idx]
        except Exception:
            return None
    x = getattr(loc, "x", None)
    y = getattr(loc, "y", None)
    if x is None or y is None:
        return None
    return float(x), float(y)

def plot_solution_with_ids(
    sol,
    data,
    save_as: str = "best_pyvrp_labeled.png",
    depot_ids: set[int] | None = None,
    fontsize: int = 8,
):
    # 先建立圖，避免 plot_solution 回 None
    fig, ax = plt.subplots(figsize=(7, 6))

    # 嘗試用官方繪圖；有些版本會回 Figure、有些回 None
    try:
        ret = plot_solution(sol, data)
        if ret is not None and hasattr(ret, "axes") and ret.axes:
            ax = ret.axes[0]
    except Exception:
        # 退路：自己簡單把路徑畫出來（若沒給 depot_ids 就只連客戶間的線）
        dep = None
        if depot_ids:
            # 任取一個 depot 作為頭尾
            dep = next(iter(depot_ids))
        for r in sol.routes():
            seq = list(r)
            if dep is not None:
                seq = [dep] + seq + [dep]
            xs, ys = [], []
            for nid in seq:
                xy = _get_xy(data, int(nid))
                if xy:
                    x, y = xy
                    xs.append(x); ys.append(y)
            if len(xs) >= 2:
                ax.plot(xs, ys, marker="o", linewidth=1)

    # 收集要標的客戶 ID（不含 depot）
    client_ids = set()
    for r in sol.routes():
        for cid in r:
            client_ids.add(int(cid))
    if depot_ids:
        client_ids = {cid for cid in client_ids if cid not in depot_ids}

    # 在各客戶座標上加文字框
    for cid in client_ids:
        xy = _get_xy(data, cid)
        if not xy:
            continue
        x, y = xy
        ax.text(
            x, y, str(cid),
            fontsize=fontsize, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7)
        )

    plt.tight_layout()
    plt.savefig(save_as, dpi=150, bbox_inches="tight")
    plt.close()



def hgs_verp(vrp_path: str,
                         frvcpy_instance: str,
                         iters: int = 300,
                         seed: int = 1):
    data, rng, ls, pm, cost_eval = build_components(vrp_path, seed)

    adapter = FrvcpyAdapter(frvcpy_instance, 94.0, id_map_pyvrp_to_frvcpy=None)
    W_EV = 1.0  # 你可先小再慢慢放大
    ranker = EVRanker(cost_eval, adapter, w_ev=W_EV)

  # --- 初始化人口後，先預熱 EV 快取 ---
    population = make_initial_population(data, rng, ls, cost_eval, INIT_POP)

    best_pyvrp: Solution | None = None
    best_ev: Solution | None = None
    best_ev_dist = float("inf")
    best_ev_details = None  # <--- 新增：儲存最佳 EV 的插站細節

    for s in population:
        if best_pyvrp is None or cost_eval.penalised_cost(s) < cost_eval.penalised_cost(best_pyvrp):
            best_pyvrp = s
        ev = ranker.ev_cost(s)  # 第一次會真的跑 FRVCPY，之後快取
        if ev < best_ev_dist:
            best_ev_dist = ev
            best_ev = s


    # 初始 best_pyvrp
    for s in population:
        if best_pyvrp is None or cost_eval.penalised_cost(s) < cost_eval.penalised_cost(best_pyvrp):
            best_pyvrp = s

    for t in range(1, iters + 1):
        # 1) 選親 + 交配（SREX）→ 子代
        p1, p2 = pick_two(population)
        child = make_offspring_with_srex(p1, p2, data, cost_eval, rng, ls)

        # 2) Local Search 改善子代

       # 3) 插站評估（FRVCPY）：計算插站後總距離
        ev_total, _details = adapter.eval_solution(child)

        # 直接寫入 EVRanker 快取，避免下一步重跑 FRVCPY
        sig = _route_signature(child)
        ranker._cache[sig] = ev_total

        # 之後拿來用就不會再呼叫 FRVCPY
        child_ev = ev_total
        child_rank = cost_eval.penalised_cost(child) + ranker.w_ev * child_ev


        # 4) 維護 best_ev（插站後）
        if ev_total < best_ev_dist:
            best_ev_dist = ev_total
            best_ev = child
            best_ev_details = _details  

        # 5) 用 PyVRP 的 penalised cost 更新 best_pyvrp（不影響搜索邏輯）
        if best_pyvrp is None or cost_eval.penalised_cost(child) < cost_eval.penalised_cost(best_pyvrp):
            best_pyvrp = child

        # 6) 簡單的 survivor：把 child 丟進人口，然後砍掉最差的一個（可換成更精緻的多樣性策略）
        # ---- Survivor selection：依據 多樣性 + EV 成本 的加權目標 ----
        population.append(child)  # 先把 child 放進來（暫時超員）

        # 先準備每個成員的 EV 成本（ranker 內含快取；child 已注入快取）
        ev_list = [ranker.ev_cost(s) for s in population]
        # 再計算多樣性分數（BPD 貢獻，越大越多樣）
        div_list = _diversity_scores(population, k=K_NEIGHBORS)

        # 正規化到 [0,1]
        ev_norm  = _minmax_norm(ev_list)
        div_norm = _minmax_norm(div_list)

        # 綜合分數（越小越好：低 EV ＆ 高多樣性）
        scores = [
            ALPHA_EV * ev_norm[i] + (1 - ALPHA_EV) * (1 - div_norm[i])
            for i in range(len(population))
        ]

        # 移除分數最高者（最差的那個）
        worst_idx = max(range(len(population)), key=lambda i: scores[i])
        population.pop(worst_idx)


        if t % 10 == 0:
            avg_div = sum(div_list) / len(div_list)
            avg_ev  = sum(ev_list)  / len(ev_list)
            print(
                f"[Iter {t}] child: EV={child_ev:.3f} | "
                f"best_EV={best_ev_dist:.3f} | "
                f"avg_div={avg_div:.3f} | avg_EV={avg_ev:.3f}"
            )
    print(best_ev_details)
    print(best_pyvrp)
    # 若你的 PyVRP depot 是 0，就用 {0}；若是 1 就用 {1}
    plot_solution_with_ids(best_pyvrp, data, save_as="best_pyvrp_labeled.png", depot_ids={0})
    print("✅ 已輸出：best_pyvrp_labeled.png（含客戶編號）")

    return {
        "best_pyvrp": best_pyvrp,
        "best_ev": best_ev,
        "best_ev_distance": best_ev_dist,
    }

if __name__ == "__main__":
    res = hgs_verp(
        VRP_INSTANCE,
        FRVCPY_INSTANCE,
        iters=ITERS,
        seed=SEED,
    )

    print("\n=== Result ===")
    print("Best EV distance (after charging insertion):", res["best_ev_distance"])
