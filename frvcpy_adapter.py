# frvcpy_adapter.py
from __future__ import annotations
from typing import Dict, List, Tuple
from pyvrp import Solution
from frvcpy.solver import Solver

class FrvcpyAdapter:
    """
    封裝：把 PyVRP 的解，轉成 FRVCPY 可讀的路徑，逐條做插站並加總距離。
    """

    def __init__(self, frvcpy_instance_path: str, q_init: float,
                 id_map_pyvrp_to_frvcpy: Dict[int, int] | None = None,
                 depot_id_pyvrp: int = 0, depot_id_frvcpy: int | None = None):
        """
        id_map_pyvrp_to_frvcpy: 把 PyVRP 的節點 ID 映到 FRVCPY instance 的節點 ID。
            - 若為 None，預設視為同一套編號（你已做轉檔就填你的 map）。
        depot_id_pyvrp:   PyVRP 內部的 depot ID（通常 0）
        depot_id_frvcpy:  FRVCPY instance 中的 depot ID（若 None，沿用 map 後的 depot）
        """
        self.instance_path = frvcpy_instance_path
        self.q_init = q_init
        self.map = id_map_pyvrp_to_frvcpy or {}
        self.depot_py = depot_id_pyvrp
        self.depot_fr = depot_id_frvcpy if depot_id_frvcpy is not None else self._map_id(depot_id_pyvrp)

    def _map_id(self, pid: int) -> int:
        return self.map.get(pid, pid)  # 預設身分映射

    def _sol_to_route_id_lists(self, sol: Solution) -> List[List[int]]:
        """
        將解拆成多條路徑，每條： depot → 客戶們 → depot
        並做 ID 映射使其符合 FRVCPY 的 instance。
        """
        routes: List[List[int]] = []
        for r in sol.routes():
            seq = [self.depot_fr] + [self._map_id(cid) for cid in r] + [self.depot_fr]
            routes.append(seq)
        return routes

    def eval_solution(self, sol: Solution) -> Tuple[float, List[dict]]:
        """
        回傳：(插站後總距離/時間, 各路徑插站明細)
        注意：FRVCPY 通常回傳「時間」，若要把它當距離，請在你的 instance/轉檔時統一單位。
        """
        route_ids_list = self._sol_to_route_id_lists(sol)
        total = 0.0
        details: List[dict] = []
        for idx, route in enumerate(route_ids_list, 1):
            frvcp = Solver(self.instance_path, route, self.q_init)
            duration, feas_route = frvcp.solve()
            total += float(duration)
            details.append(
                {"route_id": idx, "duration": float(duration), "sequence": feas_route}
            )
        return total, details
