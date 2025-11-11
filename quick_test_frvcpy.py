from frvcpy import solver

INSTANCE_PATH = "instances/frvcpy_instances/E-n22-k4_frvcpy.json"
Q_INIT = 94.0  # 初始電量自行設定（上限請參考 JSON 的 max_q）
ROUTE1 = [0, 14, 21, 19,16, 0]  # 示意路徑（索引基於 JSON：0 是 depot）
ROUTE2=[0, 10, 1, 2, 5, 7, 9, 0]
ROUTE3=[0, 17, 20, 18, 15, 12, 0]
ROUTE4=[0, 13, 11, 4, 3, 6, 8, 0]
frvcp = solver.Solver(INSTANCE_PATH, ROUTE4, Q_INIT)
duration, feas_route = frvcp.solve()
print(duration)
print(feas_route)
print(frvcp._direct_route_travel_time())
