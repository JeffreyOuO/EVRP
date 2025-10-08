from pyvrp import read, RandomNumberGenerator, PenaltyManager, Population, Solution, GeneticAlgorithm
from pyvrp.diversity import broken_pairs_distance
from pyvrp.search import LocalSearch, compute_neighbours, NODE_OPERATORS, ROUTE_OPERATORS
from pyvrp.crossover import selective_route_exchange as srex
from pyvrp.stop import MaxIterations

# ---- 1) 照官方組 components ----
data = read("instances/vrp_instances/E-n22-k4.vrp", "dimacs")
rng = RandomNumberGenerator(seed=1)

neighbours = compute_neighbours(data)
ls = LocalSearch(data, rng, neighbours)
for op in NODE_OPERATORS: ls.add_node_operator(op(data))
for op in ROUTE_OPERATORS: ls.add_route_operator(op(data))

pm = PenaltyManager.init_from(data)
pop = Population(broken_pairs_distance)

init = [Solution.make_random(data, rng) for _ in range(25)]
algo = GeneticAlgorithm(data, pm, rng, pop, ls, srex, init)  # 官方推薦的 SREX 交配 :contentReference[oaicite:2]{index=2}

# ---- 2) 每回合跑 1 次 + 立刻做 FRVCPY 插站評估 ----
from frvcpy_adapter import FrvcpyAdapter
adapter = FrvcpyAdapter("instances/frvcpy_instances/E-n22-k4_frvcpy.json", q_init=80.0)

best_ev_sol = None
best_ev_cost = float("inf")

for t in range(300):
    res = algo.run(stop=MaxIterations(1))   # 單步前進；GA 內部會做：選親→SREX→LS→加入族群
    curr = res.best                          # 目前已知最佳解（公開 API） :contentReference[oaicite:3]{index=3}

    ev_cost, _ = adapter.eval_solution(curr) # 用 FRVCPY 對「當前最佳」做插站，取插站後成本
    if ev_cost < best_ev_cost:
        best_ev_cost = ev_cost
        best_ev_sol = curr

    if (t + 1) % 10 == 0:
        # 你也可列印 pm.cost_evaluator().penalised_cost(curr) 做對照
        print(f"[Iter {t+1}] best EV(after charging) = {best_ev_cost:.3f}")
