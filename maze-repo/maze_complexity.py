"""
==============================================================================
3차원 미로의 흡수시간 기반 복잡도 계산
  기반 : Amancio, Oliveira Jr., Costa (2011), Physica A 390, 4673-4683
         absorption_time_theory.pdf (s → p → Θ → Ψ → τ)

  전체 흐름 (보고서·발표의 단계 번호와 같음)
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ [1] 미로 데이터    : 간선(두 끝 정점의 좌표, 길이) + 입구 + 출구          │
  │ [2] 미로 해석      : 데이터 → '정점: {이웃: 길이}' 그래프, 오류 검사      │
  │ [3] 직선 정점 삭제 : 일직선으로 지나가는 정점을 지우고 길이를 합침        │
  │ [4] 흡수시간 계산  : ① 길이 가중치 → τ_length  ② 차수 가중치 → τ_degree  │
  │ [5] 최종 복잡도    : C = √(τ_length × τ_degree)                          │
  └─────────────────────────────────────────────────────────────────────────┘

  Google Colab / 일반 파이썬 어디서나 그대로 실행 가능 (numpy만 필요).
==============================================================================
"""

import numpy as np
from collections import deque


# =============================================================================
# [1] 미로 데이터
# =============================================================================
# edges : (정점 A 좌표, 정점 B 좌표, 통로 길이)
#   - 좌표는 (x, y, z). 통로는 한 축 방향으로만 곧게 뻗어야 함
#     (예: (0,0,0)→(3,0,0) 은 가능, (0,0,0)→(1,1,0) 은 불가).
#   - 반대 방향은 따로 적지 않음 (무방향).
# 아래 예시는 3×3×3 격자 미로 (모든 통로 길이 1).

MAZE = {
    "entry": (0, 0, 0),
    "exit":  (2, 2, 2),
    "edges": [
        # z = 0 층
        ((0,0,0),(1,0,0),1), ((1,0,0),(2,0,0),1), ((0,1,0),(1,1,0),1),
        ((0,2,0),(1,2,0),1), ((1,2,0),(2,2,0),1), ((0,0,0),(0,1,0),1),
        ((1,0,0),(1,1,0),1), ((1,1,0),(1,2,0),1), ((2,0,0),(2,1,0),1),
        ((2,1,0),(2,2,0),1),
        # z = 1 층
        ((0,0,1),(1,0,1),1), ((0,1,1),(1,1,1),1), ((1,1,1),(2,1,1),1),
        ((0,2,1),(1,2,1),1), ((0,0,1),(0,1,1),1), ((0,1,1),(0,2,1),1),
        ((1,0,1),(1,1,1),1), ((2,0,1),(2,1,1),1), ((2,1,1),(2,2,1),1),
        # z = 2 층
        ((0,0,2),(1,0,2),1), ((1,0,2),(2,0,2),1), ((0,1,2),(1,1,2),1),
        ((0,2,2),(1,2,2),1), ((1,2,2),(2,2,2),1), ((0,0,2),(0,1,2),1),
        ((0,1,2),(0,2,2),1), ((1,0,2),(1,1,2),1), ((1,1,2),(1,2,2),1),
        ((2,0,2),(2,1,2),1),
        # 층 사이 수직 통로
        ((0,0,0),(0,0,1),1), ((2,0,0),(2,0,1),1), ((1,1,0),(1,1,1),1),
        ((0,2,0),(0,2,1),1), ((2,2,0),(2,2,1),1), ((0,0,1),(0,0,2),1),
        ((2,1,1),(2,1,2),1), ((1,2,1),(1,2,2),1), ((0,2,1),(0,2,2),1),
        ((2,2,1),(2,2,2),1),
    ],
}


# =============================================================================
# [2] 미로 해석
# =============================================================================
def direction(a, b):
    """a에서 b로 가는 방향 → (축, 부호).  축: 0=x, 1=y, 2=z"""
    diff = [q - p for p, q in zip(a, b)]
    axis = next(k for k in range(3) if diff[k] != 0)
    return axis, (1 if diff[axis] > 0 else -1)


def read_maze(maze):
    """
    미로 데이터를 그래프로 바꾼다.
      graph[정점] = {이웃 정점: 통로 길이, ...}
    잘못된 데이터는 여기서 바로 오류로 알려 준다.
    """
    graph = {}
    for a, b, length in maze["edges"]:
        a, b = tuple(a), tuple(b)
        changed_axes = sum(p != q for p, q in zip(a, b))
        if changed_axes != 1:
            raise ValueError(f"통로는 한 축 방향이어야 함: {a} - {b}")
        if length <= 0:
            raise ValueError(f"통로 길이는 양수여야 함: {a} - {b}, L={length}")
        graph.setdefault(a, {})[b] = length
        graph.setdefault(b, {})[a] = length

    entry, exit_ = tuple(maze["entry"]), tuple(maze["exit"])
    for name, v in (("입구", entry), ("출구", exit_)):
        if v not in graph:
            raise ValueError(f"{name} {v} 가 어떤 통로에도 연결되어 있지 않음")

    # 입구에서 모든 정점에 갈 수 있는지 확인 (못 가는 정점이 있으면 계산 불가)
    seen, queue = {entry}, deque([entry])
    while queue:
        v = queue.popleft()
        for u in graph[v]:
            if u not in seen:
                seen.add(u)
                queue.append(u)
    if len(seen) != len(graph):
        lost = sorted(set(graph) - seen)
        raise ValueError(f"입구에서 갈 수 없는 정점이 있음: {lost}")

    return graph, entry, exit_


# =============================================================================
# [3] 직선 정점 삭제
# =============================================================================
def is_straight(graph, v):
    """이웃이 정확히 2개이고, 두 이웃이 같은 축의 반대쪽에 있으면 '직선 정점'."""
    if len(graph[v]) != 2:
        return False
    u, w = graph[v]
    (axis_u, sign_u), (axis_w, sign_w) = direction(v, u), direction(v, w)
    return axis_u == axis_w and sign_u != sign_w


def remove_straight_vertices(graph, keep):
    """
    직선 정점을 지우고, 양쪽 통로를 길이를 더한 하나의 통로로 합친다.
      u ──L1── v ──L2── w   →   u ──(L1+L2)── w
    keep(입구·출구)은 직선이어도 지우지 않는다.
    (정점을 지워도 다른 정점의 '직선 여부'는 바뀌지 않으므로 한 번 훑으면 끝)
    """
    g = {v: dict(nb) for v, nb in graph.items()}     # 원본은 그대로 둠
    removed = []
    for v in sorted(g):
        if v in keep or not is_straight(g, v):
            continue
        u, w = g[v]
        if w in g[u]:              # 이미 u-w 통로가 있으면 합치지 않음(안전장치)
            continue
        new_length = g[v][u] + g[v][w]
        del g[u][v], g[w][v], g[v]
        g[u][w] = g[w][u] = new_length
        removed.append(v)
    return g, removed


# =============================================================================
# [4] 흡수시간 계산
# =============================================================================
# ── 4-1. 통로 가중치 W_ij (i에서 j로 갈 때 얼마나 끌리는가) ─────────────────
def weight_length(g, i, j):
    """① 길이 가중치 : W_ij = 1 / L_ij   (짧은 통로일수록 잘 감)"""
    return 1.0 / g[i][j]


def weight_degree(g, i, j):
    """② 차수 가중치 : W_ij = 1 / k_eff(j),  k_eff = max(2, 차수)   (j = 도착 정점)"""
    return 1.0 / max(2, len(g[j]))


# ── 4-2. 정점 방문 가중치 w_j (축 수 a_j 반영) ──────────────────────────────
def count_axes(g, v):
    """정점 v에서 길이 뻗어 있는 축의 개수 a (1~3). +x와 -x는 같은 축."""
    return len({direction(v, u)[0] for u in g[v]})


VISIT_WEIGHT = {
    "none": lambda a: 1,              # 축 무시 (논문 원래 식 τ = Ψu)
    "a-1":  lambda a: max(1, a - 1),  # 확정안 (2026-10-06)
    "a":    lambda a: a,              # 교차검증용
    "1/a":  lambda a: 1 / a,          # 교차검증용
}


# ── 4-3. 흡수시간 (논문 / 이론 문서 순서 그대로) ────────────────────────────
def absorption_time(g, entry, exit_, edge_weight, visit_weight="a-1"):
    """
    s_i = Σ_j W_ij  →  p_ij = W_ij / s_i  (출구는 p_ii = 1)
    →  Θ = 출구를 뺀 부분  →  Ψ = (I − Θ)⁻¹  →  τ_i = Σ_j Ψ_ij · w_j
    """
    nodes = sorted(g)
    num = {v: n for n, v in enumerate(nodes)}          # 정점 → 행렬 번호

    W = np.zeros((len(nodes), len(nodes)))
    for i in g:
        for j in g[i]:
            W[num[i], num[j]] = edge_weight(g, i, j)

    s = W.sum(axis=1)                                   # s_i
    P = W / s[:, None]                                  # p_ij
    P[num[exit_]] = 0.0
    P[num[exit_], num[exit_]] = 1.0                     # 출구 = 흡수 상태

    transient = [v for v in nodes if v != exit_]        # 출구를 뺀 정점들
    t = [num[v] for v in transient]
    Theta = P[np.ix_(t, t)]                             # Θ
    Psi = np.linalg.inv(np.eye(len(t)) - Theta)         # Ψ
    w = np.array([VISIT_WEIGHT[visit_weight](count_axes(g, v)) for v in transient])
    tau = Psi @ w                                       # τ

    return {
        "tau_entry": float(tau[transient.index(entry)]),
        "tau": dict(zip(transient, tau)),               # 정점 좌표로 바로 찾기
        "W": W, "P": P, "Theta": Theta, "Psi": Psi, "w": w,
        "nodes": nodes, "transient": transient,
    }


# =============================================================================
# [5] 최종 복잡도
# =============================================================================
MEAN = {
    "geometric":  lambda a, b: (a * b) ** 0.5,          # 확정안: 기하평균
    "arithmetic": lambda a, b: (a + b) / 2,             # 교차검증용
    "harmonic":   lambda a, b: 2 * a * b / (a + b),     # 교차검증용
}


def maze_complexity(maze, visit_weight="a-1", mean="geometric"):
    """[2] → [3] → [4] → [5] 를 차례로 실행하고, 단계별 결과를 모두 돌려준다."""
    graph, entry, exit_ = read_maze(maze)                               # [2]
    g, removed = remove_straight_vertices(graph, keep={entry, exit_})  # [3]

    if entry == exit_:
        return {"complexity": 0.0, "graph": graph, "simple": g, "removed": removed,
                "entry": entry, "exit": exit_}

    res_L = absorption_time(g, entry, exit_, weight_length, visit_weight)   # [4]①
    res_D = absorption_time(g, entry, exit_, weight_degree, visit_weight)   # [4]②
    C = MEAN[mean](res_L["tau_entry"], res_D["tau_entry"])                   # [5]

    return {
        "complexity": C,
        "tau_length": res_L["tau_entry"], "tau_degree": res_D["tau_entry"],
        "length": res_L, "degree": res_D,
        "graph": graph, "simple": g, "removed": removed,
        "entry": entry, "exit": exit_,
        "visit_weight": visit_weight, "mean": mean,
    }


# =============================================================================
# 보고서용 출력 (계산과 분리)
# =============================================================================
def print_report(r, show_matrices=False):
    line = "=" * 70
    n_edges = lambda g: sum(len(nb) for nb in g.values()) // 2

    print(line); print("[2] 미로 해석"); print(line)
    print(f"  정점 {len(r['graph'])}개, 통로 {n_edges(r['graph'])}개")
    print(f"  입구 {r['entry']}  →  출구 {r['exit']}")

    print(line); print("[3] 직선 정점 삭제"); print(line)
    print(f"  삭제한 정점 {len(r['removed'])}개: {r['removed']}")
    print(f"  남은 정점 {len(r['simple'])}개, 통로 {n_edges(r['simple'])}개")
    print(f"  {'정점':<12}{'차수':>4}{'k_eff':>6}{'축 수 a':>8}")
    for v in sorted(r["simple"]):
        d = len(r["simple"][v])
        tag = "  ← 입구" if v == r["entry"] else ("  ← 출구" if v == r["exit"] else "")
        print(f"  {str(v):<12}{d:>4}{max(2, d):>6}{count_axes(r['simple'], v):>8}{tag}")

    if r["entry"] == r["exit"]:
        print("\n  입구 = 출구 → 복잡도 0"); return

    print(line); print(f"[4] 흡수시간  (방문 가중치: {r['visit_weight']})"); print(line)
    print(f"  ① 길이 가중치 W = 1/L       →  τ_length = {r['tau_length']:.4f}")
    print(f"  ② 차수 가중치 W = 1/k_eff(j) →  τ_degree = {r['tau_degree']:.4f}")
    if show_matrices:
        np.set_printoptions(precision=3, suppress=True, linewidth=160)
        for key in ("length", "degree"):
            print(f"\n  ── {key}: 전이확률 P ──\n{r[key]['P']}")
            print(f"  ── {key}: 기본행렬 Ψ ──\n{r[key]['Psi']}")

    print(line); print(f"[5] 최종 복잡도  (결합: {r['mean']})"); print(line)
    print(f"  C = {r['complexity']:.4f}")


def print_cross_check(maze):
    """방문 가중치 × 평균 방식 조합별 복잡도 비교표 (교차검증용)."""
    means = list(MEAN)
    print("\n" + "=" * 70); print("교차검증: 방문 가중치 × 결합 방식"); print("=" * 70)
    print(f"  {'방문 가중치':<10}" + "".join(f"{m:>14}" for m in means))
    for vw in VISIT_WEIGHT:
        row = [maze_complexity(maze, vw, m)["complexity"] for m in means]
        print(f"  {vw:<10}" + "".join(f"{c:>14.4f}" for c in row))


# =============================================================================
# 실행
# =============================================================================
if __name__ == "__main__":
    result = maze_complexity(MAZE)
    print_report(result, show_matrices=False)
    print_cross_check(MAZE)
