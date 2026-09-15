"""
==============================================================================
연결 그래프의 흡수시간 기반 복잡도 계산
Amancio et al., Physica A 390 (2011) 4673-4683 의 흡수시간 알고리즘 사용

  ┌ 논문 부분 : Absorbing Markov Chain → P → Θ → N=(I-Θ)^(-1) → t=N·1
  └ 추가 부분 : Weight 정의를 두 가지로 나눠 알고리즘을 두 번 수행하고,
              입구 노드의 두 흡수시간을 기하평균으로 결합

  ① 길이 기반 : W_ij = 1 / L_ij            → T_length
  ② 차수 기반 : W_ij = 1 / k_eff(j),
                k_eff(j) = max(2, deg(j))  → T_degree

  최종 복잡도 : Complexity = √(T_length × T_degree)

Google Colab에서 그대로 실행 가능.
==============================================================================
"""

# ──────────────────────────────────────────────────────────────────────────
# 라이브러리 import
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy.linalg import inv


# =============================================================================
# [입력]  ← 사용자는 이 부분만 수정하면 됨
# =============================================================================
# edges 형식: (노드1, 노드2, 간선 길이)
# 무방향 그래프이며 자기 자신 간선은 허용하지 않는다.
edges = [
    ("A", "B", 5),
    ("A", "C", 3),
    ("B", "C", 2),
    ("B", "D", 8),
    ("C", "D", 2),
    ("C", "E", 6),
    ("D", "E", 4),
    ("D", "F", 7),
    ("E", "F", 3),
]
entry_node = "A"  # 입구
exit_node  = "F"  # 출구 (Absorbing state)


# =============================================================================
# 1. 그래프 생성 함수  [논문 부분]
# =============================================================================
def build_graph(edges):
    """
    edges 리스트로례닐 무방향 가짐길한 G=(V,E)를 생성.
    각 간선에 'length' 속성으로 통로 길이를 저장한다.
    """
    G = nx.Graph()
    for u, v, L in edges:
        G.add_edge(u, v, length=L)
    return G


# =============================================================================
# 2. 차수 및 유효차수(Effective Degree) 계산  [추가 부분]
# =============================================================================
def compute_degrees(G):
    """
    각 노드의 차수(degree)를 구하고,
    유효차수 k_eff(v) = max(2, deg(v)) 를 함께 계산한다.

    유효차수를 쓰는 이유:
      차수 1(막다른 길)과 차수 2(직선 통로)는 실제로 '분기'가 없는 지점이므로
      복잡도 관점에서 동일하게 취급한다. 따라서 하한을 2로 잡는다.
    """
    deg = dict(G.degree())
    k_eff = {v: max(2, d) for v, d in deg.items()}
    return deg, k_eff


# =============================================================================
# 3. 가중치 행렬 W 계산  [추가 부분]
# =============================================================================
def build_W_length(G, node_order):
    """
    길이 기반 Weight :  W_ij = 1 / L_ij
    간선이 없으면 W_ij = 0.
    """
    n = len(node_order)
    idx = {v: i for i, v in enumerate(node_order)}
    W = np.zeros((n, n))
    for u, v, data in G.edges(data=True):
        L = data["length"]
        W[idx[u], idx[v]] = 1.0 / L
        W[idx[v], idx[u]] = 1.0 / L
    return W


def build_W_degree(G, k_eff, node_order):
    """
    차수 기반 Weight :  W_ij = 1 / k_eff(j)
    ※ '도착 노드' j 의 유효차수를 사용한다.
       그러므로 무방향 그래프여도 W_ij ≠ W_ji 일 수 있다.
    """
    n = len(node_order)
    idx = {v: i for i, v in enumerate(node_order)}
    W = np.zeros((n, n))
    for u, v in G.edges():
        i, j = idx[u], idx[v]
        W[i, j] = 1.0 / k_eff[v]   # i → j 로 갈 때는 j의 유효차수
        W[j, i] = 1.0 / k_eff[u]   # j → i 로 갈 때는 i의 유효차수
    return W


# =============================================================================
# 4. 전이확률행렬 P 생성  [논문 부분]
# =============================================================================
def build_transition_matrix(W, node_order, exit_node):
    """
    논문 식 (9)(10) :
      s_i = Σ_j W_ij
      p_ij = W_ij / s_i    (출구가 아닌 노드)
      p_ii = 1             (출구 노드는 absorbing state)
    """
    n = W.shape[0]
    idx = {v: i for i, v in enumerate(node_order)}
    exit_idx = idx[exit_node]

    s = W.sum(axis=1)                       # 노드 강도 s_i
    P = np.zeros((n, n))
    for i in range(n):
        if i == exit_idx:
            P[i, i] = 1.0                   # 흡수 상태
        elif s[i] > 0:
            P[i] = W[i] / s[i]              # 행 정규화 = 전이 확률
    return P


# =============================================================================
# 5. Transient Matrix Θ 추출  [논문 부분]
# =============================================================================
def extract_theta(P, node_order, exit_node):
    """
    정준형 Π = [[Θ, Φ], [0, I]] 에서 Θ 블록만 추출.
    Θ 는 transient 노드끼리의 전이 확률 부분행렬.
    """
    idx = {v: i for i, v in enumerate(node_order)}
    exit_idx = idx[exit_node]
    transient_idx = [i for i in range(len(node_order)) if i != exit_idx]
    transient_nodes = [node_order[i] for i in transient_idx]
    Theta = P[np.ix_(transient_idx, transient_idx)]
    return Theta, transient_nodes


# =============================================================================
# 6. Fundamental Matrix  N = (I - Θ)^(-1)   [논문 부분, 식 (13)]
# =============================================================================
def fundamental_matrix(Theta):
    """
    N = Σ_{k=0}^∞ Θ^k = (I - Θ)^(-1)
    Θ 의 스펙트럼 반경 < 1 이므로 역행렬이 존재하며 급수가 수렴한다.
    """
    n = Theta.shape[0]
    N = inv(np.eye(n) - Theta)
    return N


# =============================================================================
# 7. 평균 흡수시간  t = N · 1   [논문 부분, 식 (19)(20)]
# =============================================================================
def absorption_time(N):
    """
    t_i = Σ_j N_ij  (N의 i번째 행의 합)
    = 노드 i에서 출발한 입자가 출구에 흡수되기까지의 기대 스텝 수.
    """
    return N.sum(axis=1)


# =============================================================================
# 8. 파이프라인 (한 가지 Weight로 논문 알고리즘을 전체 수행)  [논문 부분]
# =============================================================================
def run_pipeline(W, node_order, exit_node, label):
    """
    Weight 행렬 W 하나를 받아 논문 알고리즘을 그대로 수행하고,
    중간 산출물을 딕셔너리로 돌려준다.
    """
    P = build_transition_matrix(W, node_order, exit_node)
    Theta, transient_nodes = extract_theta(P, node_order, exit_node)
    N = fundamental_matrix(Theta)
    t = absorption_time(N)

    result = {
        "label":            label,
        "W":                W,
        "P":                P,
        "Theta":            Theta,
        "transient_nodes":  transient_nodes,
        "N":                N,
        "t":                t,
    }
    return result


# =============================================================================
# 9. 시각화  [추가 부분]
# =============================================================================
def visualize_graph(G, deg, k_eff, entry_node, exit_node):
    """
    입구 = 초록, 출구 = 빨강, 나머지는 옅은 회색.
    각 노드 옆에 deg / eff, 각 간선에 L=길이 표시.
    """
    pos = nx.spring_layout(G, seed=42)

    node_colors = []
    for v in G.nodes():
        if v == entry_node:  node_colors.append("#7ED957")   # 초록
        elif v == exit_node: node_colors.append("#F26D6D")   # 빨강
        else:                node_colors.append("#DDDDDD")

    plt.figure(figsize=(8, 6))
    nx.draw_networkx_edges(G, pos, width=1.5, edge_color="#888")
    nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                           edgecolors="#333", node_size=900, linewidths=1)

    # 노드 라벨: 이름 + (deg, eff)
    node_labels = {v: f"{v}\ndeg={deg[v]}, eff={k_eff[v]}" for v in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=9)

    # 간선 라벨: L=길이
    edge_labels = {(u, v): f"L={d['length']}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9)

    plt.title(f"Graph  (entry={entry_node}, exit={exit_node})")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


# =============================================================================
# 10. 결과 출력 도우미 (행렬을 DataFrame으로 예쁘게)
# =============================================================================
def show_matrix(mat, row_labels, col_labels, title, fmt="{:.4f}"):
    print(f"\n── {title} ──")
    df = pd.DataFrame(mat, index=row_labels, columns=col_labels)
    with pd.option_context("display.float_format", fmt.format,
                           "display.width", 200):
        print(df)


def show_vector(vec, labels, title, fmt="{:.4f}"):
    print(f"\n── {title} ──")
    df = pd.Series(vec, index=labels).to_frame(name="value")
    with pd.option_context("display.float_format", fmt.format):
        print(df)


# =============================================================================
# 11. 메인 실행부
# =============================================================================
def main():
    # -------------------------------------------------------------------------
    # 그래프 준비
    # -------------------------------------------------------------------------
    G = build_graph(edges)
    node_order = sorted(G.nodes())   # 고정된 순서 (행렬 인덱스 기준)
    n = len(node_order)

    print("=" * 74)
    print("입력 그래프 정보")
    print("=" * 74)
    print(f"노드 수 : {n}   ({node_order})")
    print(f"간선 수 : {G.number_of_edges()}")
    print(f"입구    : {entry_node}")
    print(f"출구    : {exit_node} (absorbing state)")

    # ① 각 노드의 차수, ② Effective Degree
    deg, k_eff = compute_degrees(G)
    print("\n── ① 노드 차수  /  ② 유효차수 k_eff = max(2, deg) ──")
    df_deg = pd.DataFrame(
        {"degree": [deg[v] for v in node_order],
         "k_eff":  [k_eff[v] for v in node_order]},
        index=node_order,
    )
    print(df_deg)

    # ③ 각 간선의 길이
    print("\n── ③ 간선 길이 ──")
    df_edges = pd.DataFrame(
        [(u, v, d["length"]) for u, v, d in G.edges(data=True)],
        columns=["u", "v", "length"],
    )
    print(df_edges.to_string(index=False))

    # -------------------------------------------------------------------------
    # 첫 번째 계산 : 길이 기반  W_ij = 1 / L_ij
    # -------------------------------------------------------------------------
    print("\n" + "=" * 74)
    print("[1] 길이 기반 흡수시간 계산")
    print("=" * 74)

    W_L = build_W_length(G, node_order)
    show_matrix(W_L, node_order, node_order, "④ 길이 기반 Weight W = 1/L")

    res_L = run_pipeline(W_L, node_order, exit_node, "length")
    show_matrix(res_L["P"], node_order, node_order,
                "⑤ 길이 기반 전이확률행렬 P")
    show_matrix(res_L["N"], res_L["transient_nodes"], res_L["transient_nodes"],
                "⑥ 길이 기반 Fundamental Matrix  N = (I-Θ)^(-1)")
    show_vector(res_L["t"], res_L["transient_nodes"],
                "⑦ 모든 노드의 길이 기반 흡수시간 t")

    T_length = res_L["t"][res_L["transient_nodes"].index(entry_node)]
    print(f"\n⑧ 입구 '{entry_node}' 의 T_length = {T_length:.6f}")

    # -------------------------------------------------------------------------
    # 두 번째 계산 : 차수 기반  W_ij = 1 / k_eff(j)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 74)
    print("[2] 차수 기반 흡수시간 계산")
    print("=" * 74)

    W_D = build_W_degree(G, k_eff, node_order)
    show_matrix(W_D, node_order, node_order,
                "⑨ 차수 기반 Weight W_ij = 1 / k_eff(j)")

    res_D = run_pipeline(W_D, node_order, exit_node, "degree")
    show_matrix(res_D["P"], node_order, node_order,
                "⑩ 차수 기반 전이확률행렬 P")
    show_matrix(res_D["N"], res_D["transient_nodes"], res_D["transient_nodes"],
                "⑪ 차수 기반 Fundamental Matrix  N = (I-Θ)^(-1)")
    show_vector(res_D["t"], res_D["transient_nodes"],
                "⑫ 모든 노드의 차수 기반 흡수시간 t")

    T_degree = res_D["t"][res_D["transient_nodes"].index(entry_node)]
    print(f"\n⑬ 입구 '{entry_node}' 의 T_degree = {T_degree:.6f}")

    # -------------------------------------------------------------------------
    # 최종 복잡도  Complexity = √(T_length × T_degree)   [추가 부분]
    # -------------------------------------------------------------------------
    complexity = np.sqrt(T_length * T_degree)
    print("\n" + "=" * 74)
    print("⑭ 최종 복잡도")
    print("=" * 74)
    print(f"  T_length   = {T_length:.6f}")
    print(f"  T_degree   = {T_degree:.6f}")
    print(f"  Complexity = √(T_length × T_degree) = {complexity:.6f}")

    # -------------------------------------------------------------------------
    # 그래프 시각화
    # -------------------------------------------------------------------------
    visualize_graph(G, deg, k_eff, entry_node, exit_node)


if __name__ == "__main__":
    main()


# =============================================================================
# [입력 형식 안내]  ← 다른 그래프를 쓰고 싶을 때는 아래 규칙만 지키면 됩니다.
# =============================================================================
# 1) edges 는 (노드이름, 노드이름, 길이) 튜플의 리스트입니다.
#    - 노드 이름은 문자열/숫자 무엇이든 됩니다.
#    - 길이는 양수여야 합니다 (W = 1/L 에서 0으로 나누는 것을 방지).
#    - 무방향 그래프이므로 (A,B,5) 하나만 넣으면 됩니다. 반대 방향을 따로
#      넣을 필요는 없습니다.
#
# 2) entry_node, exit_node 는 위 edges 에 등장하는 노드 이름 중 하나여야
#    합니다.
#
# 3) 그래프는 반드시 연결 그래프여야 합니다.
#    (연결되지 않은 노드가 있으면 흡수 시간이 발산합니다.)
#
# 4) 코드의 다른 부분은 수정할 필요가 없습니다.
#    입력만 바꿔서 재실행하면 됩니다.
# =============================================================================
