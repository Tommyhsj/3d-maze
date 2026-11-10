"""
3×3×3 미로 흡수 시간 계산 (그래프 단순화 포함)
Amancio et al., Physica A 390 (2011) 4673-4683

이 버전은 흡수 시간 계산 전에 그래프 단순화 단계를 거친다:
  - 차수가 2이고 두 이웃이 직선상에 있는 노드(단순 통과점)를 제거
  - 그 노드를 지나던 두 간선을 하나의 간선으로 합침
  - 입구와 출구는 항상 유지

논문의 굴절점 처리와의 관계:
  논문은 굴절점(차수 2)까지 노드로 넣는다. "결정 지점"이라는 이유에서다.
  이 코드는 그 결정 여지가 없는 노드(직선으로 지나가는 지점)만 골라
  제거해서, 진짜 갈림길과 꺾임점만 남긴 축약 그래프를 만든다.
"""

from collections import defaultdict
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# 0. 노드 정의
# ─────────────────────────────────────────────────────────────────────────────
def idx(x, y, z): return x + 3*y + 9*z
def coord(i):
    z, r = divmod(i, 9); y, x = divmod(r, 3); return x, y, z

N = 27
INPUT_NODE  = idx(0, 0, 0)
OUTPUT_NODE = idx(2, 2, 2)

# ─────────────────────────────────────────────────────────────────────────────
# 1. 미로 통로(에지) 정의
# ─────────────────────────────────────────────────────────────────────────────
raw_edges = [
    # Z=0 층
    ((0,0,0),(1,0,0)), ((1,0,0),(2,0,0)),
    ((0,1,0),(1,1,0)),
    ((0,2,0),(1,2,0)), ((1,2,0),(2,2,0)),
    ((0,0,0),(0,1,0)),
    ((1,0,0),(1,1,0)), ((1,1,0),(1,2,0)),
    ((2,0,0),(2,1,0)), ((2,1,0),(2,2,0)),
    # Z=1 층
    ((0,0,1),(1,0,1)),
    ((0,1,1),(1,1,1)), ((1,1,1),(2,1,1)),
    ((0,2,1),(1,2,1)),
    ((0,0,1),(0,1,1)), ((0,1,1),(0,2,1)),
    ((1,0,1),(1,1,1)),
    ((2,0,1),(2,1,1)), ((2,1,1),(2,2,1)),
    # Z=2 층
    ((0,0,2),(1,0,2)), ((1,0,2),(2,0,2)),
    ((0,1,2),(1,1,2)),
    ((0,2,2),(1,2,2)), ((1,2,2),(2,2,2)),
    ((0,0,2),(0,1,2)), ((0,1,2),(0,2,2)),
    ((1,0,2),(1,1,2)), ((1,1,2),(1,2,2)),
    ((2,0,2),(2,1,2)),
    # 층간 수직 통로
    ((0,0,0),(0,0,1)), ((2,0,0),(2,0,1)),
    ((1,1,0),(1,1,1)),
    ((0,2,0),(0,2,1)), ((2,2,0),(2,2,1)),
    ((0,0,1),(0,0,2)), ((2,1,1),(2,1,2)),
    ((1,2,1),(1,2,2)),
    ((0,2,1),(0,2,2)), ((2,2,1),(2,2,2)),
]
edges = [(idx(*a), idx(*b)) for a, b in raw_edges]

# ─────────────────────────────────────────────────────────────────────────────
# 2. 그래프 단순화: 차수 2 + 직선 경로 노드 제거
# ─────────────────────────────────────────────────────────────────────────────
def is_collinear(v, u, w):
    """
    노드 v와 그 두 이웃 u, w가 3D에서 직선을 이루는지 검사.
    격자 좌표에서는 u + w == 2v 이면 v가 u-w의 중점 = 직선 위.
    예)  u=(0,0,0), v=(1,0,0), w=(2,0,0)  → u+w=(2,0,0)=2v  ✓ 직선
         u=(0,0,0), v=(1,0,0), w=(1,1,0)  → u+w=(1,1,0)≠2v  ✗ 꺾임
    """
    vx, vy, vz = coord(v)
    ux, uy, uz = coord(u)
    wx, wy, wz = coord(w)
    return (ux + wx == 2*vx) and (uy + wy == 2*vy) and (uz + wz == 2*vz)

def simplify_graph(edges, keep_nodes):
    """
    차수 2이고 두 이웃이 직선상에 있는 노드를 반복적으로 제거.
    한 번 노드를 제거해 새 간선을 만들면, 이 간선의 양 끝 노드가
    다시 '차수 2 + 직선' 조건을 만족할 수 있으므로 while로 반복한다.

    keep_nodes: 절대 제거하지 않을 노드 집합 (입구, 출구)
    반환: 축약된 (edges, 남은 노드 집합, 제거 로그)
    """
    # 인접 집합으로 변환 (중복/자기 자신 방지 위해 set 사용)
    adj = defaultdict(set)
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)

    active = set(adj.keys())
    removed_log = []

    changed = True
    while changed:
        changed = False
        # 반복 중에 dict를 수정하므로 스냅샷을 뜬다
        for v in list(active):
            if v in keep_nodes:
                continue
            if len(adj[v]) != 2:
                continue
            u, w = tuple(adj[v])
            if u == w:                     # 자기 루프 방어
                continue
            if not is_collinear(v, u, w):
                continue

            # v 제거하고 u-w 직접 연결
            adj[u].discard(v)
            adj[w].discard(v)
            adj[u].add(w)
            adj[w].add(u)
            del adj[v]
            active.discard(v)
            removed_log.append((v, u, w))
            changed = True

    # 새 간선 리스트 (중복 없이)
    new_edges = set()
    for a in adj:
        for b in adj[a]:
            if a < b:
                new_edges.add((a, b))

    return sorted(new_edges), sorted(active), removed_log

# ─────────────────────────────────────────────────────────────────────────────
# 3. 노드 인덱스 리매핑
# ─────────────────────────────────────────────────────────────────────────────
def remap_nodes(edges, active_nodes):
    """
    축약 후 남은 노드들을 0, 1, 2, ... 로 다시 번호 매김.
    행렬 크기를 남은 노드 수에 맞추기 위해서 필요하다.
    반환: (새 간선 리스트, old→new 인덱스 매핑)
    """
    old_to_new = {old: new for new, old in enumerate(active_nodes)}
    new_edges = [(old_to_new[a], old_to_new[b]) for a, b in edges]
    return new_edges, old_to_new

# ─────────────────────────────────────────────────────────────────────────────
# 4. 논문 식 구현 (앞 코드와 동일)
# ─────────────────────────────────────────────────────────────────────────────
def build_W(n, edges):
    """이진 가중치 행렬"""
    W = np.zeros((n, n))
    for a, b in edges:
        W[a, b] = W[b, a] = 1.0
    return W

def compute_degree(W):
    """식 (1)(2): 차수와 평균 차수"""
    k = W.sum(axis=1)
    return k, k.mean()

def build_transition_matrix(W, output_node):
    """식 (9)(10): 전이 확률 행렬"""
    n = W.shape[0]
    s = W.sum(axis=1)                          # 식 (10)
    P = np.zeros((n, n))
    for i in range(n):
        if i == output_node:
            P[i, i] = 1.0                      # 흡수 상태
        elif s[i] > 0:
            P[i] = W[i] / s[i]                 # 식 (9)
    return P, s

def extract_theta(P, output_node):
    """정준형에서 Θ 블록 분리"""
    transient = [i for i in range(P.shape[0]) if i != output_node]
    Theta = P[np.ix_(transient, transient)]
    return Theta, transient

def compute_fundamental_matrix(Theta):
    """식 (13): Ψ = (I - Θ)^(-1)"""
    n = Theta.shape[0]
    return np.linalg.inv(np.eye(n) - Theta)

def compute_absorption_time(Psi):
    """식 (19)(20): τ_i = Σ_j Ψ_ij"""
    return Psi.sum(axis=1)

# ─────────────────────────────────────────────────────────────────────────────
# 5. 흡수 시간 계산을 한 함수로 묶음
# ─────────────────────────────────────────────────────────────────────────────
def absorption_time_pipeline(n, edges, input_node, output_node, label=""):
    """
    주어진 그래프에서 논문 식 (9)~(20)을 순서대로 실행해
    입구 노드의 흡수 시간을 반환.
    """
    W = build_W(n, edges)
    _, k_avg = compute_degree(W)
    P, _ = build_transition_matrix(W, output_node)
    Theta, transient = extract_theta(P, output_node)
    rho = np.abs(np.linalg.eigvals(Theta)).max()
    Psi = compute_fundamental_matrix(Theta)
    tau = compute_absorption_time(Psi)
    tau_input = tau[transient.index(input_node)]

    print(f"\n[{label}]")
    print(f"  노드 수: {n},  간선 수: {len(edges)}")
    print(f"  평균 차수 k = {k_avg:.4f}")
    print(f"  Θ 스펙트럼 반경 ρ(Θ) = {rho:.6f}")
    print(f"  ★ 입구 흡수 시간 τ = {tau_input:.4f}")
    return tau_input

# ─────────────────────────────────────────────────────────────────────────────
# 6. 실행
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("3×3×3 미로 흡수 시간 계산 (그래프 단순화 비교)")
    print("=" * 68)

    # ── (A) 원본 그래프 ─────────────────────────────────────────────
    tau_orig = absorption_time_pipeline(
        n=N, edges=edges,
        input_node=INPUT_NODE, output_node=OUTPUT_NODE,
        label="원본 그래프 (27 노드)"
    )

    # ── (B) 단순화된 그래프 ────────────────────────────────────────
    keep = {INPUT_NODE, OUTPUT_NODE}
    reduced_edges_old, active_nodes, removed_log = simplify_graph(edges, keep)

    print(f"\n[단순화] 제거된 노드 {len(removed_log)}개:")
    for v, u, w in removed_log:
        print(f"  {coord(v)}  ({coord(u)} — {coord(w)} 사이 직선 통과)")

    print(f"\n남은 노드 {len(active_nodes)}개:")
    for old_i in active_nodes:
        x, y, z = coord(old_i)
        tag = " ← 입구" if old_i == INPUT_NODE else \
              (" ← 출구" if old_i == OUTPUT_NODE else "")
        print(f"  ({x},{y},{z}){tag}")

    # 새 인덱스로 리매핑
    new_edges, mapping = remap_nodes(reduced_edges_old, active_nodes)
    new_input  = mapping[INPUT_NODE]
    new_output = mapping[OUTPUT_NODE]
    n_new = len(active_nodes)

    tau_reduced = absorption_time_pipeline(
        n=n_new, edges=new_edges,
        input_node=new_input, output_node=new_output,
        label=f"단순화 그래프 ({n_new} 노드)"
    )

    # ── 비교 ────────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("비교")
    print("=" * 68)
    print(f"  원본     : τ = {tau_orig:.4f}   (27 노드, {len(edges)} 간선)")
    print(f"  단순화   : τ = {tau_reduced:.4f}   ({n_new} 노드, {len(new_edges)} 간선)")
    print()
    print("  참고: 단순화하면 통과점 스텝이 사라져 τ가 작아진다.")
    print("        절대값 비교보다는 여러 미로 간 상대적 순위 비교에 사용하는 것이 좋다.")

if __name__ == "__main__":
    main()