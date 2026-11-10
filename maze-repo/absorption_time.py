"""
3×3×3 미로 흡수 시간 계산
Amancio et al., Physica A 390 (2011) 4673-4683

논문 식 순서:
  (1)(2)  : 노드 차수 k_i, 평균 차수 k
  (9)(10) : 전이 확률 p_ij = W_ij / s_i,  s_i = Σ W_ij
  정준형  : Π = [[Θ, Φ], [0, I]]  (일시/흡수 분리)
  (13)    : Ψ = (I - Θ)^{-1}
  (19)(20): τ = Ψu,  τ_i = Σ_j Ψ_ij
"""

import numpy as np

# ── 0. 노드 정의 ───────────────────────────────────────────────────────────────
# 노드 인덱스: (x, y, z) → i = x + 3*y + 9*z  (x,y,z ∈ {0,1,2})
def idx(x, y, z): return x + 3*y + 9*z
def coord(i):     z, r = divmod(i, 9); y, x = divmod(r, 3); return x, y, z

N = 27
INPUT_NODE  = idx(0, 0, 0)   # 입구
OUTPUT_NODE = idx(2, 2, 2)   # 출구

# ── 1. 통로(에지) 정의 ────────────────────────────────────────────────────────
# 그래프 시각화에서 정의한 것과 동일한 미로 구조
raw_edges = [
    # Z=0 수평 통로
    ((0,0,0),(1,0,0)), ((1,0,0),(2,0,0)),
    ((0,1,0),(1,1,0)),
    ((0,2,0),(1,2,0)), ((1,2,0),(2,2,0)),
    ((0,0,0),(0,1,0)),
    ((1,0,0),(1,1,0)), ((1,1,0),(1,2,0)),
    ((2,0,0),(2,1,0)), ((2,1,0),(2,2,0)),
    # Z=1 수평 통로
    ((0,0,1),(1,0,1)),
    ((0,1,1),(1,1,1)), ((1,1,1),(2,1,1)),
    ((0,2,1),(1,2,1)),
    ((0,0,1),(0,1,1)), ((0,1,1),(0,2,1)),
    ((1,0,1),(1,1,1)),
    ((2,0,1),(2,1,1)), ((2,1,1),(2,2,1)),
    # Z=2 수평 통로
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

# ── 2. 가중치 행렬 W 구성 ─────────────────────────────────────────────────────
# 여기서는 세 가지 가중치를 모두 계산해 비교합니다.

def build_W_binary(N, edges):
    """가중치 방법 A: 이진 (W_ij = 1)"""
    W = np.zeros((N, N))
    for a, b in edges:
        W[a, b] = W[b, a] = 1.0
    return W

def build_W_inv_degree(N, edges):
    """가중치 방법 B: W_ij = 1/k_j  (차수가 작은 노드로 더 끌린다)"""
    # 먼저 이진으로 차수 계산
    W0 = build_W_binary(N, edges)
    k = W0.sum(axis=1)  # 각 노드의 차수 (식 1 적용)
    W = np.zeros((N, N))
    for a, b in edges:
        if k[b] > 0: W[a, b] = 1.0 / k[b]
        if k[a] > 0: W[b, a] = 1.0 / k[a]
    return W

def build_W_inv_dist(N, edges):
    """가중치 방법 C: W_ij = 1/d_ij  (수직 통로는 거리 패널티 α=1.5 적용)"""
    W = np.zeros((N, N))
    for (xa,ya,za),(xb,yb,zb) in raw_edges:
        a, b = idx(xa,ya,za), idx(xb,yb,zb)
        is_vertical = (xa==xb and ya==yb)
        d = 1.5 if is_vertical else 1.0   # 수직 이동은 1.5배 더 힘들다
        W[a, b] = W[b, a] = 1.0 / d
    return W

# ── 3. 논문 식 (9)(10): 전이 확률 행렬 P ────────────────────────────────────
def build_transition_matrix(W, output_node):
    """
    식 (10): s_i = Σ_j W_ij  (노드 강도)
    식  (9): p_ij = W_ij / s_i  (출구가 아닌 노드)
    출구 노드: p_ii = 1  (흡수 상태)
    """
    N = W.shape[0]
    s = W.sum(axis=1)               # 식 (10)
    P = np.zeros((N, N))
    for i in range(N):
        if i == output_node:
            P[i, i] = 1.0           # 흡수 상태
        else:
            if s[i] > 0:
                P[i] = W[i] / s[i]  # 식 (9)
    return P, s

# ── 4. 정준형: Θ (일시→일시) 분리 ───────────────────────────────────────────
def extract_theta(P, output_node):
    """
    정준형 Π = [[Θ, Φ], [0, I]] 에서 Θ 추출
    일시 노드 = 출구가 아닌 모든 노드
    """
    transient = [i for i in range(P.shape[0]) if i != output_node]
    Theta = P[np.ix_(transient, transient)]
    return Theta, transient

# ── 5. 논문 식 (13): 기본 행렬 Ψ = (I - Θ)^{-1} ────────────────────────────
def compute_fundamental_matrix(Theta):
    """
    식 (13): Ψ = Σ_{k=0}^{∞} Θ^k = (I - Θ)^{-1}
    Θ^{n+1} → 0  (n→∞) 이므로 역행렬로 직접 계산
    """
    n = Theta.shape[0]
    Psi = np.linalg.inv(np.eye(n) - Theta)  # 식 (13)
    return Psi

# ── 6. 논문 식 (19)(20): 흡수 시간 τ ────────────────────────────────────────
def compute_absorption_time(Psi):
    """
    식 (19): τ = Ψu   (u: 모든 원소가 1인 열벡터)
    식 (20): τ_i = Σ_j Ψ_ij  (행의 합)
    """
    return Psi.sum(axis=1)   # 식 (20)

# ── 7. 식 (1)(2): 차수와 평균 차수 ──────────────────────────────────────────
def compute_degree(W):
    """식 (1): k_i = Σ_j W_ij,  식 (2): k = (1/N) Σ k_i"""
    k = (W > 0).astype(float).sum(axis=1)  # 이진화 후 차수
    return k, k.mean()

# ── 8. 전체 실행 및 출력 ─────────────────────────────────────────────────────
def run(label, W):
    print(f"\n{'='*60}")
    print(f"가중치 방법: {label}")
    print(f"{'='*60}")

    # 차수 (식 1, 2)
    k, k_avg = compute_degree(W)
    print(f"\n[식 (1)(2)] 노드 차수 k_i 및 평균 차수 k")
    for i in range(N):
        x,y,z = coord(i)
        tag = " ← 입구" if i==INPUT_NODE else (" ← 출구" if i==OUTPUT_NODE else "")
        print(f"  ({x},{y},{z})  k={int(k[i])}{tag}")
    print(f"  평균 차수 k = {k_avg:.4f}")

    # 전이 확률 행렬 (식 9, 10)
    P, s = build_transition_matrix(W, OUTPUT_NODE)
    print(f"\n[식 (9)(10)] 노드 강도 s_i = Σ W_ij")
    for i in range(N):
        x,y,z = coord(i)
        print(f"  ({x},{y},{z})  s={s[i]:.4f}")

    # Θ 추출 (정준형)
    Theta, transient = extract_theta(P, OUTPUT_NODE)
    print(f"\n[정준형] 일시 노드 수: {len(transient)}개  (출구 제외)")
    print(f"  Θ shape: {Theta.shape}")

    # Ψ = (I-Θ)^{-1} (식 13)
    Psi = compute_fundamental_matrix(Theta)
    print(f"\n[식 (13)] Ψ = (I-Θ)^{{-1}}  shape: {Psi.shape}")

    # 검증: Θ의 스펙트럼 반경이 1 미만인지 확인
    eigvals = np.abs(np.linalg.eigvals(Theta))
    rho = eigvals.max()
    print(f"  Θ 스펙트럼 반경 ρ(Θ) = {rho:.6f}  (< 1 이어야 수렴 보장)")

    # 흡수 시간 τ (식 19, 20)
    tau = compute_absorption_time(Psi)
    print(f"\n[식 (19)(20)] 흡수 시간 τ_i (입구에서 출구까지 기대 스텝 수)")
    for local_i, global_i in enumerate(transient):
        x,y,z = coord(global_i)
        tag = " ← 입구" if global_i==INPUT_NODE else ""
        print(f"  ({x},{y},{z})  τ = {tau[local_i]:.2f}{tag}")

    # 입구 흡수 시간 강조
    inp_local = transient.index(INPUT_NODE)
    print(f"\n★ 입구 (0,0,0)에서 출구 (2,2,2)까지 기대 흡수 시간: {tau[inp_local]:.4f} 스텝")
    return tau, transient

# ── 실행 ─────────────────────────────────────────────────────────────────────
print("3×3×3 미로 흡수 시간 계산")
print("Amancio et al., Physica A 390 (2011) 식 (9)→(10)→(13)→(20) 적용")

W_A = build_W_binary(N, edges)
W_B = build_W_inv_degree(N, edges)
W_C = build_W_inv_dist(N, edges)

tau_A, trans_A = run("A: 이진 (W_ij = 1)",         W_A)
tau_B, trans_B = run("B: 역차수 (W_ij = 1/k_j)",   W_B)
tau_C, trans_C = run("C: 역거리 (수직 패널티 1.5)", W_C)

# ── 비교 요약 ─────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("요약: 입구 (0,0,0) 흡수 시간 비교")
print(f"{'='*60}")
for label, tau, trans in [
    ("A 이진",    tau_A, trans_A),
    ("B 역차수",  tau_B, trans_B),
    ("C 역거리",  tau_C, trans_C),
]:
    t = tau[trans.index(INPUT_NODE)]
    print(f"  {label:12s}: {t:.4f} 스텝")

print(f"\n※ 흡수 시간이 클수록 이 가중치 체계 하에서 미로가 더 복잡함을 의미합니다.")
