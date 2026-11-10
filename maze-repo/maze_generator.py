"""
==============================================================================
무작위 미로 대량 생성 + 전처리
  - 생성 : 2D(n×n×1) / 3D(n×n×n) 격자 미로, 생성 방식 3가지, 순환(고리) 옵션
  - 전처리: maze_complexity.py 의 [2] 미로 해석 → [3] 직선 정점 삭제
  - 저장 : 미로마다 한 줄씩 JSON (maze_dataset/*.jsonl) + 요약표 summary.csv

  실행 : python maze_generator.py            (기본 설정으로 데이터셋 생성)
  읽기 : from maze_generator import load_mazes
         mazes = load_mazes("maze_dataset", dimension=3, size=5)
==============================================================================
"""

import csv
import json
import os
import random
from collections import deque

from maze_complexity import (read_maze, remove_straight_vertices, is_straight,
                             count_axes, maze_complexity)


# =============================================================================
# [A] 격자 도구
# =============================================================================
def all_cells(size):
    X, Y, Z = size
    return [(x, y, z) for z in range(Z) for y in range(Y) for x in range(X)]


def grid_neighbors(c, size):
    """격자 범위 안의 이웃 칸 (3D면 최대 6개, 2D(Z=1)면 최대 4개)."""
    out = []
    for k in range(3):
        for d in (-1, 1):
            n = list(c); n[k] += d
            if 0 <= n[k] < size[k]:
                out.append(tuple(n))
    return out


def all_grid_walls(size):
    """이웃한 두 칸 사이의 모든 벽 (뚫으면 통로가 됨)."""
    walls = []
    for c in all_cells(size):
        for n in grid_neighbors(c, size):
            if c < n:
                walls.append((c, n))
    return walls


# =============================================================================
# [B] 생성 방식 3가지  (모두 '완전 미로' = 순환 없이 모든 칸이 연결된 나무 구조)
# =============================================================================
def gen_dfs(size, rng):
    """깊이우선(되돌아가기) : 길고 구불구불한 통로, 갈림길이 적음."""
    start = (0, 0, 0)
    seen, stack, passages = {start}, [start], []
    while stack:
        c = stack[-1]
        cand = [n for n in grid_neighbors(c, size) if n not in seen]
        if cand:
            n = rng.choice(cand)
            passages.append((c, n))
            seen.add(n); stack.append(n)
        else:
            stack.pop()
    return passages


def gen_prim(size, rng):
    """프림 : 짧은 막다른 길이 많고 갈림길이 많음."""
    start = (0, 0, 0)
    seen, passages = {start}, []
    frontier = [(start, n) for n in grid_neighbors(start, size)]
    while frontier:
        i = rng.randrange(len(frontier))
        frontier[i], frontier[-1] = frontier[-1], frontier[i]
        c, n = frontier.pop()
        if n in seen:
            continue
        passages.append((c, n))
        seen.add(n)
        frontier.extend((n, m) for m in grid_neighbors(n, size) if m not in seen)
    return passages


def gen_kruskal(size, rng):
    """크루스칼 : 벽을 무작위 순서로 뚫되 고리가 생기면 건너뜀. 두 방식의 중간 성격."""
    parent = {c: c for c in all_cells(size)}

    def find(c):
        while parent[c] != c:
            parent[c] = parent[parent[c]]
            c = parent[c]
        return c

    walls = all_grid_walls(size)
    rng.shuffle(walls)
    passages = []
    for a, b in walls:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
            passages.append((a, b))
    return passages


GENERATORS = {"dfs": gen_dfs, "prim": gen_prim, "kruskal": gen_kruskal}


def add_loops(passages, size, loop_ratio, rng):
    """남은 벽 중 일부를 더 뚫어 고리(여러 갈래 길)를 만든다. 개수 = loop_ratio × 칸 수."""
    if loop_ratio <= 0:
        return passages
    have = {tuple(sorted(p)) for p in passages}
    rest = [w for w in all_grid_walls(size) if w not in have]
    k = min(len(rest), round(loop_ratio * len(all_cells(size))))
    return passages + rng.sample(rest, k)


# =============================================================================
# [C] 미로 하나 만들기  (maze_complexity.py 의 MAZE 형식과 같음)
# =============================================================================
def make_maze(size, algorithm="dfs", loop_ratio=0.0, seed=0):
    rng = random.Random(seed)
    passages = GENERATORS[algorithm](size, rng)
    passages = add_loops(passages, size, loop_ratio, rng)
    return {
        "entry": (0, 0, 0),
        "exit": (size[0] - 1, size[1] - 1, size[2] - 1),   # 대각선 반대쪽 모서리
        "edges": [(a, b, 1) for a, b in passages],
    }


# =============================================================================
# [D] 전처리 + 구조 통계
# =============================================================================
def shortest_path_steps(graph, entry, exit_):
    """입구→출구 최단 경로의 칸 이동 수 (원본 그래프 기준)."""
    dist, q = {entry: 0}, deque([entry])
    while q:
        v = q.popleft()
        if v == exit_:
            return dist[v]
        for u in graph[v]:
            if u not in dist:
                dist[u] = dist[v] + 1
                q.append(u)
    return None


def preprocess(maze):
    """[2] 해석 → [3] 직선 정점 삭제 를 하고, 분석에 쓸 구조 통계를 함께 계산."""
    graph, entry, exit_ = read_maze(maze)
    g, removed = remove_straight_vertices(graph, keep={entry, exit_})

    deg = {v: len(g[v]) for v in g}
    stats = {
        "cells": len(graph),
        "passages": sum(len(n) for n in graph.values()) // 2,
        "vertices": len(g),
        "edges": sum(len(n) for n in g.values()) // 2,
        "removed_straight": len(removed),
        "dead_ends": sum(d == 1 for d in deg.values()),
        "turns": sum(d == 2 and not is_straight(g, v) for v, d in deg.items()),
        "branches": sum(d >= 3 for d in deg.values()),
        "three_axis_vertices": sum(count_axes(g, v) == 3 for v in g),
        "loops": sum(len(n) for n in g.values()) // 2 - len(g) + 1,  # 독립 고리 수
        "shortest_path": shortest_path_steps(graph, entry, exit_),
    }
    simplified = {
        "vertices": sorted(g),
        "edges": sorted((a, b, L) for a in g for b, L in g[a].items() if a < b),
    }
    return simplified, stats


def complexity_columns(maze):
    """전처리가 끝까지 잘 되는지 확인 겸, 비교용 복잡도 값들."""
    r = maze_complexity(maze, visit_weight="a-1")
    r0 = maze_complexity(maze, visit_weight="none")
    tL, tD = r["tau_length"], r["tau_degree"]
    return {
        "tau_length": tL, "tau_degree": tD,
        "C_geometric": (tL * tD) ** 0.5,
        "C_arithmetic": (tL + tD) / 2,
        "C_harmonic": 2 * tL * tD / (tL + tD),
        "C_geometric_no_axis": r0["complexity"],
    }


# =============================================================================
# [E] 데이터셋 설정  (2D와 3D는 칸 수가 비슷한 크기끼리 짝지음)
# =============================================================================
#   2D 5×5 (25)  ↔ 3D 3³ (27)      2D 8×8 (64)   ↔ 3D 4³ (64)
#   2D 11×11 (121) ↔ 3D 5³ (125)   2D 15×15 (225) ↔ 3D 6³ (216)
#   2D 19×19 (361) ↔ 3D 7³ (343)
SIZES_2D = [5, 8, 11, 15, 19]
SIZES_3D = [3, 4, 5, 6, 7]
ALGORITHMS = ["dfs", "prim", "kruskal"]
LOOP_RATIOS = [0.0, 0.1]          # 0 = 완전 미로(나무), 0.1 = 칸 수의 10%만큼 벽을 더 뚫음
MAZES_PER_SETTING = 100


def settings():
    for n in SIZES_2D:
        yield 2, (n, n, 1)
    for n in SIZES_3D:
        yield 3, (n, n, n)


def build_dataset(out_dir="maze_dataset", per_setting=MAZES_PER_SETTING,
                  with_complexity=True):
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for dim, size in settings():
        for alg in ALGORITHMS:
            for lr in LOOP_RATIOS:
                tag = f"{dim}d_n{size[0]:02d}_{alg}_loop{int(lr * 100):02d}"
                path = os.path.join(out_dir, f"{tag}.jsonl")
                with open(path, "w", encoding="utf-8") as f:
                    for k in range(per_setting):
                        # 설정마다 겹치지 않는 고정 시드 → 언제 다시 돌려도 같은 미로
                        seed = (dim * 10**7 + size[0] * 10**5
                                + ALGORITHMS.index(alg) * 10**4
                                + int(lr * 100) * 10**3 + k)
                        maze = make_maze(size, alg, lr, seed)
                        simplified, stats = preprocess(maze)
                        rec = {
                            "id": f"{tag}_{k:03d}",
                            "dimension": dim, "size": size, "algorithm": alg,
                            "loop_ratio": lr, "seed": seed,
                            **maze, "simplified": simplified, "stats": stats,
                        }
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

                        row = {"id": rec["id"], "dimension": dim, "size": size[0],
                               "algorithm": alg, "loop_ratio": lr, "seed": seed,
                               **stats}
                        if with_complexity:
                            row.update(complexity_columns(maze))
                        rows.append(row)
                print(f"  {tag}: {per_setting}개")

    with open(os.path.join(out_dir, "summary.csv"), "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


# =============================================================================
# [F] 저장된 미로 불러오기
# =============================================================================
def _tuplify(rec):
    rec["entry"], rec["exit"] = tuple(rec["entry"]), tuple(rec["exit"])
    rec["edges"] = [(tuple(a), tuple(b), L) for a, b, L in rec["edges"]]
    s = rec["simplified"]
    s["vertices"] = [tuple(v) for v in s["vertices"]]
    s["edges"] = [(tuple(a), tuple(b), L) for a, b, L in s["edges"]]
    return rec


def load_mazes(out_dir="maze_dataset", dimension=None, size=None,
               algorithm=None, loop_ratio=None):
    """조건에 맞는 미로 목록. 각 미로는 maze_complexity() 에 바로 넣을 수 있음."""
    out = []
    for name in sorted(os.listdir(out_dir)):
        if not name.endswith(".jsonl"):
            continue
        with open(os.path.join(out_dir, name), encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if dimension is not None and rec["dimension"] != dimension: continue
                if size is not None and rec["size"][0] != size: continue
                if algorithm is not None and rec["algorithm"] != algorithm: continue
                if loop_ratio is not None and rec["loop_ratio"] != loop_ratio: continue
                out.append(_tuplify(rec))
    return out


if __name__ == "__main__":
    print("무작위 미로 데이터셋 생성")
    rows = build_dataset()
    print(f"총 {len(rows)}개 생성 완료 → maze_dataset/")
