#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import numpy as np
import yaml
import matplotlib.pyplot as plt
from PIL import Image

# shapely는 경계 내부 판정에 사용 (6컬럼 입력일 때)
from shapely.geometry import Polygon, LinearRing, Point
from shapely.geometry import LineString

# ---------------- 사용자 파라미터 ----------------
WAYPOINT_FILE = os.environ.get("outputs", "/root/KORA_K3/src/kora_k3/src/path_planning/outputs/waypoints.npy") # .npy 또는 .py
XI_ITERATIONS = int(os.environ.get("XI_ITERATIONS", "8"))         # 점별 이분 탐색 반복
LINE_ITERATIONS = int(os.environ.get("LINE_ITERATIONS", "500"))   # 전체 스캔 반복
# 시각화용 맵 파일 (환경변수로 재정의 가능)
_MAPS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "maps"))
DEFAULT_MAP_BASENAME = "real"
MAP_YAML = os.environ.get("MAP_YAML", os.path.join(_MAPS_DIR, f"{DEFAULT_MAP_BASENAME}.yaml"))
MAP_PGM = os.environ.get("MAP_PGM", os.path.join(_MAPS_DIR, f"{DEFAULT_MAP_BASENAME}.pgm"))
OUTPUT_DIR = os.environ.get(
    "OUTPUT_DIR",
    os.path.join(os.path.dirname(__file__), "outputs"),
)
# -------------------------------------------------

def load_waypoints(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        wp = np.load(path)
    elif ext == ".py":
        with open(path, "r") as f:
            code = f.read()
        # 신뢰 가능한 파일만 사용 (eval 주의)
        from numpy import array  # array(...) 인식용
        wp = eval(code)
    else:
        raise ValueError("지원하지 않는 파일 확장자: {}".format(ext))
    wp = np.asarray(wp, dtype=float)

    # 폐곡선 중복점 제거
    if len(wp) >= 2 and np.allclose(wp[0], wp[-1]):
        wp = wp[:-1]
    return wp

# 멘거 곡률(안정화 버전)
def menger_curvature_safe(p1, p2, p3, atol=1e-9):
    p1, p2, p3 = np.asarray(p1), np.asarray(p2), np.asarray(p3)
    v21 = p1 - p2
    v23 = p3 - p2
    n21 = np.linalg.norm(v21)
    n23 = np.linalg.norm(v23)
    if n21 < atol or n23 < atol:
        return 0.0
    cosv = np.dot(v21, v23) / (n21 * n23)
    cosv = np.clip(cosv, -1.0, 1.0)
    theta = math.acos(cosv)
    # 일직선(π) 보정
    if math.isclose(theta, math.pi, rel_tol=0.0, abs_tol=1e-9):
        theta = 0.0
    d13 = np.linalg.norm(p1 - p3)
    if d13 < atol:
        return 0.0
    return 2.0 * math.sin(theta) / d13

def improve_race_line(old_line, inner_border, outer_border, xi_iters=8):
    """
    K1999에서 영감: 각 점의 곡률을 이웃 평균 곡률로 맞추며 트랙(outer-홀=inner) 영역 내에서만 이동.
    이동은 prev-nexxt 중점 방향의 선분에서 이분 탐색으로 제한.
    """
    new_line = np.array(old_line, dtype=float, copy=True)

    # 도로 폴리곤 (outer가 외곽, inner가 구멍)
    inner_ring = LinearRing(inner_border)
    outer_ring = LinearRing(outer_border)
    road_poly = Polygon(outer_ring, holes=[inner_ring])

    n = len(new_line)
    for i in range(n):
        prevprev = (i - 2) % n
        prev = (i - 1) % n
        nexxt = (i + 1) % n
        nexxtnexxt = (i + 2) % n

        xi = tuple(new_line[i])
        c_i = menger_curvature_safe(new_line[prev], xi, new_line[nexxt])
        c1 = menger_curvature_safe(new_line[prevprev], new_line[prev], xi)
        c2 = menger_curvature_safe(xi, new_line[nexxt], new_line[nexxtnexxt])
        target_c = 0.5 * (c1 + c2)

        # 이분 탐색 경계: 현위치와 이웃의 중점
        b1 = tuple(xi)
        b2 = ((new_line[nexxt][0] + new_line[prev][0]) * 0.5,
              (new_line[nexxt][1] + new_line[prev][1]) * 0.5)
        p = tuple(xi)

        for _ in range(xi_iters):
            p_c = menger_curvature_safe(new_line[prev], p, new_line[nexxt])
            if math.isclose(p_c, target_c, rel_tol=1e-3, abs_tol=1e-4):
                break

            if p_c < target_c:
                # 곡률이 모자람 → 더 굽히도록 b2 쪽을 당김
                b2 = p
                new_p = ((b1[0] + p[0]) * 0.5, (b1[1] + p[1]) * 0.5)
                if not Point(new_p).within(road_poly):
                    b1 = new_p
                else:
                    p = new_p
            else:
                # 곡률이 과함 → 펴주도록 b1 쪽을 당김
                b1 = p
                new_p = ((b2[0] + p[0]) * 0.5, (b2[1] + p[1]) * 0.5)
                if not Point(new_p).within(road_poly):
                    b2 = new_p
                else:
                    p = new_p

        new_line[i] = p

    return new_line

def visualize_paths(map_yaml_path, map_pgm_path, raw_xy, optimized_xy):
    """맵 위에 원본·최적화 경로를 나란히 시각화한다."""
    if raw_xy is None or raw_xy.size == 0:
        print("[경고] 시각화할 원본 waypoints 데이터가 없습니다.")
        return
    if optimized_xy is None or optimized_xy.size == 0:
        print("[경고] 시각화할 최적화 경로 데이터가 없습니다.")
        return

    if not (os.path.exists(map_yaml_path) and os.path.exists(map_pgm_path)):
        print("[경고] 맵 시각화 파일을 찾지 못했습니다. (yaml/pgm 경로 확인)")
        return

    try:
        with open(map_yaml_path, "r") as f:
            meta = yaml.safe_load(f)

        resolution = meta.get("resolution")
        origin = meta.get("origin", [0.0, 0.0, 0.0])
        if resolution is None or len(origin) < 2:
            print("[경고] 맵 메타데이터에 resolution 또는 origin 정보가 없습니다.")
            return

        map_img = Image.open(map_pgm_path)
        width, height = map_img.size

        def to_pixels(xy):
            px = (xy[:, 0] - origin[0]) / resolution
            py_world = (xy[:, 1] - origin[1]) / resolution
            py = height - py_world  # 이미지 좌표계는 y가 아래 방향
            return px, py

        fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
        titles = ("Waypoints", "Optimal Waypoints")
        paths = (raw_xy, optimized_xy)
        colors = ("tab:blue", "tab:red")

        for ax, title, path_xy, color in zip(axes, titles, paths, colors):
            px, py = to_pixels(path_xy)
            ax.imshow(map_img, cmap="gray", origin="upper")
            ax.plot(px, py, color=color, linewidth=2)
            ax.set_title(title)
            ax.set_axis_off()

        fig.tight_layout()
        plt.show()
        plt.close(fig)
    except Exception as exc:
        print(f"[경고] 레이싱라인 시각화에 실패했습니다: {exc}")

def main():
    wp = load_waypoints(WAYPOINT_FILE)
    track_name = os.path.splitext(os.path.basename(WAYPOINT_FILE))[0]

    raw_loop = None
    if wp.shape[1] >= 6:
        # [center(xy), inner(xy), outer(xy)]
        center = wp[:, 0:2]
        inner = wp[:, 2:4]
        outer = wp[:, 4:6]

        # 최적화 시작(중복점 없는 열린 라인으로)
        raceline = center.copy()
        if np.allclose(raceline[0], raceline[-1]):
            raceline = raceline[:-1]

        for it in range(LINE_ITERATIONS):
            raceline = improve_race_line(raceline, inner, outer, xi_iters=XI_ITERATIONS)
            print(f"Iteration {it+1}/{LINE_ITERATIONS} complete.", end="\r")

        # 폐곡선으로 닫기
        loop_raceline = np.vstack([raceline, raceline[0]])
        raw_loop = center.copy()
        if not np.allclose(raw_loop[0], raw_loop[-1]):
            raw_loop = np.vstack([raw_loop, raw_loop[0]])

    elif wp.shape[1] == 2:
        # 중심선만 존재: 최적화 생략, 폐곡선만 확정
        raceline = wp.copy()
        if not np.allclose(raceline[0], raceline[-1]):
            loop_raceline = np.vstack([raceline, raceline[0]])
        else:
            loop_raceline = raceline
        raw_loop = loop_raceline.copy()
        print("[경고] 2컬럼 입력입니다. inner/outer 경계가 없어 최적화는 생략했습니다.")
    else:
        raise ValueError("입력 배열의 열 수가 예상과 다릅니다. (2 또는 6 컬럼 필요)")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    visualize_paths(MAP_YAML, MAP_PGM, raw_loop, loop_raceline)

    opt_csv = os.path.join(OUTPUT_DIR, "optimal_waypoints.csv")
    np.savetxt(opt_csv, loop_raceline, delimiter=",", fmt="%.6f")
    print(f"[save] optimal waypoint CSV: {opt_csv}")

    out_npy = os.path.join(OUTPUT_DIR, f"optimal_{track_name}.npy")
    np.save(out_npy, loop_raceline)
    print(f"[complete] save: {out_npy}  (points={loop_raceline.shape[0]})")

    # 길이 정보(참고)
    try:
        L_center = LineString(wp[:, 0:2]).length if wp.shape[1] >= 6 else LineString(loop_raceline).length
        L_race = LineString(loop_raceline).length
        print(f"Center length: {L_center:.2f}, Raceline length: {L_race:.2f}")
    except Exception:
        pass

if __name__ == "__main__":
    main()
