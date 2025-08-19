# Algorithm Visual Playground — Streamlit app
# Author: ChatGPT
# Run: streamlit run app.py

import time
import random
from collections import deque
import heapq
from dataclasses import dataclass
from typing import Generator, List, Tuple, Dict, Optional, Set

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# =========================
# Utility & State Helpers
# =========================

def init_session_state():
    ss = st.session_state
    ss.setdefault("tab", "Sorting")

    # Sorting state
    ss.setdefault("arr", None)
    ss.setdefault("arr_size", 30)
    ss.setdefault("sort_algo", "Bubble Sort")
    ss.setdefault("sort_algo_right", "Insertion Sort")
    ss.setdefault("compare_mode", False)
    ss.setdefault("sort_gen_left", None)
    ss.setdefault("sort_gen_right", None)
    ss.setdefault("sort_running", False)
    ss.setdefault("sort_speed_ms", 100)
    ss.setdefault("sort_state_left", None)
    ss.setdefault("sort_state_right", None)

    # Pathfinding state
    ss.setdefault("grid_size", 20)
    ss.setdefault("wall_density", 0.2)
    ss.setdefault("grid", None)
    ss.setdefault("start", None)
    ss.setdefault("goal", None)
    ss.setdefault("path_algo", "BFS")
    ss.setdefault("path_gen", None)
    ss.setdefault("path_running", False)
    ss.setdefault("path_speed_ms", 60)
    ss.setdefault("path_state", None)


# =========================
# Sorting Algorithms (Generators)
# Each yields dict: {"arr": list[int], "highlight": (i,j), "action": str}
# =========================

def bubble_sort(arr: List[int]) -> Generator[Dict, None, None]:
    a = arr[:]
    n = len(a)
    for i in range(n):
        for j in range(0, n - i - 1):
            yield {"arr": a[:], "highlight": (j, j+1), "action": "compare"}
            if a[j] > a[j+1]:
                a[j], a[j+1] = a[j+1], a[j]
                yield {"arr": a[:], "highlight": (j, j+1), "action": "swap"}
    yield {"arr": a[:], "highlight": None, "action": "done"}


def insertion_sort(arr: List[int]) -> Generator[Dict, None, None]:
    a = arr[:]
    for i in range(1, len(a)):
        key = a[i]
        j = i - 1
        while j >= 0 and a[j] > key:
            yield {"arr": a[:], "highlight": (j, j+1), "action": "compare"}
            a[j+1] = a[j]
            j -= 1
            yield {"arr": a[:], "highlight": (j, j+1 if j+1 < len(a) else None), "action": "shift"}
        a[j+1] = key
        yield {"arr": a[:], "highlight": (j+1, i), "action": "insert"}
    yield {"arr": a[:], "highlight": None, "action": "done"}


def selection_sort(arr: List[int]) -> Generator[Dict, None, None]:
    a = arr[:]
    n = len(a)
    for i in range(n):
        min_idx = i
        for j in range(i+1, n):
            yield {"arr": a[:], "highlight": (min_idx, j), "action": "compare"}
            if a[j] < a[min_idx]:
                min_idx = j
                yield {"arr": a[:], "highlight": (i, min_idx), "action": "new_min"}
        a[i], a[min_idx] = a[min_idx], a[i]
        yield {"arr": a[:], "highlight": (i, min_idx), "action": "swap"}
    yield {"arr": a[:], "highlight": None, "action": "done"}


def merge_sort(arr: List[int]) -> Generator[Dict, None, None]:
    a = arr[:]
    states = []

    def merge(l, m, r):
        left = a[l:m]
        right = a[m:r]
        i = j = 0
        k = l
        while i < len(left) and j < len(right):
            states.append({"arr": a[:], "highlight": (l+i, m+j), "action": "compare"})
            if left[i] <= right[j]:
                a[k] = left[i]
                i += 1
            else:
                a[k] = right[j]
                j += 1
            k += 1
            states.append({"arr": a[:], "highlight": (k-1, k-1), "action": "write"})
        while i < len(left):
            a[k] = left[i]
            i += 1
            k += 1
            states.append({"arr": a[:], "highlight": (k-1, k-1), "action": "write"})
        while j < len(right):
            a[k] = right[j]
            j += 1
            k += 1
            states.append({"arr": a[:], "highlight": (k-1, k-1), "action": "write"})

    size = 1
    n = len(a)
    while size < n:
        for l in range(0, n, 2*size):
            m = min(l + size, n)
            r = min(l + 2*size, n)
            merge(l, m, r)
        size *= 2

    # Yield recorded states
    for s in states:
        yield s
    yield {"arr": a[:], "highlight": None, "action": "done"}


def quick_sort(arr: List[int]) -> Generator[Dict, None, None]:
    a = arr[:]

    def _partition(low, high):
        pivot = a[high]
        i = low
        for j in range(low, high):
            yield {"arr": a[:], "highlight": (j, high), "action": "compare_pivot"}
            if a[j] <= pivot:
                a[i], a[j] = a[j], a[i]
                yield {"arr": a[:], "highlight": (i, j), "action": "swap"}
                i += 1
        a[i], a[high] = a[high], a[i]
        yield {"arr": a[:], "highlight": (i, high), "action": "swap_pivot"}
        return i

    def _qs(low, high):
        if low < high:
            part_gen = _partition(low, high)
            idx = None
            while True:
                try:
                    s = next(part_gen)
                    yield s
                except StopIteration as e:
                    idx = e.value
                    break
            yield from _qs(low, idx - 1)
            yield from _qs(idx + 1, high)

    if len(a) > 0:
        yield from _qs(0, len(a) - 1)
    yield {"arr": a[:], "highlight": None, "action": "done"}


def heap_sort(arr: List[int]) -> Generator[Dict, None, None]:
    a = arr[:]

    def heapify(n, i):
        largest = i
        l = 2 * i + 1
        r = 2 * i + 2
        if l < n:
            yield {"arr": a[:], "highlight": (i, l), "action": "compare"}
            if a[l] > a[largest]:
                largest = l
        if r < n:
            yield {"arr": a[:], "highlight": (largest, r), "action": "compare"}
            if a[r] > a[largest]:
                largest = r
        if largest != i:
            a[i], a[largest] = a[largest], a[i]
            yield {"arr": a[:], "highlight": (i, largest), "action": "swap"}
            yield from heapify(n, largest)

    n = len(a)
    for i in range(n // 2 - 1, -1, -1):
        yield from heapify(n, i)

    for i in range(n - 1, 0, -1):
        a[i], a[0] = a[0], a[i]
        yield {"arr": a[:], "highlight": (0, i), "action": "swap_root"}
        yield from heapify(i, 0)

    yield {"arr": a[:], "highlight": None, "action": "done"}


SORT_ALGOS = {
    "Bubble Sort": bubble_sort,
    "Insertion Sort": insertion_sort,
    "Selection Sort": selection_sort,
    "Merge Sort": merge_sort,
    "Quick Sort": quick_sort,
    "Heap Sort": heap_sort,
}


# =========================
# Pathfinding Algorithms (Generators)
# Yield dict: {"visited": set[(r,c)], "path": list[(r,c)], "current": (r,c), "grid": np.ndarray}
# Grid values: 0 free, 1 wall
# =========================

def neighbors(r, c, n):
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < n and 0 <= nc < n:
            yield nr, nc


def bfs(grid: np.ndarray, start: Tuple[int,int], goal: Tuple[int,int]) -> Generator[Dict, None, None]:
    n = grid.shape[0]
    q = deque([start])
    visited: Set[Tuple[int,int]] = {start}
    parent: Dict[Tuple[int,int], Optional[Tuple[int,int]]] = {start: None}
    while q:
        cur = q.popleft()
        yield {"grid": grid, "visited": visited.copy(), "path": reconstruct_path(parent, cur if cur==goal else None), "current": cur}
        if cur == goal:
            break
        for nr, nc in neighbors(*cur, n):
            if grid[nr, nc] == 1 or (nr, nc) in visited:
                continue
            visited.add((nr, nc))
            parent[(nr, nc)] = cur
            q.append((nr, nc))
    # Final path if reached
    final_path = reconstruct_path(parent, goal) if goal in parent else []
    yield {"grid": grid, "visited": visited, "path": final_path, "current": goal}


def dfs(grid: np.ndarray, start: Tuple[int,int], goal: Tuple[int,int]) -> Generator[Dict, None, None]:
    n = grid.shape[0]
    stack = [start]
    visited: Set[Tuple[int,int]] = set()
    parent: Dict[Tuple[int,int], Optional[Tuple[int,int]]] = {start: None}
    while stack:
        cur = stack.pop()
        if cur in visited:
            continue
        visited.add(cur)
        yield {"grid": grid, "visited": visited.copy(), "path": reconstruct_path(parent, cur if cur==goal else None), "current": cur}
        if cur == goal:
            break
        r, c = cur
        for nr, nc in [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]:
            if 0 <= nr < n and 0 <= nc < n and grid[nr, nc] != 1 and (nr, nc) not in visited:
                parent[(nr, nc)] = cur
                stack.append((nr, nc))
    final_path = reconstruct_path(parent, goal) if goal in parent else []
    yield {"grid": grid, "visited": visited, "path": final_path, "current": goal}


def dijkstra(grid: np.ndarray, start: Tuple[int,int], goal: Tuple[int,int]) -> Generator[Dict, None, None]:
    n = grid.shape[0]
    dist = {start: 0}
    parent = {start: None}
    visited: Set[Tuple[int,int]] = set()
    pq = [(0, start)]
    while pq:
        d, cur = heapq.heappop(pq)
        if cur in visited:
            continue
        visited.add(cur)
        yield {"grid": grid, "visited": visited.copy(), "path": reconstruct_path(parent, cur if cur==goal else None), "current": cur}
        if cur == goal:
            break
        for nr, nc in neighbors(*cur, n):
            if grid[nr, nc] == 1:
                continue
            nd = d + 1
            if nd < dist.get((nr, nc), float('inf')):
                dist[(nr, nc)] = nd
                parent[(nr, nc)] = cur
                heapq.heappush(pq, (nd, (nr, nc)))
    final_path = reconstruct_path(parent, goal) if goal in parent else []
    yield {"grid": grid, "visited": visited, "path": final_path, "current": goal}


def manhattan(a: Tuple[int,int], b: Tuple[int,int]) -> int:
    return abs(a[0]-b[0]) + abs(a[1]-b[1])


def a_star(grid: np.ndarray, start: Tuple[int,int], goal: Tuple[int,int]) -> Generator[Dict, None, None]:
    n = grid.shape[0]
    g = {start: 0}
    f = {start: manhattan(start, goal)}
    parent = {start: None}
    visited: Set[Tuple[int,int]] = set()
    pq = [(f[start], start)]
    while pq:
        _, cur = heapq.heappop(pq)
        if cur in visited:
            continue
        visited.add(cur)
        yield {"grid": grid, "visited": visited.copy(), "path": reconstruct_path(parent, cur if cur==goal else None), "current": cur}
        if cur == goal:
            break
        for nb in neighbors(*cur, n):
            if grid[nb] == 1:
                continue
            tentative_g = g[cur] + 1
            if tentative_g < g.get(nb, float('inf')):
                g[nb] = tentative_g
                f[nb] = tentative_g + manhattan(nb, goal)
                parent[nb] = cur
                heapq.heappush(pq, (f[nb], nb))
    final_path = reconstruct_path(parent, goal) if goal in parent else []
    yield {"grid": grid, "visited": visited, "path": final_path, "current": goal}


PATH_ALGOS = {
    "BFS": bfs,
    "DFS": dfs,
    "Dijkstra's": dijkstra,
    "A* (Manhattan)": a_star,
}


def reconstruct_path(parent: Dict[Tuple[int,int], Optional[Tuple[int,int]]], end: Optional[Tuple[int,int]]):
    if end is None or end not in parent:
        return []
    path = []
    cur = end
    while cur is not None:
        path.append(cur)
        cur = parent.get(cur)
    path.reverse()
    return path


# =========================
# Data Generators
# =========================

def generate_array(n: int) -> List[int]:
    # Unique values for clearer bar visualization
    arr = random.sample(range(1, n + 1), n)
    random.shuffle(arr)
    return arr


def generate_grid(n: int, wall_density: float) -> Tuple[np.ndarray, Tuple[int,int], Tuple[int,int]]:
    grid = (np.random.rand(n, n) < wall_density).astype(int)
    start = (0, 0)
    goal = (n-1, n-1)
    grid[start] = 0
    grid[goal] = 0
    return grid, start, goal


# =========================
# Visualization Helpers (Plotly)
# =========================

def render_bars(arr: List[int], highlight: Optional[Tuple[Optional[int], Optional[int]]] = None, title: str = ""): 
    colors = ["#A0AEC0"] * len(arr)  # gray
    if highlight and highlight[0] is not None and 0 <= highlight[0] < len(arr):
        colors[highlight[0]] = "#3182CE"  # blue
    if highlight and highlight[1] is not None and 0 <= highlight[1] < len(arr):
        colors[highlight[1]] = "#E53E3E"  # red
    fig = go.Figure(data=[go.Bar(x=list(range(len(arr))), y=arr, marker_color=colors)])
    fig.update_layout(
        title=title,
        xaxis_title="Index",
        yaxis_title="Value",
        margin=dict(l=10, r=10, t=40, b=10),
        height=300,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_grid(state: Dict, title: str = ""):
    grid = state["grid"]
    n = grid.shape[0]
    canvas = np.zeros_like(grid)
    canvas[grid == 1] = 1  # walls (black)

    # visited = 2 (yellow)
    for r, c in state.get("visited", []):
        canvas[r, c] = 2

    # path = 5 (green) on top
    for r, c in state.get("path", []):
        canvas[r, c] = 5

    # start/goal override
    if "current" in state and state["current"] is not None:
        cr, cc = state["current"]
        canvas[cr, cc] = max(canvas[cr, cc], 2)

    # start blue (3), goal red (4)
    # We don't store start/goal in state every yield; infer from corners
    canvas[0, 0] = 3
    canvas[n-1, n-1] = 4

    # Colors: 0 white, 1 black, 2 yellow, 3 blue, 4 red, 5 green
    colorscale = [
        [0.0, "#FFFFFF"],
        [0.199, "#FFFFFF"],
        [0.2, "#000000"],
        [0.399, "#000000"],
        [0.4, "#F6E05E"],
        [0.599, "#F6E05E"],
        [0.6, "#3182CE"],
        [0.699, "#3182CE"],
        [0.7, "#E53E3E"],
        [0.799, "#E53E3E"],
        [0.8, "#48BB78"],
        [1.0, "#48BB78"],
    ]

    fig = go.Figure(data=go.Heatmap(
        z=canvas,
        colorscale=colorscale,
        showscale=False,
    ))
    fig.update_layout(
        title=title,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
        margin=dict(l=10, r=10, t=40, b=10),
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)


# =========================
# Control Helpers
# =========================

def step_sort(side: str = "left"):
    gen_key = "sort_gen_left" if side == "left" else "sort_gen_right"
    state_key = "sort_state_left" if side == "left" else "sort_state_right"
    gen = st.session_state.get(gen_key)
    if gen is None:
        return False
    try:
        st.session_state[state_key] = next(gen)
        return True
    except StopIteration:
        st.session_state[gen_key] = None
        return False


def step_path():
    gen = st.session_state.get("path_gen")
    if gen is None:
        return False
    try:
        st.session_state["path_state"] = next(gen)
        return True
    except StopIteration:
        st.session_state["path_gen"] = None
        return False


# =========================
# App UI
# =========================

def sorting_tab():
    st.subheader("Sorting Algorithms")

    c1, c2, c3, c4 = st.columns([1.2, 1.2, 1, 1])
    with c1:
        st.session_state.sort_algo = st.selectbox("Algorithm (Left)", list(SORT_ALGOS.keys()), index=list(SORT_ALGOS.keys()).index(st.session_state.sort_algo))
    with c2:
        st.session_state.compare_mode = st.checkbox("Compare mode", value=st.session_state.compare_mode)
        if st.session_state.compare_mode:
            st.session_state.sort_algo_right = st.selectbox("Algorithm (Right)", list(SORT_ALGOS.keys()), index=list(SORT_ALGOS.keys()).index(st.session_state.sort_algo_right))
    with c3:
        st.session_state.arr_size = st.slider("Array size", 5, 100, st.session_state.arr_size, 1)
    with c4:
        st.session_state.sort_speed_ms = st.slider("Speed (ms/step)", 0, 1000, st.session_state.sort_speed_ms, 10)

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        if st.button("Generate new array", use_container_width=True):
            st.session_state.arr = generate_array(st.session_state.arr_size)
            # Reset gens & states
            st.session_state.sort_gen_left = None
            st.session_state.sort_gen_right = None
            st.session_state.sort_state_left = None
            st.session_state.sort_state_right = None
            st.session_state.sort_running = False
    with c6:
        if st.button("Start", use_container_width=True):
            if st.session_state.arr is None:
                st.session_state.arr = generate_array(st.session_state.arr_size)
            # Initialize generators if needed
            st.session_state.sort_gen_left = SORT_ALGOS[st.session_state.sort_algo](st.session_state.arr)
            if st.session_state.compare_mode:
                st.session_state.sort_gen_right = SORT_ALGOS[st.session_state.sort_algo_right](st.session_state.arr)
            st.session_state.sort_running = True
    with c7:
        if st.button("Pause", use_container_width=True):
            st.session_state.sort_running = False
    with c8:
        if st.button("Step", use_container_width=True):
            if st.session_state.sort_gen_left is None:
                if st.session_state.arr is None:
                    st.session_state.arr = generate_array(st.session_state.arr_size)
                st.session_state.sort_gen_left = SORT_ALGOS[st.session_state.sort_algo](st.session_state.arr)
                if st.session_state.compare_mode:
                    st.session_state.sort_gen_right = SORT_ALGOS[st.session_state.sort_algo_right](st.session_state.arr)
            step_sort("left")
            if st.session_state.compare_mode:
                step_sort("right")

    # Reset button row
    c9, c10 = st.columns([1,1])
    with c9:
        if st.button("Reset", type="secondary", use_container_width=True):
            st.session_state.sort_running = False
            st.session_state.sort_gen_left = None
            st.session_state.sort_gen_right = None
            st.session_state.sort_state_left = None
            st.session_state.sort_state_right = None

    # Visuals
    if st.session_state.arr is None:
        st.session_state.arr = generate_array(st.session_state.arr_size)

    left_state = st.session_state.sort_state_left or {"arr": st.session_state.arr, "highlight": None}
    render_bars(left_state["arr"], left_state.get("highlight"), title=st.session_state.sort_algo)

    if st.session_state.compare_mode:
        right_state = st.session_state.sort_state_right or {"arr": st.session_state.arr, "highlight": None}
        render_bars(right_state["arr"], right_state.get("highlight"), title=st.session_state.sort_algo_right)

    # Auto-play loop (one tick per run)
    if st.session_state.sort_running:
        progressed = step_sort("left")
        if st.session_state.compare_mode:
            progressed = step_sort("right") or progressed
        time.sleep(max(0, st.session_state.sort_speed_ms) / 1000.0)
        st.experimental_rerun()


def pathfinding_tab():
    st.subheader("Pathfinding Algorithms")

    c1, c2, c3, c4 = st.columns([1.2, 1.2, 1, 1])
    with c1:
        st.session_state.path_algo = st.selectbox("Algorithm", list(PATH_ALGOS.keys()), index=list(PATH_ALGOS.keys()).index(st.session_state.path_algo))
    with c2:
        st.session_state.grid_size = st.slider("Grid size", 5, 50, st.session_state.grid_size, 1)
    with c3:
        st.session_state.wall_density = st.slider("Wall density", 0.0, 0.5, float(st.session_state.wall_density), 0.01)
    with c4:
        st.session_state.path_speed_ms = st.slider("Speed (ms/step)", 0, 1000, st.session_state.path_speed_ms, 10)

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        if st.button("Generate new grid", use_container_width=True):
            g, s, t = generate_grid(st.session_state.grid_size, st.session_state.wall_density)
            st.session_state.grid, st.session_state.start, st.session_state.goal = g, s, t
            st.session_state.path_gen = None
            st.session_state.path_state = None
            st.session_state.path_running = False
    with c6:
        if st.button("Start", use_container_width=True):
            if st.session_state.grid is None:
                g, s, t = generate_grid(st.session_state.grid_size, st.session_state.wall_density)
                st.session_state.grid, st.session_state.start, st.session_state.goal = g, s, t
            st.session_state.path_gen = PATH_ALGOS[st.session_state.path_algo](st.session_state.grid, st.session_state.start, st.session_state.goal)
            st.session_state.path_running = True
    with c7:
        if st.button("Pause", use_container_width=True):
            st.session_state.path_running = False
    with c8:
        if st.button("Step", use_container_width=True):
            if st.session_state.path_gen is None:
                if st.session_state.grid is None:
                    g, s, t = generate_grid(st.session_state.grid_size, st.session_state.wall_density)
                    st.session_state.grid, st.session_state.start, st.session_state.goal = g, s, t
                st.session_state.path_gen = PATH_ALGOS[st.session_state.path_algo](st.session_state.grid, st.session_state.start, st.session_state.goal)
            step_path()

    # Reset button row
    if st.button("Reset", type="secondary", use_container_width=True):
        st.session_state.path_running = False
        st.session_state.path_gen = None
        st.session_state.path_state = None

    # Visual
    if st.session_state.grid is None:
        g, s, t = generate_grid(st.session_state.grid_size, st.session_state.wall_density)
        st.session_state.grid, st.session_state.start, st.session_state.goal = g, s, t

    state = st.session_state.path_state or {"grid": st.session_state.grid, "visited": set(), "path": [], "current": st.session_state.start}
    render_grid(state, title=st.session_state.path_algo)

    # Auto-play loop
    if st.session_state.path_running:
        step_path()
        time.sleep(max(0, st.session_state.path_speed_ms) / 1000.0)
        st.experimental_rerun()


# =========================
# Main
# =========================

def main():
    st.set_page_config(page_title="Algorithm Visual Playground", layout="wide")
    init_session_state()

    st.title("Algorithm Visual Playground")
    st.caption("Visualize classic sorting and pathfinding algorithms step-by-step. Start • Pause • Step • Reset")

    tabs = st.tabs(["Sorting", "Pathfinding", "About"])
    with tabs[0]:
        sorting_tab()
    with tabs[1]:
        pathfinding_tab()
    with tabs[2]:
        st.markdown(
            """
            ### About
            This app implements sorting and pathfinding **from scratch** using Python generators. 
            Each algorithm yields intermediate states to drive the animation in Streamlit.

            **Sorting**: Bubble, Insertion, Selection, Merge, Quick, Heap.  
            **Pathfinding**: BFS, DFS, Dijkstra’s, A* (Manhattan).

            **Controls**: Generate data, Start, Pause, Step, Reset, and control animation speed.

            Built with **Streamlit** + **Plotly**.
            """
        )


if __name__ == "__main__":
    main()
