from __future__ import annotations

from .posiform import Posiform


def build_graph(posiform: Posiform) -> dict[int, list[int]]:
    """Build the 2-SAT implication graph of the posiform's monomials.

    Each monomial must be 0 at the minimum. A pair m = a*b contributes the
    clause (~a or ~b), i.e. arcs a -> complement(b) and b -> complement(a). A
    unit m = a contributes the clause (~a), i.e. arc a -> complement(a).
    """
    adjacency: dict[int, list[int]] = {}

    def ensure(node: int) -> None:
        if node not in adjacency:
            adjacency[node] = []

    def arc(src: int, dst: int) -> None:
        ensure(src)
        ensure(dst)
        adjacency[src].append(dst)

    for key in posiform.terms:
        if len(key) == 1:
            a = key[0]
            arc(a, a ^ 1)
        else:
            a, b = key
            arc(a, b ^ 1)
            arc(b, a ^ 1)
    return adjacency


def strong_components(adjacency: dict[int, list[int]]) -> list[list[int]]:
    """Tarjan's strongly connected components, iterative to avoid deep recursion."""
    index_counter = [0]
    index: dict[int, int] = {}
    lowlink: dict[int, int] = {}
    on_stack: dict[int, bool] = {}
    stack: list[int] = []
    result: list[list[int]] = []

    def connect(root: int) -> None:
        work = [(root, 0)]
        while work:
            node, cursor = work[-1]
            if cursor == 0:
                index[node] = index_counter[0]
                lowlink[node] = index_counter[0]
                index_counter[0] += 1
                stack.append(node)
                on_stack[node] = True
            recursed = False
            neighbors = adjacency[node]
            for offset in range(cursor, len(neighbors)):
                nxt = neighbors[offset]
                if nxt not in index:
                    work[-1] = (node, offset + 1)
                    work.append((nxt, 0))
                    recursed = True
                    break
                if on_stack.get(nxt):
                    lowlink[node] = min(lowlink[node], index[nxt])
            if recursed:
                continue
            if lowlink[node] == index[node]:
                component: list[int] = []
                while True:
                    member = stack.pop()
                    on_stack[member] = False
                    component.append(member)
                    if member == node:
                        break
                result.append(component)
            work.pop()
            if work:
                parent = work[-1][0]
                lowlink[parent] = min(lowlink[parent], lowlink[node])

    for node in adjacency:
        if node not in index:
            connect(node)
    return result
