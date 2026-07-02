from aicir.optimization.qubo import Binary
from aicir.optimization.qubo.modeling import VariableRegistry
from aicir.optimization.qubo.modeling.posiform import to_posiform
from aicir.optimization.qubo.modeling.implication import build_graph, strong_components


def test_build_graph_pair_arcs():
    reg = VariableRegistry()
    a = Binary("a", registry=reg)  # id 0
    b = Binary("b", registry=reg)  # id 1
    posi = to_posiform(a * b)      # terms {(0, 2): 1}
    graph = build_graph(posi)
    # pair (0, 2) -> arcs 0 -> 2^1=3 and 2 -> 0^1=1
    assert set(graph[0]) == {3}
    assert set(graph[2]) == {1}
    assert graph[1] == []
    assert graph[3] == []


def test_build_graph_unit_arc():
    reg = VariableRegistry()
    a = Binary("a", registry=reg)  # id 0
    posi = to_posiform(-1 * a)     # terms {(1,): 1}
    graph = build_graph(posi)
    # unit (1,) -> arc 1 -> 1^1=0
    assert set(graph[1]) == {0}
    assert graph[0] == []


def test_strong_components_cycle_and_singleton():
    adjacency = {0: [1], 1: [2], 2: [0], 3: []}
    components = strong_components(adjacency)
    normalized = sorted(sorted(component) for component in components)
    assert normalized == [[0, 1, 2], [3]]
