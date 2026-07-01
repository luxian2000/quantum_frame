from __future__ import annotations

from .polynomial import Polynomial, Term
from .implication import build_graph, strong_components
from .posiform import to_posiform


def split_polynomial(poly: Polynomial) -> list[Polynomial]:
    """Decompose a polynomial into variable-disjoint subpolynomials.

    Two variables are linked when they appear together in a monomial. The
    connected components of this interaction graph induce a partition of the
    monomials in which every monomial belongs to exactly one part and no two
    parts share a variable. Because the parts share no variables, minimizing
    each part independently and summing recovers the global minimum::

        min f == sum(min f_i for f_i in split_polynomial(f))

    This is the general form of the Billionnet-Jaumard decomposition (Thm. 3.1,
    "A decomposition method for minimizing quadratic pseudo-Boolean functions"):
    each monomial appears in at most one subfunction and the minima add.
    Connectivity is resolved with a union-find (Tarjan's disjoint-set) over the
    variable ids, and works for polynomials of any degree.

    The constant term, having no variables, is returned as its own part (so the
    parts still partition the terms and the sum-of-minima identity holds). An
    empty polynomial returns an empty list.

    Extension point
    ---------------
    A stronger, sign-aware split is implemented in :func:`split_cores`
    (Billionnet-Jaumard): it converts to a posiform, builds the literal
    implication graph, runs Tarjan SCC, and peels off the components holding a
    variable and its complement. It is quadratic-only and can break a single
    connected block that this connected-components split leaves whole.
    """
    if not poly.terms:
        return []

    parent: dict[int, int] = {}

    def find(v: int) -> int:
        root = v
        while parent[root] != root:
            root = parent[root]
        while parent[v] != root:
            parent[v], v = root, parent[v]
        return root

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for term in poly.terms:
        for var_id in term:
            parent.setdefault(var_id, var_id)
        for var_id in term[1:]:
            union(term[0], var_id)

    groups: dict[int | None, dict[Term, float]] = {}
    for term, coeff in poly.terms.items():
        key = None if not term else find(term[0])
        groups.setdefault(key, {})[term] = coeff

    def order(item: tuple[int | None, dict[Term, float]]) -> tuple[int, tuple[int, ...]]:
        key, terms = item
        if key is None:
            return (1, ())  # constant part sorts last
        component_vars = tuple(sorted({var_id for term in terms for var_id in term}))
        return (0, component_vars)

    return [Polynomial(terms, poly.registry) for _, terms in sorted(groups.items(), key=order)]


def _literal_poly(node: int, registry) -> Polynomial:
    var_id = node >> 1
    variable = Polynomial({(var_id,): 1.0}, registry)
    if node & 1:  # complemented literal x_bar = 1 - x
        return Polynomial.constant(1.0, registry) - variable
    return variable


def split_cores(poly: Polynomial) -> list[Polynomial]:
    """Split a quadratic polynomial into Billionnet-Jaumard hard cores.

    Converts the polynomial to a posiform, builds the 2-SAT implication graph,
    and isolates the strongly connected components that contain both a variable
    and its complement (the "hard cores", B-J Thm 3.1). Each core is expanded
    back to an x-space subpolynomial; the remaining (consistent) monomials form
    a posiform whose minimum is provably 0 (Aspvall-Plass-Tarjan Thm 2.1) and
    are dropped. The posiform constant is returned as a trailing constant piece.

    Unlike :func:`split_polynomial`, this preserves only the *minimum*, not the
    term partition::

        min f == sum(min f_i for f_i in split_cores(f))

    It is quadratic-only; degree >= 3 raises ``ValueError``. This is the
    posiform/implication-graph refinement referenced in ``split_polynomial``:
    it can break a single *connected* quadratic block into several independent
    cores, which connected-component splitting cannot.
    """
    if poly.degree() > 2:
        raise ValueError("split_cores only supports degree <= 2 polynomials.")

    posiform = to_posiform(poly)
    adjacency = build_graph(posiform)
    components = strong_components(adjacency)

    comp_id = {node: i for i, comp in enumerate(components) for node in comp}
    self_dual: set[int] = set()
    for i, comp in enumerate(components):
        members = set(comp)
        if any((node ^ 1) in members for node in members):
            self_dual.add(i)

    cores: dict[int, Polynomial] = {}
    for key, coeff in posiform.terms.items():
        ids = {comp_id[node] for node in key}
        if len(ids) != 1:
            continue  # spans components -> consistent leftover (min 0)
        (cid,) = ids
        if cid not in self_dual:
            continue  # consistent-only component -> leftover (min 0)
        monomial = Polynomial.constant(coeff, poly.registry)
        for node in key:
            monomial = monomial * _literal_poly(node, poly.registry)
        cores[cid] = cores.get(cid, Polynomial.constant(0.0, poly.registry)) + monomial

    pieces = [cores[cid] for cid in sorted(cores)]
    if abs(posiform.constant) > 1e-12:
        pieces.append(Polynomial.constant(posiform.constant, poly.registry))
    return pieces
