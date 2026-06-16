"""用于学习和调试 QUBO 建模 API 的详细 旅行商问题 TSP （Traveling Salesman Problem）示例。

请在仓库根目录运行：

    python demos/qubo_tsp_debug_demo.py

本 demo 会保留关键中间对象，
便于开发者设置断点，观察变量、多项式、约束、模型、Builder、QUBO 字典、Ising 项、QAOA 项以及解码结果之间的关系。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from pprint import pprint
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aicir.optimization.qubo import (
    Model,
    ModelContext,
    Polynomial,
    QuboBuilder,
    VariableRegistry,
    brute_force_model,
    cardinality,
    decode_best_solutions,
    most_likely_qaoa_assignment,
    quadratic_objective,
    qubo_energy,
    run_qubo_qaoa,
    tsp_qubo_builder,
)


CityMatrix = Sequence[Sequence[float]]


def build_tsp_model_verbose(
    distances: CityMatrix,
    *,
    penalty: float,
    prefix: str,
    registry: VariableRegistry,
) -> tuple[Model, list[list[Polynomial]]]:
    """以显式方式构造 TSP 模型，并暴露每个关键 QUBO 建模对象。"""

    city_count = len(distances)

    # 步骤 1：创建变量矩阵。
    #
    # x[city][position] = 1 表示城市 ``city`` 被放在巡回路径的第 ``position`` 个位置。
    # 这些变量通过 ``ctx`` 共享同一个 ``registry``。
    ctx = ModelContext(registry)
    x = ctx.binary_array(prefix, (city_count, city_count))

    # 步骤 2：把路径距离目标函数显式构造成二次项列表。
    #
    # 如果城市 i 位于位置 t，城市 j 位于位置 t+1，则路径需要付出 distances[i][j]。
    # ``quadratic_objective`` 会保留这个稀疏结构，后续可以直接写入 QuboBuilder。
    travel_terms = [
        (distances[left][right], x[left][position], x[right][(position + 1) % city_count])
        for position in range(city_count)
        for left in range(city_count)
        for right in range(city_count)
        if left != right
    ]
    travel_objective = quadratic_objective(travel_terms)

    # 步骤 3：显式创建约束。
    #
    # 每个城市在所有位置中必须恰好出现一次。
    city_constraints = [
        cardinality(x[city], count=1, penalty=penalty, label=f"city_{city}_once")
        for city in range(city_count)
    ]

    # 每个位置必须恰好放置一个城市。
    position_constraints = [
        cardinality(
            [x[city][position] for city in range(city_count)],
            count=1,
            penalty=penalty,
            label=f"position_{position}_filled",
        )
        for position in range(city_count)
    ]

    # 步骤 4：把目标函数和约束组合成 Model。
    model = Model(ctx.zero())
    model.add_objective(travel_objective)
    model.add_constraints(city_constraints)
    model.add_constraints(position_constraints)
    return model, x


def build_tsp_builder_verbose(
    distances: CityMatrix,
    *,
    penalty: float,
    prefix: str,
    registry: VariableRegistry,
) -> QuboBuilder:
    """通过底层 Builder API 直接构造同一个 TSP QUBO。"""

    city_count = len(distances)
    builder = QuboBuilder(registry=registry)

    # 步骤 1：用整数 ID 创建变量。这是 ``ctx.binary_array`` 的底层写法。
    x_ids = [
        [registry.get_or_create(f"{prefix}[{city}][{position}]") for position in range(city_count)]
        for city in range(city_count)
    ]

    # 步骤 2：直接写入目标函数系数。
    for position in range(city_count):
        next_position = (position + 1) % city_count
        for left in range(city_count):
            for right in range(city_count):
                if left != right:
                    builder.add_quadratic(
                        x_ids[left][position],
                        x_ids[right][next_position],
                        distances[left][right],
                    )

    # 步骤 3：把排列约束直接写成 cardinality 惩罚项。
    for city in range(city_count):
        builder.add_cardinality_penalty(x_ids[city], count=1, penalty=penalty)
    for position in range(city_count):
        builder.add_cardinality_penalty(
            [x_ids[city][position] for city in range(city_count)],
            count=1,
            penalty=penalty,
        )
    return builder


def decode_tour(assignment: dict[str, int], city_count: int, prefix: str) -> list[int | None]:
    """把 x[city][position] 的取值解码成按位置排序的巡回路径。"""

    tour: list[int | None] = [None] * city_count
    for city in range(city_count):
        for position in range(city_count):
            if assignment.get(f"{prefix}[{city}][{position}]", 0) == 1:
                tour[position] = city
    return tour


def tour_distance(tour: Sequence[int | None], distances: CityMatrix) -> float | None:
    """返回闭环路径距离；如果解码路径非法，则返回 None。"""

    if any(city is None for city in tour):
        return None
    city_count = len(tour)
    return sum(
        distances[int(tour[position])][int(tour[(position + 1) % city_count])]
        for position in range(city_count)
    )


def print_section(title: str) -> None:
    print(f"\n=== {title} ===")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="详细版 TSP QUBO 建模调试示例。")
    parser.add_argument(
        "--run-qaoa",
        action="store_true",
        help="运行 dense BasicQAOA 辅助流程。该步骤默认关闭，因为 3 城市 TSP 已经需要 9 个量子比特。",
    )
    parser.add_argument("--qaoa-iters", type=int, default=10, help="启用 --run-qaoa 时的 BasicQAOA 迭代次数。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 三城市 TSP 可以让 QAOA 矩阵相对较小：3 x 3 个二进制变量对应 9 个量子比特，
    # dense Hamiltonian 的大小为 512 x 512。这个规模适合调试，同时也足够暴露建模 API。
    distances = [
        [0.0, 2.0, 9.0],
        [1.0, 0.0, 6.0],
        [15.0, 7.0, 0.0],
    ]
    city_count = len(distances)
    penalty = 30.0
    prefix = "x"

    print_section("1. VariableRegistry 与 ModelContext")
    registry = VariableRegistry()
    model, x = build_tsp_model_verbose(
        distances,
        penalty=penalty,
        prefix=prefix,
        registry=registry,
    )
    print("变量名列表:")
    pprint(registry.names())
    print("第一个变量对应的多项式:", x[0][0])
    print("第一个变量的元数据:", registry.metadata(0))

    print_section("2. Polynomial、ObjectiveFragment、Constraint、Model")
    model_polynomial = model.polynomial()
    print("模型中的目标函数片段数量:", len(model.objective_fragments))
    print("模型中的约束数量:", len(model.constraints))
    print("编译后多项式次数:", model_polynomial.degree())
    print("编译后多项式包含的变量数量:", len(model_polynomial.variables()))
    print("第一个约束标签:", model.constraints[0].label)
    print("第一个约束表达式次数:", model.constraints[0].expression.degree())

    print_section("3. 从 Model 编译到 QuboBuilder")
    model_builder = model.to_qubo_builder()
    print("Builder 中的变量数量:", len(model_builder.registry.names()))
    print("Builder 中的 QUBO 项数量:", len(model_builder.qubo))
    print("Builder 中的常数偏移:", model_builder.offset)

    print_section("4. 直接使用 QuboBuilder 的底层路径")
    builder_registry = VariableRegistry()
    direct_builder = build_tsp_builder_verbose(
        distances,
        penalty=penalty,
        prefix=prefix,
        registry=builder_registry,
    )
    library_builder = tsp_qubo_builder(
        distances,
        penalty=penalty,
        prefix=prefix,
        registry=VariableRegistry(),
    )
    print("手写 Builder 是否与 Model 编译结果一致:", direct_builder.to_qubo() == model_builder.to_qubo())
    print("库函数 tsp_qubo_builder 生成的 QUBO 项数量:", len(library_builder.qubo))

    print_section("5. QUBO、稀疏矩阵、Ising 和 QAOA 导出")
    qubo_by_id, offset = model.to_qubo_indices()
    qubo_by_name, _ = model.to_qubo()
    sparse = model.to_sparse_matrix()
    ising = model.to_ising_indices()
    qaoa_terms, qaoa_offset, variable_names = model.to_qaoa_terms()
    print("按变量 ID 表示的 QUBO 项数量:", len(qubo_by_id))
    print("按变量名表示的 QUBO 项数量:", len(qubo_by_name))
    print("QUBO 常数偏移:", offset)
    print("稀疏矩阵形状:", sparse.shape)
    print("Ising h 项数量:", len(ising.h))
    print("Ising J 项数量:", len(ising.J))
    print("QAOA 项数量:", len(qaoa_terms))
    print("QAOA 常数偏移:", qaoa_offset)
    print("前五个 QAOA 项:", qaoa_terms[:5])

    print_section("6. 穷举验证与结果解码")
    brute_force = brute_force_model(model, max_variables=12)
    decoded_solutions = decode_best_solutions(brute_force, registry)
    best_solution = decoded_solutions[0]
    best_assignment = best_solution.decisions()
    best_tour = decode_tour(best_assignment, city_count, prefix)
    print("最优 QUBO 能量:", brute_force.best_energy)
    print("最优 bit assignment 数量:", len(brute_force.best_assignments))
    print("解码得到的最优巡回路径:", best_tour)
    print("解码路径距离:", tour_distance(best_tour, distances))
    print("解码最优解对应的能量:", qubo_energy(qubo_by_id, brute_force.best_assignments[0], offset))

    print_section("7. Dense BasicQAOA 运行")
    if not args.run_qaoa:
        print("已跳过：传入 --run-qaoa 可在这个 9 量子比特 TSP 模型上执行 dense BasicQAOA")
        return

    print("说明：上面的穷举结果用于验证 QUBO 建模正确性；这里的 dense QAOA 主要用于调试 API 调用链。")
    print("说明：当 p=1 且迭代次数较少时，最大概率量子态可能并不是合法 TSP 路径。")
    result = run_qubo_qaoa(model, p=1, max_iters=args.qaoa_iters, lr=0.05, seed=11)
    if variable_names is None:
        raise RuntimeError("TSP QAOA 导出结果应当包含变量名。")
    qaoa_decoded = most_likely_qaoa_assignment(result.statevector, variable_names)
    qaoa_tour = decode_tour(qaoa_decoded.assignment, city_count, prefix)
    print("QAOA 最优能量:", round(result.energy, 6))
    print("QAOA 最大概率 bitstring:", qaoa_decoded.bitstring)
    print("QAOA 最大概率:", round(qaoa_decoded.probability, 6))
    print("QAOA 解码路径:", qaoa_tour)
    print("QAOA 解码路径距离:", tour_distance(qaoa_tour, distances))
    print("QAOA 前五个能量历史值:", [round(value, 6) for value in result.energy_history[:5]])


if __name__ == "__main__":
    main()
