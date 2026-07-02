import json
import unittest


class SupernetNativeE5Tests(unittest.TestCase):
    def test_supernet_native_module_exports_e5_evaluator_with_warm_start(self):
        from aicir.qas.primitives.ansatz import SupernetAnsatzGene
        from aicir.qas.problems.hamiltonians import VQEProblem
        from aicir.qas.vqe_loop.p0_supernet_native import build_native_supernet_e5_evaluator

        problem = VQEProblem(
            name="two_qubit_z",
            n_qubits=2,
            hamiltonian=((-1.0, "ZI"),),
            reference_energy=-1.0,
        )
        gene = SupernetAnsatzGene(
            n_qubits=2,
            single_qubit_layers=(("ry", "rz"),),
            two_qubit_layers=(("rzz",),),
            two_qubit_pairs=((0, 1),),
        )
        calls = {"factory": 0, "optimize": 0, "rank": 0, "finetune": 0, "build": 0}
        case = self

        class FakeCircuit:
            gates = [
                {"name": "ry", "parameter": 0.1},
                {"name": "rz", "parameter": 0.2},
                {"name": "rzz", "parameter": 0.3},
            ]

        class FakeSupernet:
            def __init__(self, config):
                calls["factory"] += 1
                self.config = config

            def optimize_supernet(self, objective=None, dataset=None, hamiltonian=None):
                calls["optimize"] += 1
                case.assertIsNone(objective)
                case.assertIsNone(dataset)
                case.assertEqual(hamiltonian.n_qubits, 2)
                return []

            def rank_architectures(self, objective=None, dataset=None, hamiltonian=None, *, candidates=None, split="train"):
                calls["rank"] += 1
                case.assertEqual(split, "train")
                case.assertEqual(len(candidates), 1)
                return [
                    {
                        "candidate_index": 0,
                        "architecture": candidates[0],
                        "selected_supernet_id": 1,
                        "score": -0.75,
                        "candidate_losses": [-0.5, -0.75],
                        "two_qubit_count": 1,
                    }
                ]

            def finetune_architecture(self, architecture, supernet_id, objective=None, dataset=None, hamiltonian=None):
                calls["finetune"] += 1
                case.assertEqual(supernet_id, 1)
                return None, {}, [], -1.5

            def build_circuit(self, architecture, supernet_id=0, parameters=None):
                calls["build"] += 1
                case.assertEqual(supernet_id, 1)
                case.assertIsNone(parameters)
                return FakeCircuit(), [], []

        evaluator = build_native_supernet_e5_evaluator(
            problem=problem,
            supernet_num=2,
            supernet_steps=3,
            finetune_steps=5,
            seed=7,
            supernet_factory=FakeSupernet,
        )
        row = {"architecture_id": "arch_a", "ansatz_gene": json.dumps(gene.to_jsonable()), "n_qubits": 2}

        result = evaluator(row)
        warm_start = evaluator.warm_start_parameters(dict(row))

        self.assertEqual(result["E5"], -1.5)
        self.assertEqual(result["E5_mean"], -0.625)
        self.assertEqual(result["E5_min"], -0.75)
        self.assertEqual(result["E5_std"], 0.125)
        self.assertEqual(result["two_q_count"], 1)
        self.assertEqual(warm_start, [0.1, 0.2, 0.3])
        self.assertEqual(calls, {"factory": 1, "optimize": 1, "rank": 2, "finetune": 1, "build": 1})


if __name__ == "__main__":
    unittest.main()

