import importlib.util
import pathlib


def test_demo_npu_tensor_importable():
    path = pathlib.Path(__file__).resolve().parents[2] / "demos" / "demo_npu_tensor.py"
    spec = importlib.util.spec_from_file_location("demo_npu_tensor", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # 提供可复用的核对函数，CPU 上亦可跑
    assert hasattr(module, "run_checks")
    module.run_checks(allow_cpu_fallback=True)
