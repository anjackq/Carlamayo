"""Regression tests for the lightweight spawn/import boundary.

The process road-assessment backend uses ``multiprocessing`` with the spawn
start method.  Importing the closed-loop entrypoint in one of those children
must therefore remain independent of the Alpamayo/PyTorch runtime.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULT_PREFIX = "CARLAMAYO_IMPORT_BOUNDARY_RESULT="
LAZY_INFERENCE_CALLABLES = (
    "configure_cuda_linalg_library",
    "extract_answer_text",
    "extract_cot_text",
    "extract_cot_texts",
    "extract_trajectory_samples",
    "load_model",
    "prepare_model_input",
    "run_inference",
    "run_vqa",
    "select_trajectory_by_prev_similarity",
)


def _run_probe(command: list[str], *, timeout_s: float = 60.0) -> dict:
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    assert completed.returncode == 0, (
        f"probe failed with exit code {completed.returncode}\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
    result_lines = [
        line
        for line in completed.stdout.splitlines()
        if line.startswith(RESULT_PREFIX)
    ]
    assert len(result_lines) == 1, (
        f"expected one probe result line, got {len(result_lines)}\n"
        f"stdout:\n{completed.stdout}"
    )
    return json.loads(result_lines[0][len(RESULT_PREFIX) :])


def test_closed_loop_import_keeps_inference_runtime_lazy():
    lazy_names = json.dumps(LAZY_INFERENCE_CALLABLES)
    probe = textwrap.dedent(
        f"""
        import importlib
        import json
        import sys

        lazy_names = {lazy_names}
        before = set(sys.modules)
        closed_loop = importlib.import_module("carlamayo_closed_loop")
        after = set(sys.modules)
        forbidden_roots = ("torch", "transformers", "module.inference")
        added_forbidden = sorted(
            name
            for name in after - before
            if any(
                name == root or name.startswith(root + ".")
                for root in forbidden_roots
            )
        )
        missing = sorted(
            name for name in lazy_names if name not in closed_loop.__dict__
        )
        not_callable = sorted(
            name
            for name in lazy_names
            if name in closed_loop.__dict__
            and not callable(closed_loop.__dict__[name])
        )

        # Accessing the compatibility proxy itself must not initialize torch.
        torch_proxy_present = "torch" in closed_loop.__dict__
        torch_loaded_after_proxy_access = False
        if torch_proxy_present:
            closed_loop.__dict__["torch"]
            torch_loaded_after_proxy_access = "torch" in sys.modules

        print(
            {RESULT_PREFIX!r}
            + json.dumps(
                {{
                    "added_forbidden": added_forbidden,
                    "missing": missing,
                    "not_callable": not_callable,
                    "torch_proxy_present": torch_proxy_present,
                    "torch_loaded_after_proxy_access": (
                        torch_loaded_after_proxy_access
                    ),
                }},
                sort_keys=True,
            )
        )
        """
    )

    result = _run_probe([sys.executable, "-c", probe])

    assert result["added_forbidden"] == []
    assert result["missing"] == []
    assert result["not_callable"] == []
    assert result["torch_proxy_present"] is True
    assert result["torch_loaded_after_proxy_access"] is False


def test_visualization_accepts_tensor_protocol_without_importing_torch():
    probe = textwrap.dedent(
        f"""
        import json
        import sys

        import numpy as np

        before = set(sys.modules)
        from module.visualization import project_trajectory_to_image

        class TensorLike:
            def __init__(self, array):
                self._array = array

            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return self._array

        image = np.zeros((80, 120, 3), dtype=np.uint8)
        trajectory = TensorLike(
            np.array(
                [[[2.0, 0.0, 0.0], [6.0, 0.0, 0.0]]],
                dtype=np.float32,
            )
        )
        rendered = project_trajectory_to_image(image, trajectory)
        after = set(sys.modules)
        added_torch_modules = sorted(
            name
            for name in after - before
            if name == "torch" or name.startswith("torch.")
        )
        print(
            {RESULT_PREFIX!r}
            + json.dumps(
                {{
                    "added_torch_modules": added_torch_modules,
                    "shape": list(rendered.shape),
                    "sum": int(rendered.sum()),
                }},
                sort_keys=True,
            )
        )
        """
    )

    result = _run_probe([sys.executable, "-c", probe])

    assert result["added_torch_modules"] == []
    assert result["shape"] == [80, 120, 3]
    assert result["sum"] > 0


def test_spawn_child_import_reports_lightweight_runtime_and_rss(tmp_path):
    probe_path = tmp_path / "spawn_import_probe.py"
    probe_path.write_text(
        textwrap.dedent(
            f"""
            import json
            import multiprocessing
            import resource
            import sys

            sys.path.insert(0, {str(REPO_ROOT)!r})


            def import_in_spawn_child(result_queue):
                before = set(sys.modules)
                import carlamayo_closed_loop

                after = set(sys.modules)
                forbidden_roots = ("torch", "transformers", "module.inference")
                result_queue.put(
                    {{
                        "added_forbidden": sorted(
                            name
                            for name in after - before
                            if any(
                                name == root or name.startswith(root + ".")
                                for root in forbidden_roots
                            )
                        ),
                        # Linux reports ru_maxrss in KiB.  It is diagnostic only:
                        # no machine-dependent upper bound belongs in this test.
                        "maximum_rss_kib": int(
                            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                        ),
                        "lazy_run_inference_callable": callable(
                            carlamayo_closed_loop.__dict__.get("run_inference")
                        ),
                    }}
                )


            if __name__ == "__main__":
                context = multiprocessing.get_context("spawn")
                result_queue = context.Queue(maxsize=1)
                child = context.Process(
                    target=import_in_spawn_child,
                    args=(result_queue,),
                )
                child.start()
                child.join(60.0)
                if child.is_alive():
                    child.terminate()
                    child.join(10.0)
                    raise RuntimeError("spawn import child timed out")
                if child.exitcode != 0:
                    raise RuntimeError(
                        f"spawn import child exited with {{child.exitcode}}"
                    )
                result = result_queue.get(timeout=5.0)
                result_queue.close()
                print(
                    {RESULT_PREFIX!r}
                    + json.dumps(result, sort_keys=True)
                )
            """
        ),
        encoding="utf-8",
    )

    result = _run_probe([sys.executable, str(probe_path)], timeout_s=75.0)

    assert result["added_forbidden"] == []
    assert result["lazy_run_inference_callable"] is True
    assert isinstance(result["maximum_rss_kib"], int)
    assert result["maximum_rss_kib"] > 0
