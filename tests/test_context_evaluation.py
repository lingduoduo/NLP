import contextlib
import importlib.util
import io
from pathlib import Path
import unittest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "NLP-Foundations"
    / "28_context_evaluation.py"
)
SPEC = importlib.util.spec_from_file_location("context_evaluation", MODULE_PATH)
context_evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(context_evaluation)


class RunNiahGridTests(unittest.TestCase):
    def test_header_uses_depth_by_length_label(self):
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            context_evaluation.run_niah_grid([], [])

        self.assertIn(r"depth\len", output.getvalue())


if __name__ == "__main__":
    unittest.main()
