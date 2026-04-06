import logging
import unittest
from types import SimpleNamespace

import os

# ============ [Local path override - for local debugging only] ============
LOCAL_MODEL_WEIGHTS_DIR = "/home/weights"
import sglang.test.ascend.test_ascend_utils as _utils
_utils.MODEL_WEIGHTS_DIR = LOCAL_MODEL_WEIGHTS_DIR
_utils.HF_MODEL_WEIGHTS_DIR = LOCAL_MODEL_WEIGHTS_DIR
_utils.QWEN3_32B_WEIGHTS_PATH = os.path.join(
    LOCAL_MODEL_WEIGHTS_DIR, "Qwen/Qwen3-32B"
)
# =========================================================================

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_32B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.few_shot_gsm8k import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_OTHER_ARGS = [
    "--chunked-prefill-size", "256",
    "--attention-backend", "ascend",
    "--disable-cuda-graph",
    "--mem-fraction-static", "0.8",
    "--tp-size", "4",
    "--enable-dynamic-batch-tokenizer",
    "--dynamic-batch-tokenizer-batch-size", "4",
    "--log-level", "debug",
]

MODEL_NAME = QWEN3_32B_WEIGHTS_PATH

register_npu_ci(est_time=400, suite="nightly-4-npu-a3", nightly=True)


def launch_server_with_timeout(model_name, base_url, timeout, other_args_base):
    other_args = other_args_base.copy()
    idx = other_args.index("--dynamic-batch-tokenizer-batch-timeout") + 1 if "--dynamic-batch-tokenizer-batch-timeout" in other_args else -1
    if idx > 0:
        other_args[idx] = str(timeout)
    else:
        other_args.extend(["--dynamic-batch-tokenizer-batch-timeout", str(timeout)])
    return popen_launch_server(model_name, base_url, timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH, other_args=other_args)


class BaseQwenTest(CustomTestCase):
    # Qwen3-32B baseline accuracy on GSM8K (5-shot) is ~0.87-0.88.
    # Allow 0.01 tolerance when dynamic batch tokenizer is enabled.
    accuracy = 0.86

    def _run_gsm8k_test(self, scenario):
        args = SimpleNamespace(
            num_shots=5,
            data_path="/home/x30082852/test.jsonl",
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host="127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval(args)   # uses default parameters from few_shot_gsm8k
        self.assertGreaterEqual(metrics["accuracy"], self.accuracy,
                                f"accuracy {metrics['accuracy']} < {self.accuracy}")
        server_info = requests.get(self.base_url + "/get_server_info")
        logger.info(f"{scenario}: server_info={server_info}")


class TestQwen32BTimeoutMin(BaseQwenTest):
    """GSM8K accuracy with dynamic batch tokenizer timeout = 0.001s (min recommended)."""
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = launch_server_with_timeout(MODEL_NAME, cls.base_url, 0.001, BASE_OTHER_ARGS)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k_timeout_min(self):
        self._run_gsm8k_test("timeout=0.001")


class TestQwen32BTimeoutMax(BaseQwenTest):
    """GSM8K accuracy with dynamic batch tokenizer timeout = 0.1s (max recommended)."""
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = launch_server_with_timeout(MODEL_NAME, cls.base_url, 0.1, BASE_OTHER_ARGS)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k_timeout_max(self):
        self._run_gsm8k_test("timeout=0.01")


if __name__ == "__main__":
    unittest.main()