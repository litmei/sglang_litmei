import os
import re
import unittest
import requests
from concurrent.futures import ThreadPoolExecutor, wait

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestModeImpl(CustomTestCase):
    """Testcase: Verify that the number of requests processed in a single batch by the service does not exceed
    the limit configured by --prefill-max-requests.

    [Test Category] Parameter
    [Test Target] --prefill-max-requests
    """
    PREFILL_MAX_REQUESTS = 5

    @classmethod
    def setUpClass(cls):
        cls.model = LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.log_file = "./server.log"

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
                "--model-impl",
                "transformers",
                "--prefill-max-requests",
                str(cls.PREFILL_MAX_REQUESTS),
                "--trust-remote-code",
                "--mem-fraction-static",
                "0.8",
            ],
            return_stdout_stderr=(cls.log_file, cls.log_file),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        if os.exists(cls.log_file):
            os.remove(cls.log_file)

    def send_single_request(self):
        requests.post(
            f"{self.base_url}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                    "ignore_eos": True,
                },
            },
        )

    def test_prefill_max_requests_concurrent(self):
        """Send 30 concurrent requests and verify no prefill batch exceeds the configured maximum"""
        with ThreadPoolExecutor(max_workers=30) as executor:
            futures = [executor.submit(self.send_single_request) for _ in range(30)]
            wait(futures)

        with open(self.log_file, "r", encoding="utf-8") as f:
            logs = f.read()

        pattern = re.compile(r"Prefill batch, #new-seq[:\s]+(\d+)", re.IGNORECASE)
        matches = pattern.findall(logs)

        self.assertGreater(len(matches), 0, "No prefill batch logs found")

        batch_sizes = [int(num) for num in matches]

        # All batches must not exceed the maximum value of 5
        for idx, size in enumerate(batch_sizes):
            self.assertLessEqual(
                size, self.PREFILL_MAX_REQUESTS,
                f"Batch {idx + 1} exceeds limit! current={size}, max={self.PREFILL_MAX_REQUESTS}"
            )

        # There must be at least one batch with size > 1
        has_batch_gt1 = any(size > 1 for size in batch_sizes)
        self.assertTrue(
            has_batch_gt1,
            f"No batch size > 1 found. All batches: {batch_sizes}"
        )


if __name__ == "__main__":
    unittest.main()