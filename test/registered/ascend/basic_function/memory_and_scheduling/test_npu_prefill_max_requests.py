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
    """Testcase: Verify --prefill-max-requests takes effect correctly by checking log.

    [Test Category] Parameter
    [Test Target] --prefill-max-requests
    """
    PREFILL_MAX_REQUESTS = 5

    @classmethod
    def setUpClass(cls):
        cls.model = LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.log_file = "./server.log"

        with open(cls.log_file, "w", encoding="utf-8") as f:
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
                return_stdout_stderr=(f, f),
            )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        os.remove(cls.log_file)

    def _send_single_request(self):
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
            futures = [executor.submit(self._send_single_request) for _ in range(30)]
            wait(futures)

        with open(self.log_file, "r", encoding="utf-8") as f:
            logs = f.read()

        pattern = re.compile(r"Prefill batch, #new-req[:\s]+(\d+)", re.I)
        matches = pattern.findall(logs)

        self.assertGreater(len(matches), 0, "No Prefill batch logs found")

        for idx, num_str in enumerate(matches):
            current_num = int(num_str)
            self.assertLessEqual(
                current_num,
                self.PREFILL_MAX_REQUESTS,
                f"Prefill batch {idx+1} exceeds limit! current={current_num}, max allowed={self.PREFILL_MAX_REQUESTS}"
            )


if __name__ == "__main__":
    unittest.main()