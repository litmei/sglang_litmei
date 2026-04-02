import os
import sys
import time
import unittest
from datetime import datetime

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.output_capturer import OutputCapturer
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ascend.test_npu_logging import TestNPULoggingBase
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

LOG_DUMP_FILE = f"server_request_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
CUSTOM_SERVER_WAIT_TIME = 20

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestNPUEnableRequestTimeStatsLogging(TestNPULoggingBase):
    """Testcase: Verify the functionality of --enable-request-time-stats-logging to generate Req Time Stats logs on Ascend backend with Llama-3.2-1B-Instruct model.

    [Test Category] Parameter
    [Test Target] --enable-request-time-stats-logging
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.output_capturer = OutputCapturer()
        cls.output_capturer.start()
        cls.other_args.extend(["--enable-request-time-stats-logging"])
        cls.other_args.extend(["--base-gpu-id", "10"])
        cls.launch_server()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        cls.output_capturer.stop()

    def test_enable_request_time_stats_logging(self):
        # 1. Send a request to trigger the server to generate Req Time Stats logs
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
            },
        )

        # 2. Extend the log writing waiting time to ensure Req Time Stats is fully written
        time.sleep(10)
        # 4. Assert that the request was sent successfully
        self.assertEqual(response.status_code, 200, "Failed to call generate API")

        # 5. Read the full content of the log file
        content = self.output_capturer.get_all()
        target_keyword = "Req Time Stats"
        self.assertIn(
            target_keyword,
            content,
            f"Keyword not found in server logs.",
        )


if __name__ == "__main__":
    unittest.main()
