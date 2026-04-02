import io
import json
import os
import tempfile
import time
import unittest
from pathlib import Path

import requests

# from sglang.srt.constants import HEALTH_CHECK_RID_PREFIX
from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.output_capturer import OutputCapturer
from sglang.test.ascend.test_npu_logging import TestNPULoggingBase
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=120, suite="nightly-1-npu-a3", nightly=True)

TEST_ROUTING_KEY = "test-routing-key-12345"
TEST_CUSTOM_HEADER_NAME = "X-Test-Header"
TEST_CUSTOM_HEADER_VALUE = "test-header-value-67890"
# TEST_MODEL_NAME = "Qwen/Qwen3-0.6B"
TEST_MODEL_NAME = "/home/weights/Qwen/Qwen3-0.6B"


class TestNPUEnableRequestTimeStatsLogging(TestNPULoggingBase):
    """Testcase: Verify the functionality of --enable-request-time-stats-logging to generate Req Time Stats logs on Ascend backend with Llama-3.2-1B-Instruct model.

    [Test Category] Parameter
    [Test Target] --enable-request-time-stats-logging
    """

    log_requests_format = "text"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.output_capturer = OutputCapturer()
        cls.output_capturer.start()
        cls.other_args.extend(["--log-requests-format", cls.log_requests_format])
        cls.launch_server()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        cls.output_capturer.stop()

    def test_enable_request_time_stats_logging(self):
        self.inference_once()

        content = self.output_capturer.get_all()
        source_name = "stdout"

        self.assertIn("Receive:", content, f"'Receive:' not found in {source_name}")
        self.assertIn("Finish:", content, f"'Finish:' not found in {source_name}")

class TestRequestLoggerJson(TestNPUEnableRequestTimeStatsLogging):
    log_requests_format = "json"

    def test_enable_request_time_stats_logging(self):
        self.inference_once()

        content = self.output_capturer.get_all()
        source_name = "stdout"

        received_found = False
        finished_found = False
        for line in content.splitlines():
            if not line.strip() or not line.startswith("{"):
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            # rid = data.get("rid", "")
            # if rid.startswith(HEALTH_CHECK_RID_PREFIX):
            #     continue

            if data.get("event") == "request.received":
                self.assertIn("rid", data)
                self.assertIn("obj", data)
                received_found = True
            elif data.get("event") == "request.finished":
                self.assertIn("rid", data)
                self.assertIn("obj", data)
                self.assertIn("out", data)
                finished_found = True

        self.assertTrue(
            received_found, f"request.received event not found in {source_name}"
        )
        self.assertTrue(
            finished_found, f"request.finished event not found in {source_name}"
        )




if __name__ == "__main__":
    unittest.main()