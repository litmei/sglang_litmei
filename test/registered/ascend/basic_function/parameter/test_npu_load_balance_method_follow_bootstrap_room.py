import os
import unittest

import requests

from sglang.test.ascend.disaggregation_utils import TestDisaggregationBase
from sglang.test.ascend.test_ascend_utils import (
    QWEN3_30B_A3B_INSTRUCT_2507_WEIGHTS_PATH
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_pd_server,
)

register_npu_ci(est_time=400, suite="nightly-4-npu-a3", nightly=True)


class TestNPULoadBalanceMethodFollowBootstrapRoom(TestDisaggregationBase):
    """Testcase：Verify that the inference is successful when --load-balance-method is set to follow_bootstrap_room.

    [Test Category] Parameter
    [Test Target] --load-balance-method
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = QWEN3_30B_A3B_INSTRUCT_2507_WEIGHTS_PATH
        os.environ["ASCEND_MF_STORE_URL"] = "tcp://127.0.0.1:24666"

        # Non blocking start servers
        cls.start_prefill()
        cls.start_decode()

        # Block until both
        cls.wait_server_ready(cls.prefill_url + "/health")
        cls.wait_server_ready(cls.decode_url + "/health")

        cls.launch_lb()

    @classmethod
    def start_prefill(cls):
        prefill_args = [
            "--disaggregation-mode",
            "prefill",
            "--tp-size",
            "2",
            "--enable-dp-attention",
            "--dp",
            "2",
            "--load-balance-method",
            "follow_bootstrap_room",
            "--disaggregation-transfer-backend",
            "ascend",
            "--disable-cuda-graph",
            "--attention-backend",
            "ascend",
            "--mem-fraction-static",
            0.8,
            "--dist-init-addr",
            "127.0.0.1:10100",
            "--base-gpu-id",
            4,
        ]

        cls.process_prefill = popen_launch_pd_server(
            cls.model,
            cls.prefill_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=prefill_args,
        )

    @classmethod
    def start_decode(cls):
        decode_args = [
            "--disaggregation-mode",
            "decode",
            "--base-gpu-id",
            2,
            "--tp-size",
            "2",
            "--enable-dp-attention",
            "--dp",
            "2",
            "--load-balance-method",
            "auto",
            "--disaggregation-transfer-backend",
            "ascend",
            "--disable-cuda-graph",
            "--attention-backend",
            "ascend",
            "--mem-fraction-static",
            0.8,
            "--dist-init-addr",
            "127.0.0.1:10000",
        ]

        cls.process_decode = popen_launch_pd_server(
            cls.model,
            cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
        )

    def test_inference(self, max_new_tokens=32):
        """Send a basic inference request to test inference function."""
        response = requests.post(
            f"{self.lb_url}/generate",
            json={
                "text": "What is the capital of France?",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_new_tokens,
                },
            },
            timeout=60,
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("Paris", response.text)
        return response.text

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("ASCEND_MF_STORE_URL")
        super().tearDownClass()


if __name__ == "__main__":
    unittest.main()
