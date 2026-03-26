import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=150, suite="nightly-1-npu-a3", nightly=True)


class TestNPUKVCacheDtype(CustomTestCase):
    """Testcase：Verify set --kv_cache_dtype is auto, bf16 or bfloat16, request inference successful.

    [Test Category] Parameter
    [Test Target] --kv_cache_dtype
    """

    model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
    kv_cache_dtype = "auto"

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--dtype",
            "auto",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--kv-cache-dtype",
            cls.kv_cache_dtype,
        ]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_dtype_options(self):
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
        self.assertEqual(response.status_code, 200)
        self.assertIn("Paris", response.text)


class TestNPUKVCacheDtypeBf16(TestNPUKVCacheDtype):
    kv_cache_dtype = "bf16"


class TestNPUKVCacheDtypeBfloat16(TestNPUKVCacheDtype):
    kv_cache_dtype = "bfloat16"


if __name__ == "__main__":
    unittest.main()
