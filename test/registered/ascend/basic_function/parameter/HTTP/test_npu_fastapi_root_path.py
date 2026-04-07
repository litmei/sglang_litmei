import unittest
from urllib.parse import urlparse

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_0_6B_WEIGHTS_PATH as MODEL_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=100, suite="nightly-1-npu-a3", nightly=True)


class TestAscendFastapiRootPath(CustomTestCase):
    """
    Testcase：Verify that the correct path is set in the openai.json file when --fastapi-root-path is set

    [Test Category] Parameter
    [Test Target] --fastapi-root-path
    """

    fastapi_root_path = "/sglang"

    @classmethod
    def setUpClass(cls):
        cls.model = MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.url = urlparse(cls.base_url)
        cls.common_args = [
            "--trust-remote-code",
            "--mem-fraction-static",
            0.8,
            "--attention-backend",
            "ascend",
            "--fastapi-root-path",
            cls.fastapi_root_path,
        ]

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.common_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_fastapi_root_path(self):
        response = self.send_request(f"{self.base_url}/generate")
        self.assertEqual(
            response.status_code, 200, "The request status code is not 200."
        )
        self.assertIn(
            "Paris", response.text, "The inference result does not include Paris."
        )

        response = requests.get(f"{self.base_url}/openapi.json")
        self.assertEqual(
            response.status_code, 200, "The request status code is not 200."
        )
        self.assertIn(
            self.fastapi_root_path,
            response.text,
            "The correct path is not set in the openai.",
        )

    def send_request(self, url):
        return requests.post(
            url,
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
            },
        )


class TestAscendFastapiRootPathMultiLevel(TestAscendFastapiRootPath):
    fastapi_root_path = "/test/fastapi/root/path"


if __name__ == "__main__":
    unittest.main()
