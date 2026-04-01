import unittest
from urllib.parse import urlparse
import os
import requests
# ============【本地路径覆盖 - 仅影响本文件】============
# 配置：服务器实际模型根目录
LOCAL_MODEL_WEIGHTS_DIR = "/home/weights"

# 在导入 test_ascend_utils 之后，立即覆盖其中的路径常量
import sglang.test.ascend.test_ascend_utils as utils

# 覆盖根目录常量（可选，如果其他代码依赖这个）
utils.MODEL_WEIGHTS_DIR = LOCAL_MODEL_WEIGHTS_DIR
utils.HF_MODEL_WEIGHTS_DIR = LOCAL_MODEL_WEIGHTS_DIR

# 覆盖 5 个模型路径常量（使用服务器实际路径）
utils.QWEN3_0_6B_WEIGHTS_PATH = os.path.join(
    LOCAL_MODEL_WEIGHTS_DIR, "Qwen/Qwen3-0.6B"
)
utils.QWEN3_30B_A3B_W8A8_WEIGHTS_PATH = os.path.join(
    LOCAL_MODEL_WEIGHTS_DIR, "Qwen/Qwen3-30B-A3B-W8A8"  # 注意：实际是大写 W8A8
)
utils.QWEN3_32B_EAGLE3_WEIGHTS_PATH = os.path.join(
    LOCAL_MODEL_WEIGHTS_DIR, "Qwen/Eagle3-Qwen3-32B-zh"  # 注意：实际目录名不同
)
utils.QWEN3_32B_W8A8_MINDIE_WEIGHTS_PATH = os.path.join(
    LOCAL_MODEL_WEIGHTS_DIR, "Qwen/Qwen3-32B-w8a8-MindIE"  # 注意：实际父目录是 Qwen 不是 aleoyang
)
utils.LLAMA_2_7B_WEIGHTS_PATH = os.path.join(
    LOCAL_MODEL_WEIGHTS_DIR, "llama-2-7b"
)
# ====================================================
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
