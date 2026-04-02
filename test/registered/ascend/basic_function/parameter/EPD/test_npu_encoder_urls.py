import unittest

import requests

import time
import os

# ============ [Local path override - for local debugging only] ============
LOCAL_MODEL_WEIGHTS_DIR = "/home/weights"
import sglang.test.ascend.test_ascend_utils as _utils
_utils.MODEL_WEIGHTS_DIR = LOCAL_MODEL_WEIGHTS_DIR
_utils.HF_MODEL_WEIGHTS_DIR = LOCAL_MODEL_WEIGHTS_DIR
_utils.QWEN3_VL_30B_A3B_INSTRUCT_WEIGHTS_PATH = os.path.join(
    LOCAL_MODEL_WEIGHTS_DIR, "Qwen/Qwen3-VL-30B-A3B-Instruct"
)
# =========================================================================

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_VL_30B_A3B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=300, suite="nightly-2-npu-a3", nightly=True)

# Placeholder encoder URLs used to validate parameter parsing.
# These do not need to point to live encoder servers; the language-only server
# only connects to encoder URLs when it receives a multimodal request.
# Using loopback addresses ensures no external dependency.
_ENCODER_URL_PRIMARY = "http://127.0.0.1:8100"
_ENCODER_URL_SECONDARY = "http://127.0.0.1:8101"


class TestEncoderUrlsBase(CustomTestCase):
    __test__ = False


    _next_port = 21000
    """Testcase: Verify --encoder-urls parameter is accepted at server startup on Ascend NPU.

    --encoder-urls specifies a list of encoder server addresses used in VLM encoder
    disaggregation. The language-only server uses these URLs to forward visual encoding
    requests to remote encoder-only servers.

    The server does not attempt live connections at startup; connections are made lazily
    when multimodal requests arrive. Therefore a healthy server confirms the parameter
    was parsed and stored correctly without requiring live encoder servers.

    [Test Category] Parameter
    [Test Target] --encoder-urls; --language-only; --encoder-transfer-backend
    """

    # Subclasses set the encoder-transfer-backend to use alongside encoder-urls.
    # Supported values: "zmq_to_scheduler", "zmq_to_tokenizer", "mooncake"
    transfer_backend: str = None

    @classmethod
    def setUpClass(cls):
        env = os.environ.copy()
        env["SGLANG_MM_SKIP_COMPUTE_HASH"] = "True"
        cls.model = QWEN3_VL_30B_A3B_INSTRUCT_WEIGHTS_PATH
        cls.port = cls._next_port
        cls._next_port += 1  # 下一个子类使用新端口
        cls.base_url = f"http://127.0.0.1:{cls.port}"
        other_args = [
            # language-only is the natural counterpart for --encoder-urls:
            # the language server receives embeddings from a remote encoder server
            "--language-only",
            "--port", str(cls.port),
            "--encoder-urls",
            _ENCODER_URL_PRIMARY,
            _ENCODER_URL_SECONDARY,
            "--encoder-transfer-backend",
            cls.transfer_backend,
            "--attention-backend",
            "ascend",
            "--tp-size",
            "2",
            "--base-gpu-id",
            "14",
            "--disable-cuda-graph",
            "--trust-remote-code",
            "--mem-fraction-static",
            "0.8",
        ]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            env=env,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        # Restore the test environment by stopping the server process (coding standard rule 8)
        kill_process_tree(cls.process.pid)
        time.sleep(2)

    def test_server_health(self):
        """Verify language-only server with encoder-urls is healthy.

        HTTP 200 from /health confirms:
        1. --encoder-urls values were accepted by the parameter parser.
        2. --encoder-transfer-backend was accepted alongside --encoder-urls.
        3. The server initialized without attempting live connections to the listed URLs.
        """
        response = requests.get(f"{self.base_url}/health", timeout=10)
        self.assertEqual(
            response.status_code,
            200,
            f"Language-only server with --encoder-urls and "
            f"--encoder-transfer-backend={self.transfer_backend} "
            "failed health check; parameter combination may have been rejected.",
        )


class TestEncoderUrlsOnly(TestEncoderUrlsBase):
    """Testcase 4.1: Verify --encoder-urls is accepted with default transfer backend.

    Uses zmq_to_scheduler (the default) to isolate the encoder-urls parameter.
    """

    transfer_backend = "zmq_to_scheduler"


class TestEncoderUrlsWithZmqToTokenizer(TestEncoderUrlsBase):
    """Testcase 4.2: Verify --encoder-urls combined with --encoder-transfer-backend=zmq_to_tokenizer.
    zmq_to_tokenizer is the supported transfer backend for NPU deployments.
    This case verifies the combination is accepted correctly.
    """
    transfer_backend = "zmq_to_tokenizer"


if __name__ == "__main__":
    unittest.main()
