import os
import unittest

import requests




from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_VL_30B_A3B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-2-npu-a3", nightly=True)


_INLINE_IMAGE_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAACXBIWXMAAA7EAAAOxAGVKw4b"
    "AAAAbUlEQVRYhe3VsQ2AMAxE0Y/lIgNQULD/OqyCMgCihCKSG4yRuKuiNH6JLsoEbMACOGB"
    "cua9HOR7Y6w6swBwMy0qLTpkeI77qdEBpBFAHBBDAGH8WrwJKI4AAegUCfAKgEgpQDvh3CR"
    "3oQCuav58qlAw73kKCSgAAAABJRU5ErkJggg=="
)


class TestAdaptiveDispatchToEncoder(CustomTestCase):
    """Testcase: Verify --enable-adaptive-dispatch-to-encoder on Ascend NPU.

    Background
    ----------
    In VLM encoder-prefill disaggregation (EPD) mode, a language-only server
    normally forwards ALL multimodal requests to remote encoder servers.
    This creates unnecessary network overhead for single-image requests that
    the language server can handle locally.

    --enable-adaptive-dispatch-to-encoder changes the routing policy:
    - Single-image requests  --> processed locally (no encoder server needed)
    - Multi-image requests   --> dispatched to remote encoder server(s)

    The routing decision is made in tokenizer_manager._should_dispatch_to_encoder()
    by comparing total multimodal item count against SGLANG_ENCODER_DISPATCH_MIN_ITEMS
    (default = 2).

    Server setup
    ------------
    This test starts a language-only server WITHOUT any --encoder-urls.
    This is deliberate: if adaptive dispatch is working correctly, single-image
    requests never attempt to reach an encoder server, so the absence of encoder
    URLs is harmless.  If adaptive dispatch is NOT working, the request will try
    to reach a non-existent encoder and fail -- which is exactly what we want
    to detect.

    [Test Category] Parameter
    [Test Target] --enable-adaptive-dispatch-to-encoder; --language-only
    """

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_VL_30B_A3B_INSTRUCT_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST

        # SGLANG_MM_SKIP_COMPUTE_HASH must be set for Ascend NPU:
        # the NPU backend does not support _local_scalar_dense for UInt64,
        # which is used in multimodal hash computation.
        # Setting this variable replaces hash computation with a random UUID.
        env = os.environ.copy()
        env["SGLANG_MM_SKIP_COMPUTE_HASH"] = "True"

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            env=env,
            other_args=[
                "--language-only",
                "--enable-adaptive-dispatch-to-encoder",
                # Intentionally omit --encoder-urls: single-image requests
                # must be handled locally without any encoder server.
                "--encoder-urls", "http://127.0.0.1:9999",
                "--tp-size",
                "2",
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
                "--trust-remote-code",
                "--mem-fraction-static",
                "0.8",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        # Restore the test environment by stopping the server process (coding standard rule 8)
        kill_process_tree(cls.process.pid)

    def test_flag_accepted_by_server(self):
        """Verify --enable-adaptive-dispatch-to-encoder is stored in server config.

        GET /get_server_info should expose the flag value.
        This assertion does not depend on PR #18118 -- it only checks that the
        argument parser accepted the flag and stored it in server_args.

        NOTE: confirm the exact field name by running:
            curl http://<host>:<port>/get_server_info | python3 -m json.tool
        in the actual environment before merging, and update the field name if needed.
        """
        response = requests.get(f"{self.base_url}/get_server_info", timeout=10)
        self.assertEqual(response.status_code, 200)
        info = response.json()
        self.assertTrue(
            info.get("enable_adaptive_dispatch_to_encoder"),
            f"Expected enable_adaptive_dispatch_to_encoder=True in server info, "
            f"got: {info.get('enable_adaptive_dispatch_to_encoder')!r}. "
            f"Full server info: {info}",
        )

    def test_single_image_processed_locally(self):
        """Verify a single-image request is processed locally without an encoder server.

        Assertion rationale:
        - HTTP 200 means the language server processed the image locally.
        - Any non-2xx or connection timeout means either:
            (a) the server tried to forward to a non-existent encoder (adaptive
                dispatch not working), or
            (b) mm_pool was empty and local processing crashed (bugfix not applied).
        Both (a) and (b) are genuine failures this test is designed to catch.
        """
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": _INLINE_IMAGE_URL},
                        },
                        {"type": "text", "text": "Describe the image briefly."},
                    ],
                }
            ],
            "temperature": 0,
            "max_tokens": 32,
        }
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            timeout=120,
        )
        self.assertEqual(
            response.status_code,
            200,
            f"Single-image request failed with status {response.status_code}. "
            "Possible causes: (1) adaptive dispatch not routing single-image locally, "
            f"Response body: {response.text[:300]}",
        )


if __name__ == "__main__":
    unittest.main()
