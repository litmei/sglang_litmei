import os
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    QWEN3_32B_EAGLE3_WEIGHTS_PATH,
    QWEN3_32B_W8A8_MINDIE_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-2-npu-a3", nightly=True)

_ASCEND_BACKEND = "ascend"

# Common EAGLE3 arguments shared across tests.
# --speculative-draft-model-quantization unquant: draft head in full precision.
# --speculative-num-steps 4: draft head runs 4 auto-regressive steps per iteration.
# --speculative-eagle-topk 1: single beam; required by SpecV2 overlap scheduler.
# --speculative-num-draft-tokens 5: maximum draft tokens submitted for verification.
# --speculative-attention-mode decode: draft attention in single-token decode mode.
_COMMON_ARGS = [
    "--trust-remote-code",
    "--attention-backend",
    _ASCEND_BACKEND,
    "--quantization",
    "modelslim",
    "--disable-radix-cache",
    "--speculative-draft-model-quantization",
    "unquant",
    "--speculative-algorithm",
    "EAGLE3",
    "--speculative-draft-model-path",
    QWEN3_32B_EAGLE3_WEIGHTS_PATH,
    "--speculative-num-steps",
    "4",
    "--speculative-eagle-topk",
    "1",
    "--speculative-num-draft-tokens",
    "5",
    "--speculative-attention-mode",
    "decode",
    "--mem-fraction-static",
    "0.7",
    "--disable-cuda-graph",
    "--dtype",
    "bfloat16",
]

# Use 2-card tensor parallelism and dummy draft load format.
# --speculative-draft-load-format dummy: initializes draft weights with random values.
# --speculative-draft-model-revision is NOT passed here, testing the default behavior.
_SERVER_ARGS = _COMMON_ARGS + [
    "--tp-size",
    "2",
    "--speculative-draft-load-format",
    "dummy",
]


class TestNpuSpeculativeDraftParams(CustomTestCase):
    """
    Test --speculative-draft-load-format dummy with 2-card TP.

    Dummy format initializes draft weights with random values.
    Only verify server startup and inference returns non-empty content.
    No acceptance rate check because weights are random.

    [Test Category] Parameter & Multi-NPU
    [Test Target]
        --tp-size 2
        --speculative-draft-load-format dummy
        (implicit) --speculative-draft-model-revision default (not passed)
    [Model]
        Target: aleoyang/Qwen3-32B-w8a8-MindIE
        Draft: Qwen/Qwen3-32B-Eagle3 (dummy weights)
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.base_url = DEFAULT_URL_FOR_TEST
        env = os.environ.copy()
        env.update(
            {
                "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
                "SGLANG_ENABLE_SPEC_V2": "1",
            }
        )
        cls.process = popen_launch_server(
            QWEN3_32B_W8A8_MINDIE_WEIGHTS_PATH,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * 5,
            other_args=_SERVER_ARGS,
            env=env,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        kill_process_tree(cls.process.pid)

    def test_dummy_format_with_tp2(self):
        """
        Send an inference request and verify:
        1. HTTP 200
        2. Response contains valid structure
        3. Generated content is non-empty and contains expected answer "Paris".
        """
        prompt = "What is the capital of France?"
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": QWEN3_32B_W8A8_MINDIE_WEIGHTS_PATH,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 64,
                "temperature": 0,
            },
            timeout=300,
        )

        # Assert HTTP status
        self.assertEqual(
            response.status_code, 200, f"Request failed with status {response.status_code}: {response.text}"
        )

        result = response.json()
        # Assert response structure
        self.assertIn("choices", result, "Response missing 'choices' field")
        self.assertGreater(len(result["choices"]), 0, "No choices in response")

        content = result["choices"][0]["message"]["content"]
        # Assert content is non-empty
        self.assertGreater(len(content.strip()), 0, "Generated content is empty")

        # Assert expected answer is present
        self.assertIn(
            "Paris",
            content,
            f"Expected 'Paris' in response, but got: {content[:200]}",
        )

        print(f"Q: {prompt}")
        print(f"A: {content}")


if __name__ == "__main__":
    unittest.main()