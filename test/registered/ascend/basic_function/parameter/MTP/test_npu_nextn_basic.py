import os
import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    DEEPSEEK_V3_2_W8A8_WEIGHTS_PATH,
    assert_spec_decoding_active,
    send_inference_request,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-16-npu-a3", nightly=True)

_ASCEND_BACKEND = "ascend"

_SERVER_ARGS = [
    "--trust-remote-code",
    "--attention-backend",
    _ASCEND_BACKEND,
    "--disable-radix-cache",
    # Use NEXTN algorithm (MTP) – no draft model needed
    "--speculative-algorithm",
    "EAGLE",
    # Number of auto-regressive steps per iteration (tune based on GPU memory)
    "--speculative-num-steps",
    "3",  # Lower for memory, increase for speed
    # Branching factor (1 = greedy, >1 for speculative sampling, SPEC-V2 now only support 1)
    "--speculative-eagle-topk",
    "1",  # Branching factor
    # Maximum draft tokens to verify per step
    "--speculative-num-draft-tokens",
    "5", # Lower for memory, increase for speed
    "--speculative-attention-mode",
    "decode",
    "--tp-size",
    "16",   # Tensor parallelism – adjust according to available NPUs）
    "--mem-fraction-static",
    "0.85",  # adjust according to available HBM
    "--disable-cuda-graph",
    "--dtype",
    "bfloat16",
]

class TestNpuNextnDeepSeek(CustomTestCase):
    """
    [Test Category] Parameter
    [Test Target] --speculative-algorithm; --speculative-num-steps;
                  --speculative-eagle-topk; --speculative-num-draft-tokens;
                  --speculative-attention-mode; --attention-backend
    [Model] DeepSeek-V3.2-W8A8 (vllm-ascend/DeepSeek-V3.2-W8A8)
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.base_url = DEFAULT_URL_FOR_TEST
        env = os.environ.copy()
        os.environ.update(
            {
                "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
                "SGLANG_ENABLE_SPEC_V2": "1",
            }
        )
        cls.process = popen_launch_server(
            DEEPSEEK_V3_2_W8A8_WEIGHTS_PATH,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * 5,
            other_args=_SERVER_ARGS,
            env=env,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        kill_process_tree(cls.process.pid)

    def test_nextn_basic_inference(self):
        """
        Test steps:
          1. Send a single inference request to the DeepSeek-V3.2 server.
          2. Assert the response payload is structurally valid.
          3. Assert avg_spec_accept_length > 1.0 (multi-token acceptance confirmed).
        """
        response = send_inference_request(
            self.base_url,
            DEEPSEEK_V3_2_W8A8_WEIGHTS_PATH,
            "Explain quantum computing in simple terms.",
        )

        content = response["choices"][0]["message"]["content"]
        print(f"Q: Explain quantum computing in simple terms.")
        print(f"A: {content}")

        self.assertIn("choices", response)
        self.assertGreater(len(response["choices"]), 0)
        self.assertGreater(len(content.strip()), 0)

        assert_spec_decoding_active(self, self.base_url, threshold=1.0)


if __name__ == "__main__":
    unittest.main()
