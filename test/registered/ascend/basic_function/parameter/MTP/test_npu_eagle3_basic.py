import os
import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    QWEN3_32B_EAGLE3_WEIGHTS_PATH,
    QWEN3_32B_W8A8_MINDIE_WEIGHTS_PATH,
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

register_npu_ci(est_time=200, suite="nightly-1-npu-a3", nightly=True)

_ASCEND_BACKEND = "ascend"

_SERVER_ARGS = [
    "--trust-remote-code",
    "--attention-backend",
    _ASCEND_BACKEND,
    "--quantization",
    "modelslim",
    "--disable-radix-cache",
    # --speculative-draft-model-quantization unquant: draft head loaded in full precision.
    "--speculative-draft-model-quantization",
    "unquant",
    "--speculative-algorithm",
    "EAGLE3",
    "--speculative-draft-model-path",
    QWEN3_32B_EAGLE3_WEIGHTS_PATH,
    # --speculative-num-steps 4: draft head runs 4 auto-regressive steps per iteration.
    "--speculative-num-steps",
    "4",
    # --speculative-eagle-topk 1: single beam; required by SpecV2 overlap scheduler.
    "--speculative-eagle-topk",
    "1",
    # --speculative-num-draft-tokens 5: maximum draft tokens submitted for verification.
    "--speculative-num-draft-tokens",
    "5",
    # --speculative-attention-mode decode: draft attention in single-token decode mode.
    "--speculative-attention-mode",
    "decode",
    "--tp-size",
    "1",
    "--mem-fraction-static",
    "0.7",
    "--disable-cuda-graph",
    "--dtype",
    "bfloat16",
]


class TestNpuEagle3Basic(CustomTestCase):
    """
    [Test Category] Parameter
    [Test Target] --speculative-algorithm; --speculative-draft-model-path;
                  --speculative-num-steps; --speculative-eagle-topk;
                  --speculative-num-draft-tokens; --speculative-attention-mode;
                  --speculative-draft-model-quantization; --attention-backend
    [Model] aleoyang/Qwen3-32B-w8a8-MindIE; Qwen/Qwen3-32B-Eagle3
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
            QWEN3_32B_W8A8_MINDIE_WEIGHTS_PATH,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * 5,
            other_args=_SERVER_ARGS,
            env=env,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        kill_process_tree(cls.process.pid)

    def test_eagle3_basic_inference(self):
        """
        Test steps:
          1. Send a single inference request to the EAGLE3-enabled server.
          2. Assert the response payload is structurally valid.
          3. Assert avg_spec_accept_length > 1.0 (multi-token acceptance confirmed).
        """
        response = send_inference_request(
            self.base_url,
            QWEN3_32B_W8A8_MINDIE_WEIGHTS_PATH,
            "List 3 countries and their capitals.",
        )

        content = response["choices"][0]["message"]["content"]
        print(f"Q: List 3 countries and their capitals")
        print(f"A: {content}")

        self.assertIn("choices", response)
        self.assertGreater(len(response["choices"]), 0)
        self.assertGreater(len(response["choices"][0]["message"]["content"].strip()), 0)

        # avg_spec_accept_length > 1.0: more than one draft token accepted per
        # target forward pass, confirming the EAGLE3 pipeline is genuinely active.
        assert_spec_decoding_active(self, self.base_url, threshold=1.0)


if __name__ == "__main__":
    unittest.main()
