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

register_npu_ci(est_time=200, suite="nightly-2-npu-a3", nightly=True)

_ASCEND_BACKEND = "ascend"

_SERVER_ARGS = [
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
    # --tp-size 2: shard the 32B model across 2 NPU cards; tests TP-aware
    # hidden state handoff between the EAGLE3 draft head and target model.
    "--tp-size",
    "2",
    "--mem-fraction-static",
    "0.7",
    "--disable-cuda-graph",
    "--dtype",
    "bfloat16",
]


class TestNpuSpeculativeMultiNpu(CustomTestCase):
    """
    [Test Category] Parameter
    [Test Target] --tp-size; --speculative-algorithm; --speculative-draft-model-path;
                  --speculative-num-steps; --speculative-eagle-topk;
                  --speculative-num-draft-tokens; --speculative-attention-mode;
                  --attention-backend
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

    def test_eagle3_multi_npu_inference(self):
        """
        Test steps:
          1. Send a single inference request to the 2-card EAGLE3-enabled server.
          2. Assert the response payload is structurally valid.
          3. Assert avg_spec_accept_length > 1.0 to confirm the EAGLE3 draft-verify
             hidden state handoff is correct across both TP ranks.
        """
        response = send_inference_request(
            self.base_url,
            QWEN3_32B_W8A8_MINDIE_WEIGHTS_PATH,
            "Explain the principles of quantum computing in simple terms.",
        )

        content = response["choices"][0]["message"]["content"]
        print(f"Q: Explain the principles of quantum computing in simple terms")
        print(f"A: {content}")

        self.assertIn("choices", response)
        self.assertGreater(len(response["choices"]), 0)
        self.assertGreater(len(response["choices"][0]["message"]["content"].strip()), 0)

        # avg_spec_accept_length > 1.0: TP shard misalignment or all-reduce errors
        # would collapse acceptance to near zero without raising any exception.
        assert_spec_decoding_active(self, self.base_url, threshold=1.0)


if __name__ == "__main__":
    unittest.main()
