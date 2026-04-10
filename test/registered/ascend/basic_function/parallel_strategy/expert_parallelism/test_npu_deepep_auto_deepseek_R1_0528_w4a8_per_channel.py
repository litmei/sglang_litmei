import os
import unittest
from types import SimpleNamespace
from urllib.parse import urlparse

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    DEEPSEEK_R1_0528_W4A8_PER_CHANNEL_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.run_eval import run_eval
from sglang.test.few_shot_gsm8k import run_eval as run_eval_gsm8k
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="full-16-npu-a3", nightly=True)


class TestDeepEpAutoDeepseekR1(CustomTestCase):
    """Testcase: Verify the accuracy of DeepSeek-R1 model on MMLU and GSM8K tasks with --deepep-mode auto on Ascend backend.

    [Test Category] Parameter
    [Test Target] --moe-a2a-backend; --deepep-mode
    """

    accuracy = 0.96

    @classmethod
    def setUpClass(cls):
        cls.model = DEEPSEEK_R1_0528_W4A8_PER_CHANNEL_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.host = urlparse(cls.base_url).hostname
        cls.port = urlparse(cls.base_url).port
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * 5,
            other_args=[
                "--tp",
                "16",
                "--trust-remote-code",
                "--attention-backend",
                "ascend",
                "--device",
                "npu",
                "--quantization",
                "modelslim",
                "--watchdog-timeout",
                "9000",
                "--cuda-graph-bs",
                "4",
                "8",
                "20",
                "21",
                "22",
                "--mem-fraction-static",
                "0.78",
                "--max-running-requests",
                "352",
                "--disable-radix-cache",
                "--chunked-prefill-size",
                "-1",
                "--max-prefill-tokens",
                "1500",
                "--moe-a2a-backend",
                "deepep",
                "--deepep-mode",
                "auto",
                "--enable-dp-attention",
                "--dp-size",
                "16",
                "--enable-dp-lm-head",
                "--speculative-algorithm",
                "NEXTN",
                "--speculative-num-steps",
                "2",
                "--speculative-eagle-topk",
                "1",
                "--speculative-num-draft-tokens",
                "3",
                "--dtype",
                "bfloat16",
            ],
            env={
                "SGLANG_SET_CPU_AFFINITY": "1",
                "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
                "STREAMS_PER_DEVICE": "32",
                "SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE": "1",
                "SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES": "200",
                "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "88",
                "HCCL_BUFFSIZE": "1600",
                "DEEPEP_NORMAL_LONG_SEQ_ROUND": "10",
                "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "512",
                "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
                "SGLANG_NPU_USE_MLAPO": "1",
                "SGLANG_USE_FIA_NZ": "1",
                "SGLANG_ENABLE_SPEC_V2": "1",
                "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
                **os.environ,
            },

        )

    @classmethod
    def tearDownClass(cls):
        # Terminate the model server process after all tests in the class are completed
        kill_process_tree(cls.process.pid)

    def test_mmlu(self):
        # Test Scenario: Verify the model's performance on MMLU dataset (general knowledge evaluation)
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=64,
            num_threads=32,
        )

        # Execute MMLU evaluation and get metrics
        metrics = run_eval(args)
        self.assertGreater(metrics["score"], 0.5)

    def test_gsm8k(self):
        # Test Scenario: Verify the model's accuracy on GSM8K dataset (mathematical reasoning evaluation)
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=300,
            eval_name="gsm8k",
            host=self.host,
            port=self.port,
        )
        # Execute GSM8K evaluation and get metrics
        metrics = run_eval_gsm8k(args)
        # Assertion: The GSM8K accuracy is not lower than the preset threshold (0.96)
        self.assertGreaterEqual(
            metrics["accuracy"],
            self.accuracy,
            f'Accuracy of {self.model} is {str(metrics["accuracy"])}, is lower than {self.accuracy}',
        )


if __name__ == "__main__":
    unittest.main()
