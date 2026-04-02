import os
import unittest
from types import SimpleNamespace
from urllib.parse import urlparse
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
utils.LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH = os.path.join(
    LOCAL_MODEL_WEIGHTS_DIR, "LLM-Research/Llama-3.2-1B-Instruct"
)
# ====================================================
from sglang.test.ascend.disaggregation_utils import TestDisaggregationBase
from sglang.test.ascend.test_ascend_utils import (
    QWEN3_32B_EAGLE3_WEIGHTS_PATH,
    QWEN3_32B_W8A8_MINDIE_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_pd_server,
)

register_npu_ci(
    est_time=400,
    suite="nightly-8-npu-a3",
    nightly=True,
    disabled="run failed",
)


class TestAscendSpeculativeAttentionMode(TestDisaggregationBase):
    """Testcase: Verify that in the PD disaggregation + MTP scenario, the model inference accuracy remains
    uncompromised when the Prefill service is launched with the parameter --speculative-attention-mode decode
    and the Decode service is configured with --speculative-attention-mode prefill.

    [Test Category] Parameter
    [Test Target] --speculative-attention-mode
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = QWEN3_32B_W8A8_MINDIE_WEIGHTS_PATH
        cls.accuracy = 0.81
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.url = urlparse(DEFAULT_URL_FOR_TEST)
        os.environ["ASCEND_MF_STORE_URL"] = "tcp://127.0.0.1:24666"

        # Non blocking start servers
        cls.start_prefill()
        cls.start_decode()

        # Block until both
        cls.wait_server_ready(cls.prefill_url + "/health")
        cls.wait_server_ready(cls.decode_url + "/health")

        cls.launch_lb()

    @classmethod
    def start_prefill(cls):
        prefill_args = [
            "--base-gpu-id",
            "4",
            "--disaggregation-mode",
            "prefill",
            "--disaggregation-decode-tp",
            "2",
            "--disaggregation-transfer-backend",
            "ascend",
            "--disable-cuda-graph",
            "--trust-remote-code",
            "--attention-backend",
            "ascend",
            "--device",
            "npu",
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
            "--tp-size",
            "4",
            "--mem-fraction-static",
            "0.7",
            "--disable-cuda-graph",
            "--dtype",
            "bfloat16",
        ]
        cls.extra_envs = {
            "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
            "SGLANG_ENABLE_SPEC_V2": "1",
        }
        os.environ.update(cls.extra_envs)
        cls.process_prefill = popen_launch_pd_server(
            cls.model,
            cls.prefill_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=prefill_args,
        )

    @classmethod
    def start_decode(cls):
        decode_args = [
            "--disaggregation-mode",
            "decode",
            "--base-gpu-id",
            4,
            "--disaggregation-transfer-backend",
            "ascend",
            "--num-reserved-decode-tokens",
            128,
            "--disaggregation-decode-polling-interval",
            2,
            "--trust-remote-code",
            "--attention-backend",
            "ascend",
            "--device",
            "npu",
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
            "prefill",
            "--tp-size",
            "4",
            "--mem-fraction-static",
            "0.7",
            "--disable-cuda-graph",
            "--dtype",
            "bfloat16",
        ]
        cls.extra_envs = {
            "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
            "SGLANG_ENABLE_SPEC_V2": "1",
        }
        os.environ.update(cls.extra_envs)
        cls.process_decode = popen_launch_pd_server(
            cls.model,
            cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
        )

    def test_gsm8k(self):
        print(f"##=== Testing accuracy: {self.model} ===##")
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=1319,
            max_new_tokens=512,
            parallel=128,
            host=f"http://{self.url.hostname}",
            port=int(self.url.port),
        )

        metrics = run_eval_few_shot_gsm8k(args)
        self.assertGreaterEqual(
            metrics["accuracy"],
            self.accuracy,
        )

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("ASCEND_MF_STORE_URL")
        super().tearDownClass()


if __name__ == "__main__":
    unittest.main()
