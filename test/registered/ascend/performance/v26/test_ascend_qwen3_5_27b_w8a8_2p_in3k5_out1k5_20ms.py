import unittest

from sglang.test.ascend.e2e.test_npu_performance_utils import (
    QWEN3_5_27B_W8A8_MODEL_PATH,
    TestAscendPerformanceTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=1800,
    suite="nightly-4-npu-a3",
    nightly=True,
    disabled="Currently it is executed by the npu performance workflow.",
)

ENVS = {
    "ASCEND_LAUNCH_BLOCKING": "1",
    "STREAMS_PER_DEVICE": "32",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "32",
    "HCCL_BUFFSIZE": "3000",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "SGLANG_NPU_PROFILING": "0",
    "SGLANG_NPU_PROFILING_STAGE": "prefill",
    "DEEPEP_NORMAL_LONG_SEQ_ROUND": "32",
    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "3584",
    "ASCEND_MF_STORE_URL": "tcp://127.0.0.1:24669",
    "SGLANG_DISAGGREGATION_WAITING_TIMEOUT": "3600",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
}

OTHER_ARGS = [
    "--trust-remote-code",
    "--nnodes",
    "1",
    "--node-rank",
    "0",
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--tp-size",
    4,
    "--chunked-prefill-size",
    -1,
    "--max-prefill-tokens",
    100000,
    "--disable-radix-cache",
    "--max-total-tokens",
    800000,
    "--max-running-requests",
    32,
    "--mem-fraction-static",
    0.75,
    "--cuda-graph-bs",
    2,
    4,
    6,
    8,
    10,
    16,
    24,
    28,
    32,
    48,
    56,
    64,
    96,
    112,
    "--enable-multimodal",
    "--quantization",
    "modelslim",
    "--mm-attention-backend",
    "ascend_attn",
    "--dtype",
    "bfloat16",
    "--mamba-ssm-dtype",
    "bfloat16",
    "--speculative-algorithm",
    "NEXTN",
    "--speculative-num-steps",
    3,
    "--speculative-eagle-topk",
    1,
    "--speculative-num-draft-tokens",
    4,
]


class TestQwen3527B(TestAscendPerformanceTestCaseBase):
    model = QWEN3_5_27B_W8A8_MODEL_PATH
    # model = "/home/weights/Eco-Tech/Qwen3.5-27B-W8A8"
    # model = "/data/ascend-ci-share-pkking-sglang/modelscope/hub/models/Eco-Tech/Qwen3.5-27B-w8a8-mtp"
    other_args = OTHER_ARGS
    envs = ENVS
    dataset_name = "random"
    max_concurrency = 16
    num_prompts = int(max_concurrency) * 4
    input_len = 3500
    output_len = 1500
    random_range_ratio = 1
    tpot = 21
    output_token_throughput = 900

    def test_qwen3_5_27b(self):
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
