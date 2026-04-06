import unittest

from sglang.test.ascend.e2e.test_npu_performance_utils import (
    KIMI_K2_5_W4A8_MODEL_PATH,
    TestAscendPerformanceTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=1800,
    suite="nightly-16-npu-a3",
    nightly=True,
    disabled="Currently it is executed by the npu performance workflow.",
)

KIMI_K2_5_ENVS = {
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "STREAMS_PER_DEVICE": "32",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "32",
    "HCCL_BUFFSIZE": "2100",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
}

KIMI_K2_5_OTHER_ARGS = [
    "--trust-remote-code",
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--quantization",
    "modelslim",
    "--dtype",
    "bfloat16",
    "--tp-size",
    16,
    "--mem-fraction-static",
    0.82,
    "--max-running-requests",
    256,
    "--chunked-prefill-size",
    65536,
    "--context-length",
    8192,
    "--max-prefill-tokens",
    16384,
    "--enable-multimodal",
    "--mm-attention-backend",
    "ascend_attn",
    "--sampling-backend",
    "ascend",
    "--enable-dp-attention",
    "--dp-size",
    16,
    "--enable-dp-lm-head",
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--cuda-graph-bs",
    1,
    2,
    4,
    8,
    16,
    "--disable-radix-cache",
    "--model-loader-extra-config",
    '{"enable_multithread_load": true}',
]


class TestKimiK25W4A8(TestAscendPerformanceTestCaseBase):
    model = KIMI_K2_5_W4A8_MODEL_PATH
    other_args = KIMI_K2_5_OTHER_ARGS
    envs = KIMI_K2_5_ENVS
    backend = "sglang"
    dataset_name = "random"
    max_concurrency = 128
    num_prompts = 128
    input_len = 3500
    output_len = 1500
    random_range_ratio = 1
    warmup_requests = 16
    tpot = 50
    output_token_throughput = 1545

    def test_kimi_k2_5_w4a8(self):
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
