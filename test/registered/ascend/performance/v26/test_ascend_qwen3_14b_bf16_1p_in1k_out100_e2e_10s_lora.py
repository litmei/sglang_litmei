import unittest

from sglang.test.ascend.e2e.test_npu_performance_utils import (
    QWEN3_14B_LORA_MODEL_PATH,
    QWEN3_14B_MODEL_PATH,
    TestAscendPerformanceTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=1800,
    suite="nightly-2-npu-a3",
    nightly=True,
    disabled="Currently it is executed by the npu performance workflow.",
)

QWEN3_14B_ENVS = {
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "ASCEND_USE_FIA": "0",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE": "1",
    "SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES": "200",
}

QWEN3_14B_OTHER_ARGS = [
    "--trust-remote-code",
    "--nnodes",
    "1",
    "--node-rank",
    "0",
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--disable-radix-cache",
    "--mem-fraction-static",
    0.85,
    "--cuda-graph-bs",
    96,
    "--tp-size",
    2,
    "--dp-size",
    1,
    "--sampling-backend",
    "ascend",
    "--max-running-requests",
    96,
    "--served-model-name",
    "Qwen3-14B",
    "--chunked-prefill-size",
    -1,
    "--enable-lora",
    "--lora-backend",
    "torch_native",
    "--lora-target-modules",
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "--lora-paths",
    f"Copilot={QWEN3_14B_LORA_MODEL_PATH}",
    "--max-lora-rank",
    8,
]

# ===============依赖PR:17794的合入,当前版本需要规避,回退这个PR:18046==============
import os


def delete_specific_lines(file_path, start_line, end_line, backup=True):
    if start_line > end_line or start_line < 1:
        raise ValueError("Invalid line range: start line must be ≤ end line and ≥ 1")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    if not os.access(file_path, os.R_OK | os.W_OK):
        raise PermissionError(f"No read/write permission for: {file_path}")

    if backup:
        backup_path = f"{file_path}.bak"
        with open(file_path, "r", encoding="utf-8") as src, open(
            backup_path, "w", encoding="utf-8"
        ) as dst:
            dst.write(src.read())
        print(f"Backup file created: {backup_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    new_lines = []
    for line_num, line_content in enumerate(lines, start=1):
        if not (start_line <= line_num <= end_line):
            new_lines.append(line_content)

    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    print(f"Successfully deleted lines {start_line}-{end_line} from {file_path}")


target_file = "/usr/local/python3.11.14/lib/python3.11/site-packages/sglang/srt/lora/lora_manager.py"
try:
    delete_specific_lines(target_file, 157, 160)
except Exception as e:
    print(f"Operation failed: {e}")
# ===============依赖PR:17794的合入,当前版本需要规避,回退这个PR:18046==============


class TestQwen14B(TestAscendPerformanceTestCaseBase):
    model = QWEN3_14B_MODEL_PATH
    other_args = QWEN3_14B_OTHER_ARGS
    envs = QWEN3_14B_ENVS
    dataset_name = "random"
    backend = "sglang-oai-chat"
    num_prompts = 1000
    input_len = 1024
    output_len = 100
    random_range_ratio = 1
    request_rate = 7
    seed = 1000
    mean_e2e_latency = 10000
    output_token_throughput = 682

    def test_qwen3_14b(self):
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
