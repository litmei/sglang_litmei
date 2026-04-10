import os
import re
import tempfile
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

register_npu_ci(est_time=400, suite="nightly-4-npu-a3", nightly=True)

_ASCEND_BACKEND = "ascend"


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

# --speculative-draft-load-format dummy: initializes draft weights with random values.
#   Valid options: auto (default), dummy.
#   dummy is used for profiling; output quality is not guaranteed.
# --speculative-draft-model-revision main: specifies a branch/tag/commit for the
#   draft model. Valid range: any valid git ref string; None means default branch.
_SERVER_ARGS = _COMMON_ARGS + [
    "--tp-size",
    "4",
    "--speculative-draft-load-format",
    "dummy",
    "--speculative-draft-model-revision",
    "main",
]


class TestNpuSpeculativeDraftParams(CustomTestCase):
    """Test --speculative-draft-load-format and --speculative-draft-model-revision
    with 4-card TP.

    [Test Category] Parameter & Multi-NPU
    [Test Target]
        --tp-size 4;
        --speculative-draft-load-format dummy;
        --speculative-draft-model-revision main
    [Model]
        Target: aleoyang/Qwen3-32B-w8a8-MindIE
        Draft: Qwen/Qwen3-32B-Eagle3 (dummy weights, revision=main)
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.out_log_file = tempfile.NamedTemporaryFile(
            mode="w+", encoding="utf-8", delete=False, suffix=".txt"
        )
        cls.err_log_file = tempfile.NamedTemporaryFile(
            mode="w+", encoding="utf-8", delete=False, suffix=".txt"
        )
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
            return_stdout_stderr=(cls.out_log_file, cls.err_log_file),
        )

    @classmethod
    def tearDownClass(cls) -> None:
        kill_process_tree(cls.process.pid)
        cls.out_log_file.close()
        cls.err_log_file.close()
        os.unlink(cls.out_log_file.name)
        os.unlink(cls.err_log_file.name)

    def test_draft_params_via_server_info(self):
        """Verify draft load format and revision are set correctly via /get_server_info."""
        # 1. Health check
        health_resp = requests.get(f"{self.base_url}/health", timeout=10)
        self.assertEqual(health_resp.status_code, 200)

        with open(self.out_log_file.name, "r", encoding="utf-8") as f:
            log_content = f.read()

        match = re.search(r"command=sglang serve\s+(.*?)(?:\n|$)", log_content)
        self.assertIsNotNone(
            match,
            f"Command line not found in server logs. Log snippet (first 500 chars): {log_content[:500]}",
        )
        cmd_line = match.group(1)

        self.assertIn("--speculative-draft-load-format dummy", cmd_line)
        self.assertIn("--speculative-draft-model-revision main", cmd_line)

        prompt = "What is the capital of France?"
        resp = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": QWEN3_32B_W8A8_MINDIE_WEIGHTS_PATH,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 64,
                "temperature": 0,
            },
            timeout=60,
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("choices", data)
        content = data["choices"][0]["message"]["content"]
        self.assertGreater(len(content.strip()), 0)

        self.assertIn(
            "paris",
            content.lower(),
            f"Expected 'Paris' in response, but got: {content[:200]}",
        )

        print(f"Q: {prompt}")
        print(f"A: {content}")


if __name__ == "__main__":
    unittest.main()
