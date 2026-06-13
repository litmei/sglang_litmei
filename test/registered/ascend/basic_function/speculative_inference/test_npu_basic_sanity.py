"""Stage-a basic sanity: small-but-broad server smoke that downstream
stages depend on. Multiple sanity-kit mixins driving one shared server,
covering protocol, decode correctness, scheduler stress, occupancy, and
hellaswag accuracy."""

import unittest

from sglang.srt.utils import is_hip, kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_8B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.kits.basic_api_contract_kit import BasicAPIContractMixin
from sglang.test.kits.basic_decode_correctness_kit import BasicDecodeCorrectnessMixin
from sglang.test.kits.basic_scheduler_stress_kit import BasicSchedulerStressMixin
from sglang.test.kits.fwd_occupancy_kit import FwdOccupancyMixin
from sglang.test.kits.hellaswag_kit import HellaswagMixin
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=200, suite="full-1-npu-a3", nightly=True)



class TestBasicSanity(
    BasicAPIContractMixin,
    BasicDecodeCorrectnessMixin,
    BasicSchedulerStressMixin,
    FwdOccupancyMixin,
    HellaswagMixin,
    CustomTestCase,
):
    served_model_name = QWEN3_8B_WEIGHTS_PATH
    # 5090 + Llama-3.1-8B single-batch decode with overlap scheduler +
    # cuda graph measured ~99 median in CI; async-assert probes are off in
    # base-a, so the threshold can sit right under the measured median.
    # AMD also measures ~99 but with less headroom; keep ~1pp of margin
    # there so small per-step changes don't flake the gate.
    fwd_occupancy_threshold = 98.0 if is_hip() else 99.0

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            QWEN3_8B_WEIGHTS_PATH,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--cuda-graph-max-bs",
                "4",
                "--mem-fraction-static",
                "0.7",
                "--enable-metrics",
                "--attention-backend",
                "ascend",
            ],
            env={"SGLANG_ENABLE_METRICS_DEVICE_TIMER": "1"},
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


if __name__ == "__main__":
    unittest.main()
