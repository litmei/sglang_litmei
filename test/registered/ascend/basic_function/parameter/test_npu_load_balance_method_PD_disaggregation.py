import itertools
import json
import os
import unittest
from time import sleep
from types import SimpleNamespace
from urllib.parse import urlparse

import requests

from sglang.test.ascend.disaggregation_utils import TestDisaggregationBase
from sglang.test.ascend.test_ascend_utils import (
    QWEN3_30B_A3B_INSTRUCT_2507_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_pd_server,
)

register_npu_ci(est_time=200, suite="nightly-4-npu-a3", nightly=True)

load_balance_method_options = [
    "auto",
    "round_robin",
    "total_requests",
    "total_tokens",
    "follow_bootstrap_room"
]
all_params = list(itertools.product(load_balance_method_options, repeat=2))


class BaseTestNPULoadBalanceMethodDPDisaggregation(TestDisaggregationBase):
    """Testcase：Verify that the model accuracy did not decrease when --load-balance-method is set to round_robin, auto,
    total_requests, total_tokens or follow_bootstrap_room in PD disaggregation scenario

    [Test Category] Parameter
    [Test Target] --load-balance-method
    """

    params = ("auto", "auto")

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.prefill_load_balance_method, cls.decode_load_balance_method = cls.params
        cls.model = QWEN3_30B_A3B_INSTRUCT_2507_WEIGHTS_PATH
        os.environ["ASCEND_MF_STORE_URL"] = "tcp://127.0.0.1:24666"

        # Non blocking start servers
        cls.start_prefill()
        cls.start_decode()

        # Block until both
        cls.wait_server_ready(cls.prefill_url + "/health")
        cls.wait_server_ready(cls.decode_url + "/health")

        cls.launch_lb()
        cls.url = urlparse(cls.lb_url)

    @classmethod
    def start_prefill(cls):
        prefill_args = [
            "--disaggregation-mode",
            "prefill",
            "--tp-size",
            "2",
            "--enable-dp-attention",
            "--dp",
            "2",
            "--load-balance-method",
            cls.prefill_load_balance_method,
            "--disaggregation-transfer-backend",
            "ascend",
            "--disable-cuda-graph",
            "--attention-backend",
            "ascend",
            "--mem-fraction-static",
            0.8,
            "--dist-init-addr",
            "127.0.0.1:10100",
            "--base-gpu-id",
            4,
        ]

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
            2,
            "--tp-size",
            "2",
            "--enable-dp-attention",
            "--dp",
            "2",
            "--load-balance-method",
            cls.decode_load_balance_method,
            "--disaggregation-transfer-backend",
            "ascend",
            "--disable-cuda-graph",
            "--attention-backend",
            "ascend",
            "--mem-fraction-static",
            0.8,
            "--dist-init-addr",
            "127.0.0.1:10000",
        ]

        cls.process_decode = popen_launch_pd_server(
            cls.model,
            cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
        )

    def test_gsm8k(self):
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host=f"http://{self.url.hostname}",
            port=int(self.url.port),
        )

        metrics = run_eval_few_shot_gsm8k(args)
        self.assertGreaterEqual(
            metrics["accuracy"],
            # 0.95 with 0.02 tolerable fluctuation
            0.93,
        )

    def test_server_info(self):
        response = requests.get(f"{self.lb_url}/get_server_info")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.text)
        self.assertEqual(
            (
                "follow_bootstrap_room"
                if self.prefill_load_balance_method == "auto"
                else self.prefill_load_balance_method
            ),
            data.get("prefill")[0].get("load_balance_method")
        )
        self.assertEqual(
            (
                "round_robin"
                if self.decode_load_balance_method == "auto"
                else self.decode_load_balance_method
            ),
            data.get("decode")[0].get("load_balance_method")
        )

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("ASCEND_MF_STORE_URL")
        super().tearDownClass()
        # wait for server release source
        sleep(10)


for index, param_tuple in enumerate(all_params):
    if param_tuple == BaseTestNPULoadBalanceMethodDPDisaggregation.params:
        continue

    prefill_load_balance_method, decode_load_balance_method = param_tuple
    class_name = f"Test_{index:02d}_prefill_{prefill_load_balance_method}_decode_{decode_load_balance_method}"
    new_class = type(
        class_name,
        (BaseTestNPULoadBalanceMethodDPDisaggregation,),
        {"params": param_tuple},
    )
    globals()[class_name] = new_class

if __name__ == "__main__":
    unittest.main()
