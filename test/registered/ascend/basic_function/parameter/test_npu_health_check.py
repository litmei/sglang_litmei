import unittest
import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=100, suite="nightly-1-npu-a3", nightly=True)


class TestHealthCheckEndpoint(CustomTestCase):
    """
    Testcase: Verify that the /health and /health_generate HTTP endpoints return
              HTTP 200 OK when the server is running normally.

    [Test Category] Parameter
    [Test Target] /health; /health_generate
    """

    @classmethod
    def setUpClass(cls):
        cls.model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_health_endpoint(self):
        """Verify /health endpoint is reachable and reports healthy status."""
        # Step 1: Send GET request to the /health liveness probe
        response = requests.get(f"{self.base_url}/health")
        # Checkpoint: a running server must return HTTP 200 on /health
        self.assertEqual(response.status_code, 200)

    def test_health_generate_endpoint(self):
        """Verify /health_generate endpoint returns 200 when the model is ready to serve."""
        # Step 1: Send GET request to /health_generate
        # This endpoint performs a lightweight internal generate call to confirm
        # the model is loaded and the inference pipeline is functional.
        response = requests.get(f"{self.base_url}/health_generate")
        # Checkpoint: HTTP 200 confirms the model is loaded and inference is ready
        self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
