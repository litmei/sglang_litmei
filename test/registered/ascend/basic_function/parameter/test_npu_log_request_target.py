import os
import tempfile
import unittest
from pathlib import Path

import requests

from sglang.test.ascend.test_npu_logging import TestNPULoggingBase
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=100, suite="nightly-1-npu-a3", nightly=True)


class TestNPULogRequestTarget(TestNPULoggingBase):
    """Test case class for verifying that logs are stored in the target path by setting --log-requests-target

    [Test Category] Parameter
    [Test Target] --log-requests-target;
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._temp_dir_obj = tempfile.TemporaryDirectory()
        cls.temp_dir = cls._temp_dir_obj.name
        cls.temp_multi_level_dir = os.path.join(cls.temp_dir, "level1", "level2", "level3")
        os.makedirs(cls.temp_multi_level_dir, exist_ok=True)
        cls.other_args.extend(
            ["--log-requests-target", "stdout", cls.temp_dir, cls.temp_multi_level_dir]
        )
        cls.launch_server()

    def test_log_requests_target(self):
        """Validate that request logs are correctly output to the target files configured via --log-requests-target."""
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": "What is the capital of France?",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
            },
            timeout=60,
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("Paris", response.text)

        log_files = list(Path(self.temp_dir).glob("*.log"))
        self.assertGreater(len(log_files), 0)

        file_content = log_files[0].read_text()
        self.assertIn("Receive:", file_content)
        self.assertIn("Finish:", file_content)

        log_files = list(Path(self.temp_multi_level_dir).glob("*.log"))
        self.assertGreater(len(log_files), 0)

        file_content = log_files[0].read_text()
        self.assertIn("Receive:", file_content)
        self.assertIn("Finish:", file_content)

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        cls._temp_dir_obj.cleanup()


if __name__ == "__main__":
    unittest.main()
