import os
import tempfile
import unittest
from pathlib import Path

from test_npu_logging import TestNPULoggingBase

from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_server,
)

register_npu_ci(est_time=100, suite="nightly-2-npu-a3", nightly=True)


class TestNPULoggingCase0(TestNPULoggingBase):
    """Test case class for validating specific scenarios of logging feature parameters (complementary to Case1/2/3).

    Core Functionality:
        This class tests the behavior of SGLang logging/metrics features under a specific parameter configuration set,
        working with Case1/2/3 to fully cover the parameter value range of the logging feature. The tested configuration includes:
        - Explicitly set parameters:
          --log-requests; --log-requests-level=0; --enable-metrics; --collect-tokens-histogram;
          --log-requests-target target_file_path_list; --gc-warning-threshold-secs 0.01;
        - Implicitly used parameters (default values, no custom configuration):
          --enable-metrics-for-all-scheduler (default False); --uvicorn-access-log-exclude-prefixes (default False);
          --bucket-time-to-first-token (default boundaries); --bucket-inter-token-latency (default boundaries); --bucket-e2e-request-latency (default boundaries);
          --prompt-tokens-buckets (default boundaries); --generation-tokens-buckets (default boundaries);

        Note:
        1. --tp-size=2 is configured to verify that only TP rank 0 metrics are recorded when --enable-metrics-for-all-scheduler=False
        2. --gc-warning-threshold-secs=0.01 (a tiny value) is set to ensure GC duration exceeds the alarm threshold.
        3. --log-requests, --enable-metrics, and --collect-tokens-histogram are feature-enabling parameters;
            their validity is verified indirectly through the behavior of other parameters (no dedicated validation required).

    [Test Category] Parameter
    [Test Target] --log-requests; --log-requests-level; --log-requests-target; --uvicorn-access-log-exclude-prefixes;
    --enable-metrics; --enable-metrics-for-all-scheduler;
    --bucket-time-to-first-token; --bucket-inter-token-latency; --bucket-e2e-request-latency;
    --collect-tokens-histogram; --prompt-tokens-buckets; --generation-tokens-buckets;
    --gc-warning-threshold-secs
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls.log_requests_level = 0
        # --log-requests-target supports single-level and multi-level directories.
        cls._temp_dir_obj = tempfile.TemporaryDirectory()
        cls.temp_dir = cls._temp_dir_obj.name
        cls.temp_multi_level_dir = os.path.join(cls.temp_dir, "level1")
        cls.temp_multi_level_dir = os.path.join(cls.temp_dir, "level2")
        cls.temp_multi_level_dir = os.path.join(cls.temp_dir, "level3")
        os.makedirs(cls.temp_level3_dir, exist_ok=True)
        target_config = ["stdout", cls.temp_dir, cls.temp_level3_dir]

        cls.other_args.extend(["--log-requests"])
        cls.other_args.extend(["--log-requests-level", str(cls.log_requests_level)])
        cls.other_args.extend(["--enable-metrics"])
        cls.other_args.extend(["--tp-size", 2])
        cls.other_args.extend(["--collect-tokens-histogram"])
        cls.other_args.extend(["--gc-warning-threshold-secs", "0.01"])
        cls.other_args.extend(["--log-requests-target", *target_config])

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.other_args,
            return_stdout_stderr=(cls.out_log_file, cls.err_log_file),
        )

    def test_logging_case_0(self):
        self._verify_inference()

        self._verify_log_requests_level(self.log_requests_level, self.out_log_file)

        self._verify_log_requests_target()

        # test --uvicorn-access-log-exclude-prefixes
        self._verify_log_exclude_prefixes(False, self.out_log_file)

        self._verify_enable_metrics_for_all_scheduler(False)

        self._verify_metrics_and_bucket_boundary(
            expected_time_to_first_token_bucket=self.default_time_to_first_token_bucket,
            expected_inter_token_latency_bucket=self.default_inter_token_latency_bucket,
            expected_e2e_request_latency_bucket=self.default_e2e_request_latency_bucket,
            expected_prompt_tokens_bucket=self.default_tokens_bucket,
            expected_generation_tokens_bucket=self.default_tokens_bucket,
        )

        self._verify_gc_warning_threshold(self.err_log_file)

    def _verify_log_requests_target(self):
        """Validate that request logs are correctly output to the target files configured via --log-requests-target."""
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
