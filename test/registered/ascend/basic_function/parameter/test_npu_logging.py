import os
import re
import tempfile
import threading
import unittest
from time import sleep

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
)


class TestNPULoggingBase(CustomTestCase):
    """Testcase：Verify the correct functionality of parameters in the logging feature.

    [Test Category] Parameter
    [Test Target] --log-requests; --log-requests-level; --log-requests-target; --uvicorn-access-log-exclude-prefixes;
    --enable-metrics; --enable-metrics-for-all-scheduler;
    --bucket-time-to-first-token; --bucket-inter-token-latency; --bucket-e2e-request-latency;
    --collect-tokens-histogram; --prompt-tokens-buckets; --generation-tokens-buckets;
    --tokenizer-metrics-custom-labels-header; --tokenizer-metrics-allowed-custom-labels;
    --gc-warning-threshold-secs
    """

    @staticmethod
    def get_lines_with_keyword(filename, keyword):
        """Find and return lines matching a regex keyword from a specified file, with line numbers and content.

        Function Description:
            Reads the target file line by line, uses the input keyword as a regular expression pattern to match each line's content.
            For each line that matches the regex pattern, encapsulates the line number (1-indexed) and content into a dictionary,
            and finally returns a list of dictionaries containing all matched lines.

        Args:
            filename (str): Path to the file to be read
            keyword (str): Regular expression pattern for matching

        Returns:
            List[Dict[str, Union[str, int]]]
                List of dictionaries for matched lines, each dictionary contains two key-value pairs:
                - "line_number": int - Line number of the matched line (starts from 1)
                - "content": str - Full text content of the matched line
        """
        results = []
        try:
            with open(filename, "r", encoding="utf-8") as file:
                for line_num, line in enumerate(file, 1):
                    if re.match(keyword, line):
                        results.append(
                            {
                                "line_number": line_num,
                                "content": line.strip(),
                            }
                        )
            return results
        except Exception as e:
            print(f"error:{e}")
            return []

    @classmethod
    def setUpClass(cls):
        cls.model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.other_args = [
            "--trust-remote-code",
            "--mem-fraction-static",
            "0.8",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
        ]
        cls.out_log_file_obj = tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8", delete=False, suffix=".txt")
        cls.out_log_name = cls.out_log_file_obj.name
        cls.out_log_file = cls.out_log_file_obj
        cls.err_log_file_obj = tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8", delete=False, suffix=".txt")
        cls.err_log_name = cls.err_log_file_obj.name
        cls.err_log_file = cls.err_log_file_obj
        cls.test_prompt = "What is the capital of France?"
        cls.expected_output = "Paris"

        cls.prepare_args_related_data()

        cls.process = None

    @classmethod
    def tearDownClass(cls):
        if cls.process:
            kill_process_tree(cls.process.pid)
        cls.out_log_file.close()
        os.remove(cls.out_log_name)
        cls.err_log_file.close()
        os.remove(cls.err_log_name)

    @classmethod
    def prepare_args_related_data(cls):
        """Initialize and configure static data for validating the behavior of logging feature.

        Core Purpose:
            This class method sets up regex patterns, keyword identifiers, and default/custom bucket boundaries for
            validating the behavior of logging feature.

        Configured Parameter Data Categories:
            1. --log-requests / --log-requests-level: Regex patterns and keywords for validating request log verbosity
               - Base log pattern for request completion entries
               - Level-specific regex patterns (0-3) for log content validation
               - Keywords to extract token ID arrays from logs
            2. --uvicorn-access-log-exclude-prefixes: List of path prefixes to exclude from Uvicorn access logs
            3. --enable-metrics: Default/custom bucket boundaries for latency metrics (time-to-first-token, inter-token, E2E)
            4. --collect-tokens-histogram: Default/custom bucket boundaries for token count histograms (prompt/generation tokens)
               - Includes Two-Sided Exponential (TSE) bucket strategy configuration
            5. --tokenizer-metrics-custom-labels-header / --tokenizer-metrics-allowed-custom-labels:
               HTTP header and label name for custom metrics labeling
        """
        # --------------------------------------------------------------------------
        # 1. --log-requests / --log-requests-level Configuration
        # --------------------------------------------------------------------------
        # Base regex: Matches minimum required content in request completion logs (--log-requests=True)
        # Applies to all --log-requests-level values (0-3)
        cls.message = (
            r".*Finish: obj=GenerateReqInput\(.*rid='\w+', http_worker_ipc=None, .*"
        )

        # Level-specific regex patterns for --log-requests-level (0-3)
        # Each pattern validates that logs include level-appropriate content
        cls.log_request_message_dict = {
            "0": r".*Finish: obj=GenerateReqInput\(.*rid='\w+', http_worker_ipc=None, video_data=None,.*",
            "1": r".*Finish: obj=GenerateReqInput\(.*rid='\w+', http_worker_ipc=None, video_data=None, sampling_params=.*",
            "2": r".*Finish: obj=GenerateReqInput\(.*rid='\w+', http_worker_ipc=None, text=.*",
            "3": r".*Finish: obj=GenerateReqInput\(.*rid='\w+', http_worker_ipc=None, text=.*",
        }

        # Keywords for extracting token ID arrays from request completion logs
        cls.keyword_Finish = r".*Finish: obj=GenerateReqInput\(.*http_worker_ipc=None, text='just.*"  # Match target log line
        cls.keyword_output_id_start = (
            "'output_ids': ["  # Start delimiter for token ID array
        )
        cls.keyword_output_id_end = "], 'meta_info'"  # End delimiter for token ID array

        # --------------------------------------------------------------------------
        # 2. --uvicorn-access-log-exclude-prefixes Configuration
        # --------------------------------------------------------------------------
        # List of path prefixes to exclude from Uvicorn access logs
        cls.log_exclude_prefixes = ["/health", "/get_server_info"]

        # --------------------------------------------------------------------------
        # 3. --enable-metrics Configuration (Latency Buckets)
        # --------------------------------------------------------------------------
        # Default bucket boundaries for time-to-first-token latency (seconds)
        # Used if --bucket-time-to-first-token is not explicitly configured
        cls.default_time_to_first_token_bucket = [
            "0.1",
            "0.2",
            "0.4",
            "0.6",
            "0.8",
            "1.0",
            "2.0",
            "4.0",
            "6.0",
            "8.0",
            "10.0",
            "20.0",
            "40.0",
            "60.0",
            "80.0",
            "100.0",
            "200.0",
            "400.0",
        ]

        # Default bucket boundaries for inter-token latency (seconds)
        # Used if --bucket-inter-token-latency is not explicitly configured
        cls.default_inter_token_latency_bucket = [
            "0.002",
            "0.004",
            "0.006",
            "0.008",
            "0.01",
            "0.015",
            "0.02",
            "0.025",
            "0.03",
            "0.035",
            "0.04",
            "0.06",
            "0.08",
            "0.1",
            "0.2",
            "0.4",
            "0.6",
            "0.8",
            "1.0",
            "2.0",
            "4.0",
            "6.0",
            "8.0",
        ]

        # Default bucket boundaries for end-to-end (E2E) request latency (seconds)
        # Used if --bucket-e2e-request-latency is not explicitly configured
        cls.default_e2e_request_latency_bucket = [
            "0.1",
            "0.2",
            "0.4",
            "0.6",
            "0.8",
            "1.0",
            "2.0",
            "4.0",
            "6.0",
            "8.0",
            "10.0",
            "20.0",
            "40.0",
            "60.0",
            "80.0",
            "100.0",
            "200.0",
            "400.0",
            "600.0",
            "1200.0",
            "1800.0",
            "2400.0",
        ]

        # Custom latency bucket boundaries (for testing non-default configurations)
        cls.my_bucket = ["0.1", "0.5", "1.0", "5.0", "10.0"]

        # --------------------------------------------------------------------------
        # 4. --collect-tokens-histogram Configuration (Token Count Buckets)
        # --------------------------------------------------------------------------
        # Default bucket boundaries for prompt/generation token count histograms
        # Used if --prompt-tokens-buckets / --generation-tokens-bucket are not explicitly configured
        # Note: Prompt and generation token buckets use identical default boundaries
        cls.default_tokens_bucket = [
            "100.0",
            "300.0",
            "500.0",
            "700.0",
            "1000.0",
            "1500.0",
            "2000.0",
            "3000.0",
            "4000.0",
            "5000.0",
            "6000.0",
            "7000.0",
            "8000.0",
            "9000.0",
            "10000.0",
            "12000.0",
            "15000.0",
            "20000.0",
            "22000.0",
            "25000.0",
            "30000.0",
            "35000.0",
            "40000.0",
            "66000.0",
            "99000.0",
            "132000.0",
            "300000.0",
            "600000.0",
            "900000.0",
            "1.1e+06",
        ]

        # Custom token count bucket boundaries (for testing custom configurations)
        cls.my_tokens_bucket = [
            "100.0",
            "1000.0",
            "10000.0",
            "100000.0",
            "300000.0",
            "600000.0",
            "900000.0",
        ]

        # Two-Sided Exponential (TSE) Bucket Strategy Configuration
        # Format: [base_value, exponential_factor, number_of_steps]
        cls.my_tse_set = ["1000", "2", "8"]
        # Precomputed custom bucket boundaries using the TSE strategy
        cls.my_tse_bucket = [
            "984.0",
            "992.0",
            "996.0",
            "998.0",
            "1000.0",
            "1002.0",
            "1004.0",
            "1008.0",
            "1016.0"
        ]

        # --------------------------------------------------------------------------
        # 5. --tokenizer-metrics-custom-labels Configuration
        # --------------------------------------------------------------------------
        # HTTP header name for custom metrics labels (--tokenizer-metrics-custom-labels-header)
        cls.labels_header = "X-Metrics-Labels"
        # Allowed custom label name (--tokenizer-metrics-allowed-custom-labels)
        cls.my_label = "business_line"

    def _verify_inference(self, max_new_tokens=32):
        """Send a basic inference request to test inference function."""
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": self.test_prompt,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_new_tokens,
                },
            },
            timeout=60,
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(self.expected_output, response.text)
        return response.text

    def _verify_log_requests_level(self, log_requests_level, out_log_file):
        """
        Validate that log content complies with expectations for different --log-requests-level configurations.

        Core Functionality:
            1. Send a request to the model to generate the longest possible string, with token generation limits optimized for efficiency:
               - Max 100 new tokens for --log-requests-level ≤ 1 (reduce generation time for low-detail logging)
               - Max 2500 new tokens for --log-requests-level ≥ 2 (exceeds 2048 to test truncation behavior)
            2. Verify the log file contains level-specific keywords matching the target log_requests_level
            3. Validate token count preservation rules in logs:
               - Level 2: Logs are truncated to retain only 2048 tokens (partial input/output)
               - Level 3: Logs retain all generated tokens (full input/output)
               - Levels ≤1: No token count validation (only metadata/sampling params logged)

        Args:
            log_requests_level (int): Target log verbosity level (0/1/2/3) for validation; maps to the
                --log-requests-level parameter
            out_log_file (file object): Open file object of the out log file
        """
        # Step 1: Send a request to the model to generate the longest possible string, with token generation limits optimized for efficiency:
        max_new_token = 2500 if log_requests_level >= 2 else 100
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": f"just return me a long string, generate as much as possible.",
                "sampling_params": {"temperature": 0, "max_new_tokens": max_new_token},
            },
        )
        self.assertEqual(response.status_code, 200)

        # Step 2: Verify the log file contains level-specific keywords matching the target log_requests_level
        out_log_file.seek(0)
        content = out_log_file.read()
        self.assertTrue(len(content) > 0)
        self.assertIsNotNone(
            re.search(self.log_request_message_dict[str(log_requests_level)], content)
        )
        # The total number of generated tokens should equal the configured maximum number of generated tokens
        lines = self.get_lines_with_keyword(self.out_log_name, self.keyword_Finish)
        finish_message = lines[0]["content"]
        self.assertIn(f"'completion_tokens': {max_new_token}", finish_message)

        # Step 3: Validate token count preservation rules in logs:
        if log_requests_level >= 2:
            # Extract the content of output_ids to count the number of generated tokens recorded in the logs
            output_ids_start_index = finish_message.find(
                self.keyword_output_id_start
            ) + len(self.keyword_output_id_start)
            output_ids_end_index = finish_message.find(self.keyword_output_id_end)
            output_ids_list_str = finish_message[
                output_ids_start_index:output_ids_end_index
            ].strip()
            if log_requests_level == 2:
                # When --log-requests-level=2, the log records a maximum of 2048 tokens (truncated content)
                self.assertIn("] ... [", output_ids_list_str)
                output_ids_list_str = output_ids_list_str.replace("] ... [", ", ")
                token_id_count = len(
                    [
                        x.strip()
                        for x in re.split(r",\s*", output_ids_list_str)
                        if x.strip()
                    ]
                )
                self.assertTrue(token_id_count == 2048)
            else:
                # When --log-requests_level=3, the log records all generated token content (no truncation)
                token_id_count = len(
                    [
                        x.strip()
                        for x in re.split(r",\s*", output_ids_list_str)
                        if x.strip()
                    ]
                )
                self.assertTrue(token_id_count > 2048)

    def _verify_log_exclude_prefixes(self, if_enable, out_log_file):
        """Validate that Uvicorn access logs exclude requests with specified path prefixes when the
        --uvicorn-access-log-exclude-prefixes configuration is active.

        Args:
            if_enable (bool):  Whether the --uvicorn-access-log-exclude-prefixes feature is activated
            out_log_file (file object): Open file object of the out log file
        """
        response = requests.get(f"{self.base_url}/health", timeout=10)
        self.assertEqual(response.status_code, 200)
        response = requests.get(f"{self.base_url}/get_server_info", timeout=10)
        self.assertEqual(response.status_code, 200)
        out_log_file.seek(0)
        content = out_log_file.read()
        health_message = '"GET /health HTTP/1.1" 200 OK'
        get_server_info_message = '"GET /get_server_info HTTP/1.1" 200 OK'
        if if_enable:
            self.assertNotIn(health_message, content)
            self.assertNotIn(get_server_info_message, content)
        else:
            self.assertIn(health_message, content)
            self.assertIn(get_server_info_message, content)

    def _verify_metrics_and_bucket_boundary(
        self,
        expected_time_to_first_token_bucket=None,
        expected_inter_token_latency_bucket=None,
        expected_e2e_request_latency_bucket=None,
        expected_prompt_tokens_bucket=None,
        expected_generation_tokens_bucket=None,
    ):
        """Validate that metrics buckets align with expected boundaries when --enable-metrics and bucket configuration parameters are set."""
        response = requests.get(f"{self.base_url}/metrics", timeout=10)
        self.assertEqual(response.status_code, 200)
        metrics_content = response.text
        if expected_time_to_first_token_bucket:
            for le in expected_time_to_first_token_bucket:
                message = f'sglang:time_to_first_token_seconds_bucket{{le="{le}",model_name="{self.model}"}}'
                self.assertIn(message, metrics_content)
        if expected_inter_token_latency_bucket:
            for le in expected_inter_token_latency_bucket:
                message = f'sglang:inter_token_latency_seconds_bucket{{le="{le}",model_name="{self.model}"}}'
                self.assertIn(message, metrics_content)
        if expected_e2e_request_latency_bucket:
            for le in expected_e2e_request_latency_bucket:
                message = f'sglang:e2e_request_latency_seconds_bucket{{le="{le}",model_name="{self.model}"}}'
                self.assertIn(message, metrics_content)
        if expected_prompt_tokens_bucket:
            for le in expected_prompt_tokens_bucket:
                message = f'sglang:prompt_tokens_histogram_bucket{{le="{le}",model_name="{self.model}"}}'
                self.assertIn(message, metrics_content)
        if expected_generation_tokens_bucket:
            for le in expected_generation_tokens_bucket:
                message = f'sglang:generation_tokens_histogram_bucket{{le="{le}",model_name="{self.model}"}}'
                self.assertIn(message, metrics_content)
        return metrics_content

    def _verify_enable_metrics_for_all_scheduler(self, if_enable):
        """Validate that the --enable-metrics-for-all-scheduler parameter controls per-TP-rank scheduler request metric collection.

        Args:
            if_enable (bool): Whether the --enable-metrics-for-all-scheduler parameter is enabled
                - True: Expect metrics for both TP rank 0 and TP rank 1
                - False: Expect metrics only for TP rank 0 (TP rank 1 metrics absent)
        """
        response = requests.get(f"{self.base_url}/metrics", timeout=10)
        message_0 = (
            f'sglang:num_decode_transfer_queue_reqs{{engine_type="unified",model_name="{self.model}"'
            f',moe_ep_rank="0",pp_rank="0",tp_rank="0"}}'
        )
        message_1 = (
            f'sglang:num_decode_transfer_queue_reqs{{engine_type="unified",model_name="{self.model}"'
            f',moe_ep_rank="0",pp_rank="0",tp_rank="1"}}'
        )
        self.assertIn(message_0, response.text)
        if if_enable:
            self.assertIn(message_1, response.text)
        else:
            self.assertNotIn(message_1, response.text)

    def _verify_log_metrics_tokenizer_label(self):
        """Validate independent statistical aggregation of requests with custom labels via tokenizer metrics label parameters."""
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "Content-Type": "application/json",
                "X-Metrics-Labels": f"{self.my_label}=customer_service",
                "text": self.test_prompt,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn(self.expected_output, response.text)

        response = requests.get(f"{self.base_url}/metrics", timeout=10)
        self.assertEqual(response.status_code, 200)
        metrics_content = response.text
        message = f"sglang:time_to_first_token_seconds_bucket{{{self.my_label}="
        self.assertIn(message, metrics_content)
        message = f"sglang:inter_token_latency_seconds_bucket{{{self.my_label}="
        self.assertIn(message, metrics_content)
        message = f"sglang:e2e_request_latency_seconds_bucket{{{self.my_label}="
        self.assertIn(message, metrics_content)

    def _verify_gc_warning_threshold(self, err_log_file):
        """Validate SGLang logs GC warnings when GC duration exceeds --gc-warning-threshold-secs threshold.

        Core Functionality:
            1. Generate high-concurrency requests with long sequences to create large temporary objects in SGLang service
            2. Trigger garbage collection (GC) by overwhelming the service with memory-intensive requests
            3. Verify that when GC time exceeds the configured threshold, a specific GC warning log is recorded in the error log file

        Args:
            err_log_file (file object): Open file object of the error log file
        """
        prompt_template = (
            "just return me a string with of 10000 characters: " + "A" * 5000
        )
        max_token = 1000

        def send_request():
            requests.post(
                f"{self.base_url}/generate",
                json={
                    "text": prompt_template,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": max_token
                    },
                },
            )

        threads = []
        for _ in range(200):
            t = threading.Thread(target=send_request)
            t.start()
            threads.append(t)
            sleep(0.01)

        for t in threads:
            t.join()

        GC_info = "LONG GARBAGE COLLECTION DETECTED"
        err_log_file.seek(0)
        content = err_log_file.read()
        self.assertTrue(len(content) > 0)
        self.assertIn(GC_info, content)

