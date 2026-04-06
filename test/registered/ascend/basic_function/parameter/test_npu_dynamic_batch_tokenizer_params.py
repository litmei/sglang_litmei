import threading
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH,
    send_concurrent_requests,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=300, suite="nightly-1-npu-a3", nightly=True)

NUM_REQUESTS = 20
NUM_CONCURRENT = 8
SHORT_PROMPTS = ["Hi", "OK", "Yes"]
MEDIUM_PROMPTS = [
    "What is the capital of France?",
    "Explain what a neural network is.",
    "Describe the water cycle briefly.",
]
LONG_PROMPTS = [
    "Describe the history of the Roman Empire and its influence on modern culture " * 3,
    "Explain how large language models are trained, evaluated, and deployed " * 3,
]
SAMPLING_CONFIGS = [
    {"temperature": 0.0, "max_new_tokens": 32},
    {"temperature": 0.7, "max_new_tokens": 32},
    {"temperature": 1.0, "max_new_tokens": 32},
    {"temperature": 0.0, "top_p": 0.9, "max_new_tokens": 32},
]




class TestBatchSize64Timeout0p001(CustomTestCase):
    """batch_size=64, timeout=0.001: large capacity with minimal wait."""
    model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model, cls.base_url, timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--attention-backend", "ascend",
                "--disable-cuda-graph",
                "--enable-dynamic-batch-tokenizer",
                "--dynamic-batch-tokenizer-batch-size", "64",
                "--dynamic-batch-tokenizer-batch-timeout", "0.001",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_concurrent_requests(self):
        results = send_concurrent_requests(self.base_url, num_requests=80, num_concurrent=16)
        success = sum(1 for r in results if r["status_code"] == 200)
        self.assertEqual(success, 80, f"Expected 80 successes, got {success}")
        for r in results:
            self.assertIn("Paris", r["text"])


class TestBatchSize1Timeout0p1(CustomTestCase):
    """batch_size=1, timeout=0.1: minimal capacity with long wait (timeout ignored)."""
    model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model, cls.base_url, timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--attention-backend", "ascend",
                "--disable-cuda-graph",
                "--enable-dynamic-batch-tokenizer",
                "--dynamic-batch-tokenizer-batch-size", "1",
                "--dynamic-batch-tokenizer-batch-timeout", "0.1",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_concurrent_requests(self):
        results = send_concurrent_requests(self.base_url, num_requests=NUM_REQUESTS, num_concurrent=4)
        success = sum(1 for r in results if r["status_code"] == 200)
        self.assertEqual(success, NUM_REQUESTS, f"Expected {NUM_REQUESTS} successes, got {success}")
        for r in results:
            self.assertIn("Paris", r["text"])




class TestDynamicBatchTokenizerCombo(CustomTestCase):
    """All remaining scenarios share one server with default parameters."""
    model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model, cls.base_url, timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--attention-backend", "ascend",
                "--disable-cuda-graph",
                "--enable-dynamic-batch-tokenizer",
                "--dynamic-batch-tokenizer-batch-size", "32",
                "--dynamic-batch-tokenizer-batch-timeout", "0.002",
                "--disable-radix-cache",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_high_concurrency(self):
        results = send_concurrent_requests(self.base_url, num_requests=100, num_concurrent=20)
        success = sum(1 for r in results if r["status_code"] == 200)
        self.assertEqual(success, 100, f"High concurrency: {success}/100 succeeded")
        for r in results:
            self.assertIn("Paris", r["text"])

    def test_mixed_text_lengths(self):
        prompts = SHORT_PROMPTS + MEDIUM_PROMPTS + LONG_PROMPTS
        results, lock = [], threading.Lock()

        def send(p):
            try:
                resp = requests.post(f"{self.base_url}/generate", json={
                    "text": p, "sampling_params": {"temperature": 0, "max_new_tokens": 32}
                }, timeout=60)
                with lock:
                    results.append(resp.status_code)
            except Exception:
                with lock:
                    results.append(-1)

        threads = [threading.Thread(target=send, args=(p,)) for p in prompts]
        for t in threads: t.start()
        for t in threads: t.join()
        success = sum(1 for s in results if s == 200)
        self.assertEqual(success, len(prompts), f"Mixed lengths: {success}/{len(prompts)} succeeded")

    def test_streaming_requests(self):
        prompts = ["The capital of France is", "The largest planet is", "The speed of light is"]
        results, lock = [], threading.Lock()

        def send_stream(p):
            try:
                resp = requests.post(f"{self.base_url}/generate", json={
                    "text": p, "sampling_params": {"temperature": 0, "max_new_tokens": 32}, "stream": True
                }, stream=True, timeout=60)
                has_content = any(line and line.startswith(b"data: ") and line[6:] != b"[DONE]" for line in resp.iter_lines())
                with lock:
                    results.append((resp.status_code, has_content))
            except Exception:
                with lock:
                    results.append((-1, False))

        threads = [threading.Thread(target=send_stream, args=(p,)) for p in prompts]
        for t in threads: t.start()
        for t in threads: t.join()
        for code, has in results:
            self.assertEqual(code, 200, "Streaming request non-200")
            self.assertTrue(has, "Streaming response no content")

    def test_different_sampling_params(self):
        payloads = SAMPLING_CONFIGS * 5
        results, lock = [], threading.Lock()

        def send(sp):
            try:
                resp = requests.post(f"{self.base_url}/generate", json={
                    "text": "The capital of France is", "sampling_params": sp
                }, timeout=60)
                with lock:
                    results.append(resp.status_code)
            except Exception:
                with lock:
                    results.append(-1)

        threads = [threading.Thread(target=send, args=(p,)) for p in payloads]
        for t in threads: t.start()
        for t in threads: t.join()
        success = sum(1 for s in results if s == 200)
        self.assertEqual(success, len(payloads), f"Sampling params: {success}/{len(payloads)} succeeded")

    def test_disable_radix_cache(self):
        # Already covered by --disable-radix-cache in server args; just verify connectivity.
        results = send_concurrent_requests(self.base_url, num_requests=NUM_REQUESTS, num_concurrent=NUM_CONCURRENT)
        success = sum(1 for r in results if r["status_code"] == 200)
        self.assertEqual(success, NUM_REQUESTS, f"Radix cache disabled: {success}/{NUM_REQUESTS} succeeded")
        for r in results:
            self.assertIn("Paris", r["text"])


if __name__ == "__main__":
    unittest.main()