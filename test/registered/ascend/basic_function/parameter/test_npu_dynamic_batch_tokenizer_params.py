import logging
import threading
import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH,
    QWEN3_32B_WEIGHTS_PATH,
    send_concurrent_requests,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.few_shot_gsm8k import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------
NUM_REQUESTS = 20
NUM_CONCURRENT = 8

SHORT_PROMPTS = ["Hi", "OK", "Yes"]
MEDIUM_PROMPTS = [
    ("What is the capital of France?", "Paris"),
    ("Explain what a neural network is.", "neural"),
    ("Describe the water cycle briefly.", "water"),
]
LONG_PROMPTS = [
    ("Describe the history of the Roman Empire and its influence on modern culture " * 3, "Roman"),
    ("Explain how large language models are trained, evaluated, and deployed " * 3, "trained"),
]

SAMPLING_CONFIGS = [
    {"temperature": 0.0, "max_new_tokens": 32},
    {"temperature": 0.7, "max_new_tokens": 32},
    {"temperature": 1.0, "max_new_tokens": 32},
    {"temperature": 0.0, "top_p": 0.9, "max_new_tokens": 32},
]

# ------------------------------------------------------------
# Single‑card parameter tests (Llama-3.2-1B)
# ------------------------------------------------------------
register_npu_ci(est_time=300, suite="nightly-1-npu-a3", nightly=True)

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

    def test_high_concurrency(self):
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

    def test_high_concurrency(self):
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
        prompts_with_keywords = SHORT_PROMPTS + MEDIUM_PROMPTS + LONG_PROMPTS
        results = []
        lock = threading.Lock()

        def send(item):
            if isinstance(item, tuple):
                prompt, keyword = item
            else:
                prompt, keyword = item, None
            try:
                resp = requests.post(f"{self.base_url}/generate", json={
                    "text": prompt,
                    "sampling_params": {"temperature": 0, "max_new_tokens": 32}
                }, timeout=60)
                with lock:
                    results.append((resp.status_code, resp.text, keyword))
            except Exception:
                with lock:
                    results.append((-1, "", keyword))

        threads = [threading.Thread(target=send, args=(item,)) for item in prompts_with_keywords]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for status_code, text, keyword in results:
            self.assertEqual(status_code, 200, f"Request failed with status {status_code}")
            if keyword:
                self.assertIn(keyword, text,
                              f"Expected keyword '{keyword}' not found in response: {text[:200]}...")

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
        for t in threads:
            t.start()
        for t in threads:
            t.join()
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
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        success = sum(1 for s in results if s == 200)
        self.assertEqual(success, len(payloads), f"Sampling params: {success}/{len(payloads)} succeeded")

    def test_disable_radix_cache(self):
        results = send_concurrent_requests(self.base_url, num_requests=NUM_REQUESTS, num_concurrent=NUM_CONCURRENT)
        success = sum(1 for r in results if r["status_code"] == 200)
        self.assertEqual(success, NUM_REQUESTS, f"Radix cache disabled: {success}/{NUM_REQUESTS} succeeded")
        for r in results:
            self.assertIn("Paris", r["text"])


# ------------------------------------------------------------
# Four‑card GSM8K accuracy test (Qwen3-32B)
# ------------------------------------------------------------
register_npu_ci(est_time=400, suite="nightly-4-npu-a3", nightly=True)

BASE_OTHER_ARGS = [
    "--chunked-prefill-size", "256",
    "--attention-backend", "ascend",
    "--disable-cuda-graph",
    "--mem-fraction-static", "0.8",
    "--tp-size", "4",
    "--enable-dynamic-batch-tokenizer",
    "--dynamic-batch-tokenizer-batch-size", "4",
    "--log-level", "debug",
]

def launch_server_with_timeout(model_name, base_url, timeout, other_args_base):
    other_args = other_args_base.copy()
    idx = other_args.index("--dynamic-batch-tokenizer-batch-timeout") + 1 if "--dynamic-batch-tokenizer-batch-timeout" in other_args else -1
    if idx > 0:
        other_args[idx] = str(timeout)
    else:
        other_args.extend(["--dynamic-batch-tokenizer-batch-timeout", str(timeout)])
    return popen_launch_server(model_name, base_url, timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH, other_args=other_args)

def run_gsm8k_test(base_url, scenario):
    accuracy = 0.86
    args = SimpleNamespace(
        num_shots=5,
        data_path=None,
        num_questions=200,
        max_new_tokens=512,
        parallel=128,
        host="127.0.0.1",
        port=int(base_url.split(":")[-1]),
    )
    metrics = run_eval(args)
    assert metrics["accuracy"] >= accuracy, f"accuracy {metrics['accuracy']} < {accuracy}"
    server_info = requests.get(base_url + "/get_server_info")
    logger.info(f"{scenario}: server_info={server_info}")

class TestQwen32BTimeout001(CustomTestCase):
    """GSM8K accuracy with dynamic batch tokenizer timeout = 0.001s."""
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = launch_server_with_timeout(QWEN3_32B_WEIGHTS_PATH, cls.base_url, 0.001, BASE_OTHER_ARGS)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k_timeout_001(self):
        run_gsm8k_test(self.base_url, "timeout=0.001")

class TestQwen32BTimeout005(CustomTestCase):
    """GSM8K accuracy with dynamic batch tokenizer timeout = 0.005s."""
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST   # 修正拼写
        cls.process = launch_server_with_timeout(QWEN3_32B_WEIGHTS_PATH, cls.base_url, 0.005, BASE_OTHER_ARGS)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k_timeout_005(self):          # 方法名与实际 timeout 一致
        run_gsm8k_test(self.base_url, "timeout=0.005")

if __name__ == "__main__":
    unittest.main()
