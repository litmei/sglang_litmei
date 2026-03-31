import os
import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH,
    send_concurrent_requests,
    verify_process_terminated,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=300, suite="nightly-1-npu-a3", nightly=True)


class TestDynamicBatchTokenizerBatchSize64(CustomTestCase):
    """Testcase: Verify dynamic batch tokenizer with --dynamic-batch-tokenizer-batch-size=64.
    80 requests are sent (> one batch cycle) so the tokenizer must handle multiple
    successive batches without dropping any request.

    [Test Category] Parameter
    [Test Target] --enable-dynamic-batch-tokenizer; --dynamic-batch-tokenizer-batch-size
    """

    model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
    batch_size = 64
    num_requests = 80
    num_concurrent = 16

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
                "--enable-dynamic-batch-tokenizer",
                "--dynamic-batch-tokenizer-batch-size",
                str(cls.batch_size),
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        verify_process_terminated(cls.process, cls.__name__)

    def test_batch_size_64(self):
        # 80 concurrent requests with batch_size=64: the tokenizer must process
        # two tokenization batches (64 + 16) and return all 80 results.
        results = send_concurrent_requests(
            base_url=self.base_url,
            num_requests=self.num_requests,
            num_concurrent=self.num_concurrent,
        )
        success_count = sum(1 for r in results if r["status_code"] == 200)
        self.assertEqual(
            success_count,
            self.num_requests,
            f"Expected {self.num_requests} successful requests, "
            f"but only {success_count} succeeded.",
        )


class TestDynamicBatchTokenizerBatchSize1(CustomTestCase):
    """Testcase: Verify dynamic batch tokenizer degrades gracefully when
    --dynamic-batch-tokenizer-batch-size is set to 1 (no grouping benefit).
    Each request is tokenized individually; all must still complete correctly.

    [Test Category] Parameter
    [Test Target] --enable-dynamic-batch-tokenizer; --dynamic-batch-tokenizer-batch-size
    """

    model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
    batch_size = 1
    num_requests = 20
    num_concurrent = 4

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
                "--enable-dynamic-batch-tokenizer",
                "--dynamic-batch-tokenizer-batch-size",
                str(cls.batch_size),
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_batch_size_1(self):
        # With batch_size=1 the loop never accumulates more than one request.
        # This is equivalent to disabling batching.  Verify correctness.
        results = send_concurrent_requests(
            base_url=self.base_url,
            num_requests=self.num_requests,
            num_concurrent=self.num_concurrent,
        )
        success_count = sum(1 for r in results if r["status_code"] == 200)
        self.assertEqual(
            success_count,
            self.num_requests,
            f"Expected {self.num_requests} successful requests, "
            f"but only {success_count} succeeded.",
        )
        for result in results:
            self.assertIn(
                "Paris",
                result["text"],
                f"Task {result['task_id']}: inference result does not contain 'Paris'.",
            )


if __name__ == "__main__":
    unittest.main()
