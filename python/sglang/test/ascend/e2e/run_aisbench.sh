#!/bin/bash

set -e

echo "===== Install aisbench in virtual env - Begin ====="
PYTHON_ENV_FOR_AISBENCH=test_env_aisbench
PIP_FOR_AISBENCH=${PYTHON_ENV_FOR_AISBENCH}/bin/pip
python -m venv ${PYTHON_ENV_FOR_AISBENCH}
AISBENCH_SOURCE_PATH=/root/.cache/.cache/benchmark
if [ ! -d "${AISBENCH_SOURCE_PATH}" ]; then
  echo "The dependent aisbench package does not exist: ${AISBENCH_SOURCE_PATH}."
  echo "git clone https://github.com/AISBench/benchmark.git"
  git clone https://github.com/AISBench/benchmark.git
  AISBENCH_SOURCE_PATH="./benchmark/"
fi
${PIP_FOR_AISBENCH} install -U pip -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
${PIP_FOR_AISBENCH} install -e ${AISBENCH_SOURCE_PATH} --use-pep517 -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
${PIP_FOR_AISBENCH} install -r ${AISBENCH_SOURCE_PATH}/requirements/api.txt -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
${PIP_FOR_AISBENCH} install -r ${AISBENCH_SOURCE_PATH}/requirements/extra.txt -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
echo "===== Install aisbench in virtual env - End ====="

# Check if the correct number of arguments are provided
if [ $# -ne 9 ]; then
    echo -e "\033[31mUsage:\033[0m"
    echo "  $0 <IP> <PORT> <MODEL_NAME> <MODEL_PATH> <DATASET> <MAX_OUT_LEN> <BATCH_SIZE> <NUM_PROMPTS> <OUTPUT_PATH>"
    echo "  Example: $0 127.0.0.1 54321 Qwen2-7B-Instruct /models/qwen gsm8k_gen 1024 32 128 ./result"
    exit 1
fi

# Assign command-line arguments to variables
IP=$1
PORT=$2
MODEL=$3
PATH=$4
DATASET_PATH=$5
MAX_OUT_LEN=$6
BATCH_SIZE=$7
NUM_PROMPTS=$8
OUTPUT_PATH=$9

TMP_CFG=/tmp/vllm_api_${MODEL}.py
cat > "$TMP_CFG" << EOF
from ais_bench.benchmark.models import VLLMCustomAPIChatStream
models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChatStream,
        abbr='vllm-api-stream-chat',
        path="$PATH",
        model="$MODEL",
        stream=True,
        request_rate=0,
        use_timestamp=False,
        retry=2,
        api_key="",
        host_ip="$IP",
        host_port=$PORT,
        url="",
        max_out_len=$MAX_OUT_LEN,
        batch_size=$BATCH_SIZE,
        trust_remote_code=True,
        generation_kwargs=dict(
            temperature=0,
            ignore_eos=True
        )
        pred_postprocessor=dict(type=extract_non_reasoning_content),
    )
]
EOF


TMP_DATASET=/tmp/mm_custom_gen_${MODEL}.py
cat > "$TMP_DATASET" << EOF
from ais_bench.benchmark.openicl.icl_prompt_template.icl_prompt_template_mm import MMPromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.datasets import MMCustomDataset, MMCustomEvaluator


mm_custom_reader_cfg = dict(
    input_columns=['question', 'image', 'video', 'audio'],
    output_column='answer'
)


mm_custom_infer_cfg = dict(
    prompt_template=dict(
        type=MMPromptTemplate,
        template=dict(
            round=[
                dict(role="HUMAN", prompt_mm={
                    "text": {"type": "text", "text": "{question}"},
                    "image": {"type": "image_url", "image_url": {"url": "file://{image}"}},
                    "video": {"type": "video_url", "video_url": {"url": "file://{video}"}},
                    "audio": {"type": "audio_url", "audio_url": {"url": "file://{audio}"}},
                })
            ]
            )
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer)
)

mm_custom_eval_cfg = dict(
    evaluator=dict(type=MMCustomEvaluator)
)

mm_custom_datasets = [
    dict(
        abbr='mm_custom',
        type=MMCustomDataset,
        path=$DATASET_PATH,
        mm_type="path",
        num_frames=5,
        reader_cfg=mm_custom_reader_cfg,
        infer_cfg=mm_custom_infer_cfg,
        eval_cfg=mm_custom_eval_cfg,
        k=1,
        n=1,
    )
]
EOF


echo "IP: $IP | Port: $PORT | Model: $MODEL | Path: $PATH"
echo "Output tokens: $MAX_OUT_LEN | Batch size: $BATCH_SIZE | Prompts num: $NUM_PROMPTS"
echo -e "API config file: $TMP_CFG"
echo -e "Dataset config file: $TMP_DATASET"

source ${PYTHON_ENV_FOR_AISBENCH}/bin/activate
CMD="ais_bench --models $TMP_CFG --datasets $TMP_DATASET --mode perf --num-prompts $NUM_PROMPTS --output-dir $OUTPUT_PATH"
echo "Run command: ${CMD}"
eval "${CMD}"
