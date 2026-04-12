#!/bin/bash

set -e

echo "===== Install aisbench in virtual env - Begin ====="
PYTHON_ENV_FOR_AISBENCH=test_env_aisbench
PIP_FOR_AISBENCH=${PYTHON_ENV_FOR_AISBENCH}/bin/pip
python -m venv ${PYTHON_ENV_FOR_AISBENCH}
AISBENCH_SOURCE_PATH=/root/.cache/.cache/benchmark
AISBENCH_PKG_PATH=/root/.cache/.cache/aisbench-packages
if [ ! -d "${AISBENCH_SOURCE_PATH}" ]; then
  echo "The aisbench source does not exist: ${AISBENCH_SOURCE_PATH}."
  echo "git clone https://github.com/AISBench/benchmark.git"
  git clone https://github.com/AISBench/benchmark.git
  AISBENCH_SOURCE_PATH="./benchmark/"
fi
if [ ! -d "${AISBENCH_PKG_PATH}" ]; then
  echo "The dependent aisbench package does not exist: ${AISBENCH_PKG_PATH}."
  echo "Install aisbench online."
  ${PIP_FOR_AISBENCH} install -U pip -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
  ${PIP_FOR_AISBENCH} install -e ${AISBENCH_SOURCE_PATH} --use-pep517 -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
  ${PIP_FOR_AISBENCH} install -r ${AISBENCH_SOURCE_PATH}/requirements/api.txt -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
  ${PIP_FOR_AISBENCH} install -r ${AISBENCH_SOURCE_PATH}/requirements/extra.txt -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
else
  echo "Install aisbench locally."
  ${PIP_FOR_AISBENCH} install -U pip --no-index --find-links=${AISBENCH_PKG_PATH}
  ${PIP_FOR_AISBENCH} install -e ${AISBENCH_SOURCE_PATH} --use-pep517 --no-index --find-links=${AISBENCH_PKG_PATH}
  ${PIP_FOR_AISBENCH} install -r ${AISBENCH_SOURCE_PATH}/requirements/api.txt --no-index --find-links=${AISBENCH_PKG_PATH}
  ${PIP_FOR_AISBENCH} install -r ${AISBENCH_SOURCE_PATH}/requirements/extra.txt --no-index --find-links=${AISBENCH_PKG_PATH}
fi
echo "===== Install aisbench in virtual env - End ====="

# Check if the correct number of arguments are provided
if [ $# -ne 10 ]; then
    echo -e "\033[31mUsage:\033[0m"
    echo "  $0 <IP> <PORT> <MODEL_NAME> <MODEL_PATH> <DATASET_TYPE> <DATASET_PATH> <MAX_OUT_LEN> <BATCH_SIZE> <NUM_PROMPTS> <OUTPUT_PATH>"
    echo "  Example: $0 127.0.0.1 54321 Qwen2-7B-Instruct /models/qwen gsm8k_gen /path/to/gsm8k 1024 32 128 ./result"
    exit 1
fi

# Assign command-line arguments to variables
IP=$1
PORT=$2
MODEL=$3
MODEL_PATH=$4
DATASET_TYPE=$5
DATASET_PATH=$6
MAX_OUT_LEN=$7
BATCH_SIZE=$8
NUM_PROMPTS=$9
OUTPUT_PATH=${10}

AISBENCH_CINFG_PATH=/tmp/ais_configs

MODEL_CONFIG_PATH=${AISBENCH_CINFG_PATH}/models
/bin/mkdir -p ${MODEL_CONFIG_PATH}
TMP_CFG=vllm_api_${MODEL}
/bin/cat > "${MODEL_CONFIG_PATH}/${TMP_CFG}.py" << EOF
from ais_bench.benchmark.models import VLLMCustomAPIChatStream
from ais_bench.benchmark.utils.postprocess.model_postprocessors import extract_non_reasoning_content
models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChatStream,
        abbr='vllm-api-stream-chat',
        path="$MODEL_PATH",
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
        generation_kwargs=dict(temperature=0,ignore_eos=True),
        pred_postprocessor=dict(type=extract_non_reasoning_content),
    )
]
EOF


DATASETS_CONFIG_PATH=${AISBENCH_CINFG_PATH}/datasets
/bin/mkdir -p ${DATASETS_CONFIG_PATH}
TMP_DATASET=$DATASET_PATH
if [ "$DATASET_TYPE" == "mm-custom-gen" ]; then
    if [ ! -f "$DATASET_PATH" ]; then
        echo "The mm-custom-gen dataset file does not exist: ${DATASET_PATH}."
        exit 1
    fi

    TMP_DATASET=mm_custom_gen_${MODEL}
    /bin/cat > "${DATASETS_CONFIG_PATH}/${TMP_DATASET}.py" << EOF
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
        path="$DATASET_PATH",
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

elif [ "$DATASET_TYPE" == "gsm8k" ]; then
    TMP_DATASET=gsm8k_gen_${MODEL}
    /bin/cat > "${DATASETS_CONFIG_PATH}/${TMP_DATASET}.py" << EOF
from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.openicl.icl_evaluator import AccEvaluator
from ais_bench.benchmark.datasets import GSM8KDataset, gsm8k_postprocess, gsm8k_dataset_postprocess, Gsm8kEvaluator
gsm8k_reader_cfg = dict(input_columns=['question'], output_column='answer')

gsm8k_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template="{question}"),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer))

gsm8k_eval_cfg = dict(evaluator=dict(type=Gsm8kEvaluator),
                      pred_role='BOT',
                      pred_postprocessor=dict(type=gsm8k_postprocess),
                      dataset_postprocessor=dict(type=gsm8k_dataset_postprocess))

gsm8k_datasets = [
    dict(
        abbr='gsm8k',
        type=GSM8KDataset,
        path='ais_bench/datasets/gsm8k',  # 数据集路径，使用相对路径时相对于源码根路径，支持绝对路径
        reader_cfg=gsm8k_reader_cfg,
        infer_cfg=gsm8k_infer_cfg,
        eval_cfg=gsm8k_eval_cfg)
]
EOF

else
    echo "The dataset type $DATASET_TYPE is not supported."
    exit 1
fi


echo "IP: $IP | Port: $PORT | Model: $MODEL | Model Path: $MODEL_PATH"
echo "Output tokens: $MAX_OUT_LEN | Batch size: $BATCH_SIZE | Prompts num: $NUM_PROMPTS"
echo -e "Model config: $TMP_CFG"
echo -e "Dataset config: $TMP_DATASET"

source ${PYTHON_ENV_FOR_AISBENCH}/bin/activate
CMD="ais_bench --config-dir ${AISBENCH_CINFG_PATH} --models $TMP_CFG --datasets $TMP_DATASET --mode perf --num-prompts $NUM_PROMPTS --work-dir $OUTPUT_PATH"
echo "Run command: ${CMD}"
eval "${CMD}"
