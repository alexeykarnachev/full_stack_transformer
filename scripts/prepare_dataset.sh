# All paths need to be absolute or relative to the ./scripts directory.
DOCUMENTS_TRAIN_FILE="../data/documents/ru_rap/train.txt"
DOCUMENTS_VALID_FILE="../data/documents/ru_rap/valid.txt"
TOKENIZER_CLS_NAME="RuTransformersTokenizer"
DATASETS_ROOT="../data/datasets/"
DATASET_NAME="ru_rap"
MAX_SAMPLE_LENGTH=80
MIN_SAMPLE_LENGTH=10
END_OF_DOCUMENT="|"

BASEDIR=$(dirname "$0")
cd "${BASEDIR}" || exit
PY_SCRIPT="../lm_trainer/scripts/prepare_dataset.py"

python ${PY_SCRIPT} \
--documents_train_file "${DOCUMENTS_TRAIN_FILE}" \
--documents_valid_file "${DOCUMENTS_VALID_FILE}" \
--end_of_document "${END_OF_DOCUMENT}" \
--tokenizer_cls_name "${TOKENIZER_CLS_NAME}" \
--max_sample_length "${MAX_SAMPLE_LENGTH}" \
--min_sample_length "${MIN_SAMPLE_LENGTH}" \
--datasets_root "${DATASETS_ROOT}" \
--dataset_name "${DATASET_NAME}"