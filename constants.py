PROCESSED_DIR = 'processed_datasets'
RAW_DATASET_PATH = 'raw_datasets/labels_with_stackx.csv'
PROCESSED_DATASET = 'labels_with_stackx_tokenized'
TRAIN_SET = 'train_set'
DEV_SET = 'dev_set'
TEST_SET = 'test_set'
STACKX_COLUMNS = [
    'question_asker_intent_understanding',
     'question_body_critical',
     'question_conversational',
     'question_expect_short_answer',
     'question_fact_seeking',
     'question_has_commonly_accepted_answer',
     'question_interestingness_others',
     'question_interestingness_self',
     'question_multi_intent',
     'question_not_really_a_question',
     'question_opinion_seeking',
     'question_type_choice',
     'question_type_compare',
     'question_type_consequence',
     'question_type_definition',
     'question_type_entity',
     'question_type_instructions',
     'question_type_procedure',
     'question_type_reason_explanation',
     'question_type_spelling',
     'question_well_written'
]
NUM_SX_FEATURES = len(STACKX_COLUMNS)
MAX_LEN = 1024
