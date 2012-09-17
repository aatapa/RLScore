TRAIN_FEATURES = 'train_features'
TRAIN_LABELS = 'train_labels'
TRAIN_QIDS = 'train_qids'
TRAIN_PREFERENCES = 'train_preferences'
VALIDATION_FEATURES = 'validation_features'
VALIDATION_LABELS = 'validation_labels'
VALIDATION_QIDS = 'validation_qids'
VALIDATION_PREFERENCES = 'validation_preferences'
BASIS_VECTORS = 'basis_vectors'
SVD_ADAPTER = 'svd_adapter'
CVFOLDS = 'cross-validation_folds'
MODEL = 'model'
PREDICTION_FEATURES = 'prediction_features'
PREDICTION_QIDS = 'test_qids'
PREDICTED_LABELS = 'predicted_labels'
TEST_LABELS = 'test_labels'
TEST_PREFERENCES = 'test_preferences'
PREDICTED_CLUSTERS_FOR_TRAINING_DATA = 'predicted_clusters_for_training_data'
SELECTED_FEATURES = 'selected_features'
GREEDYRLS_LOO_PERFORMANCES = 'GreedyRLS_LOO_performances'
GREEDYRLS_TEST_PERFORMANCES = 'GreedyRLS_test_performances'
PARAMETERS = 'parameters'
KMATRIX = 'kmatrix'
REGGRID_RESULTS = 'mselection_performances'
TEST_PERFORMANCE = 'test_performance'
FIXED_INDICES = 'fixed_indices'
TRAIN_SET = 'train_set'
TEST_SET = 'test_set'
VALIDATION_SET = 'validation_set'
KERNEL_OBJ = 'kernel_obj'
CALLBACK_FUNCTION = 'callback_obj'
PERFORMANCE_MEASURE = 'measure'
TIKHONOV_REGULARIZATION_PARAMETER = 'regparam'
BASIS_VECTORS_TYPE = 'basis_vectors_variable_type'
INT_LIST_TYPE = 'int_list_variable_type'
FLOAT_LIST_TYPE = 'float_list_variable_type'


VARIABLE_TYPES = {
                  TRAIN_FEATURES: "spmatrix",
                  VALIDATION_FEATURES: "spmatrix",
                  PREDICTION_FEATURES: "spmatrix",
                  TRAIN_LABELS: "matrix",
                  VALIDATION_LABELS: "matrix",
                  TEST_LABELS: "matrix",
                  PREDICTED_LABELS: "matrix",
                  PREDICTED_CLUSTERS_FOR_TRAINING_DATA: INT_LIST_TYPE,
                  TRAIN_QIDS: "qids",
                  VALIDATION_QIDS: "qids",
                  PREDICTION_QIDS: "qids",
                  TRAIN_PREFERENCES: "preferences",
                  VALIDATION_PREFERENCES: "preferences",
                  TEST_PREFERENCES: "preferences",
                  CVFOLDS: "index_partition",
                  MODEL: 'model',
                  BASIS_VECTORS: BASIS_VECTORS_TYPE,
                  SELECTED_FEATURES: INT_LIST_TYPE,
                  GREEDYRLS_LOO_PERFORMANCES: FLOAT_LIST_TYPE,
                  GREEDYRLS_TEST_PERFORMANCES: FLOAT_LIST_TYPE,
                  PARAMETERS: dict,
                  PERFORMANCE_MEASURE: "measure",
                  KMATRIX: "matrix",
                  TRAIN_SET: "data_set",
                  VALIDATION_SET: "data_set",
                  TEST_SET: "data_set",
                  REGGRID_RESULTS: "matrix"
                  }

COMPOSITES = {TRAIN_SET: (TRAIN_FEATURES, TRAIN_LABELS, TRAIN_QIDS),
              VALIDATION_SET: (VALIDATION_FEATURES, VALIDATION_LABELS, VALIDATION_QIDS),
              TEST_SET: (PREDICTION_FEATURES, TEST_LABELS, PREDICTION_QIDS)
              }

