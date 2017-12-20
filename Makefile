DATASET_PATH := $(dataset_path)

init:
	pip install -r requirements.txt

test_deps:
	pip install -r requirements-dev.txt

test: init test_deps
	mamba -f documentation ./test

check_dataset:
ifndef DATASET_PATH
	@echo "The dataset path is not available. Please provide a dataset as parameter ('dataset_path=PATH/TO/DATASET')"
	@exit 1
endif

create_model: check_dataset init
	python -m spamdetector.model_builder $(DATASET_PATH)

select_pipeline: check_dataset init
	python -m spamdetector.pipeline_selector $(DATASET_PATH)

.PHONY: init test
