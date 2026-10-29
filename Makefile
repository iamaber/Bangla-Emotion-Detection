.PHONY: help setup test profile notebooks strip clean

help:
	@echo "setup      create .venv and install requirements"
	@echo "test       run the unit tests"
	@echo "profile    regenerate results/dataset_profile.json from the CSV"
	@echo "notebooks  execute all three notebooks headlessly"
	@echo "strip      remove notebook outputs (needs nbstripout)"

setup:
	python3 -m venv .venv
	./.venv/bin/pip install -r requirements.txt nbstripout

test:
	PYTHONHASHSEED=42 ./.venv/bin/python -m unittest discover -s tests -v

profile:
	./.venv/bin/python profile_dataset.py

notebooks:
	PYTHONHASHSEED=42 ./.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
		"XGBoost.ipynb" "Multi-Task-Neural network.ipynb" \
		"Neural network with banglabert_small.ipynb"

strip:
	./.venv/bin/nbstripout *.ipynb

clean:
	rm -rf results/*.joblib results/*.weights.h5 results/lstm_tokenizer.json
	rm -rf __pycache__ tests/__pycache__
