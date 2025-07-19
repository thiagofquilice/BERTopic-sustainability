.PHONY: help tree-%

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "%-15s %s\n", $$1, $$2}'

# Build ASCII hierarchy for DATASET=% (guardian|papers)
tree-%: ## Build ASCII hierarchy for DATASET=% (guardian|papers)
	python utils/visualize_tree.py \
		--model_path results/$*/$*_bertopic_model \
		--hier_csv  results/$*/hierarchical_topics.csv \
		--output_file results/$*/topic_tree.txt
