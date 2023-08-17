check_dirs := .

quality:
	black --check $(check_dirs) --line-length 119
	ruff $(check_dirs) --line-length 119

style:
	black $(check_dirs) --line-length 119
	ruff $(check_dirs) --line-length 119 --fix