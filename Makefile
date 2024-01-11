help:
	@echo "  setup				set up environment and installing dependencies"
	@echo "  check 				check the source for style errors"
	@echo "  fix                fix style errors"
	@echo "  fix-check          fix style errors and check"
	@echo "  unit-tests			run the unit testsuite"
	@echo "  all				run check and tests"
	@echo "  clean				remove temporary files"

.PHONY: setup
setup:
	@echo ">>> Setting up environment and installing dependencies <<<"
	poetry install --no-interaction

.PHONY: check
check:
	@echo ">>> Checking the source code for style errors <<<"
	poetry run poe check

.PHONY: fix
fix:
	@echo ">>> Fixing the source code from style errors <<<"
	poetry run poe fix

.PHONY: fix-check
fix-check:
	@echo ">>> Fixing the source code from style errors and run check <<<"
	poetry run poe fix_check

.PHONY: tests
tests:
	@echo ">>> Running tests <<<"
	poetry run poe tests

.PHONY: all
all:
	@echo ">>> Running checks and tests <<<"
	poetry run poe all

.PHONY: clean
clean:
	@echo ">>> Removing temporary files <<<"
	rm -rf .artifacts .mypy_cache .pytest_cache .coverage
