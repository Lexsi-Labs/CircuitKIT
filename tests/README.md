# CircuitKit Test Suite

This directory contains all tests for CircuitKit, organized by test type and category.

## Directory Structure

### unit/
Unit tests for individual components and functions. Each test file focuses on a single module or feature.

**Core Tests:**
- `test_api.py` - CircuitKit main API functionality
- `test_bootstrap.py` - Task bootstrapping and initialization
- `test_cli.py` - Command-line interface tests
- `test_modules_exist.py` - Module import and structure verification
- `test_core_functionality.py` - Core circuit discovery functionality

**Feature Tests:**
- `test_circuit_scores.py` - Circuit scoring mechanisms
- `test_pillars.py` - Pillar-based circuit analysis
- `test_soft_healing.py` - Soft healing methodology
- `test_steering.py` - Circuit-based steering
- `test_stability_robustness_reports.py` - Stability and robustness analysis
- `test_structural_pruner.py` - Structural pruning implementation

**Metric Tests:**
- `test_graph_viz.py` - Graph visualization functionality
- `test_graph_viz_standalone.py` - Standalone graph visualization
- `test_multi_token_answers.py` - Multi-token answer handling
- `test_multi_token_unit.py` - Multi-token unit tests
- `test_perplexity_direct.py` - Direct perplexity metric testing
- `test_perplexity_metric.py` - Perplexity metric implementation
- `test_perplexity_metric_simple.py` - Simplified perplexity tests
- `test_ranking_metrics.py` - Ranking metrics (M7.0.5)

**Validation Tests:**
- `test_validators_direct.py` - Direct validator testing
- `test_validators_standalone.py` - Standalone validator implementation
- `test_paraphrase_direct.py` - Paraphrase validation

**Schema & Configuration Tests:**
- `test_extended_schema.py` - Extended task schema
- `test_d5_d7_import.py` - D5/D7 model imports
- `test_config_defaults.py` - Default configuration validation
- `test_mlp1.py` - MLP component testing

### integration/
Integration tests that verify interaction between multiple components and systems.

- `test_soft_healing_integration.py` - End-to-end soft healing workflow
- `test_steering_ioi.py` - IOI task steering integration
- `test_transfer_integration.py` - Transfer learning integration
- `test_transfer_matrix.py` - Transfer matrix functionality
- `test_transfer_minimal.py` - Minimal transfer learning tests
- `verify_implementation.py` - Implementation verification script
- `verify_implementation_standalone.py` - Standalone verification

### corruption/
Tests for data corruption and validation strategies used in circuit discovery.

- `test_corruption_pipeline.py` - End-to-end corruption pipeline
- `test_distractor.py` - Distractor-based corruption
- `test_entity_swap.py` - Entity swapping corruption
- `test_paraphrase.py` - Paraphrase-based corruption
- `test_role_swap.py` - Role swapping corruption
- `test_token_swap.py` - Token swapping corruption
- `test_strategies_integration.py` - Corruption strategy integration
- `test_strategies_standalone.py` - Standalone strategy tests
- `test_validators.py` - Validator tests
- `test_practical_examples.py` - Practical corruption examples

### tasks/
Task-specific tests for circuit discovery on various benchmark tasks.

- `test_generic_task.py` - Generic task specification and execution
- `test_ioi_regression.py` - IOI task regression tests

### conftest.py
Pytest configuration and shared fixtures for all tests.

## Running Tests

### Run All Tests
```bash
pytest tests/
```

### Run Specific Test Category
```bash
pytest tests/unit/              # Run all unit tests
pytest tests/integration/       # Run integration tests
pytest tests/corruption/        # Run corruption tests
pytest tests/tasks/             # Run task tests
```

### Run Specific Test File
```bash
pytest tests/unit/test_api.py
pytest tests/integration/test_steering_ioi.py
```

### Run with Coverage
```bash
pytest --cov=src tests/
```

### Run with Verbose Output
```bash
pytest -v tests/
```

### Run Specific Test Function
```bash
pytest tests/unit/test_api.py::test_discover_circuit
```

## Test Organization Principles

1. **Unit Tests**: Fast, isolated tests of individual functions/classes
2. **Integration Tests**: Tests of multiple components working together
3. **Corruption Tests**: Tests of data corruption strategies
4. **Task Tests**: Tests specific to benchmark tasks (IOI, MMLU, etc.)

## Adding New Tests

When adding new tests:
1. Place unit tests in `unit/`
2. Place integration tests in `integration/`
3. Place specialized tests in appropriate subdirectory
4. Follow naming convention: `test_*.py` for test files, `test_*` for test functions
5. Use shared fixtures from `conftest.py`
6. Aim for good coverage but prioritize meaningful tests

## Continuous Integration

Tests are automatically run in CI on:
- Every pull request
- Every commit to main/development branches
- Pre-release verification

See `.github/workflows/` for CI configuration.

## Performance Notes

- Unit tests: < 1 minute total
- Integration tests: 5-15 minutes
- Full test suite: 20-30 minutes

Consider using `pytest-xdist` for parallel test execution on large test runs.
