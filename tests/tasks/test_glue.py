"""
Tests for GLUE task implementation.

Tests GLUETaskSpec for:
- Task initialization and validation
- Configuration validation
- Example formatting
- Dataset loading (mocked)
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

from circuitkit.tasks.builtins.glue import GLUE_TASKS, GLUETaskSpec, create_glue_task


class TestGLUETaskInit:
    """Test GLUE task initialization."""

    def test_valid_tasks(self):
        """Test that all valid GLUE tasks can be instantiated."""
        for task_name in GLUE_TASKS.keys():
            task = GLUETaskSpec(task_name)
            assert task.task_name == task_name
            # `name` is the stable task family identifier ("glue"); the
            # active subtask is tracked separately via `task_name`.
            assert task.name == "glue"

    def test_default_task(self):
        """Test default task is SST-2."""
        task = GLUETaskSpec()
        assert task.task_name == "sst2"

    def test_invalid_task(self):
        """Test that invalid task raises ValueError."""
        with pytest.raises(ValueError, match="Invalid GLUE task"):
            GLUETaskSpec("invalid_task")

    def test_case_insensitive(self):
        """Test task name is case-insensitive."""
        task1 = GLUETaskSpec("SST2")
        task2 = GLUETaskSpec("sst2")
        assert task1.task_name == task2.task_name


class TestGLUEValidateConfig:
    """Test GLUE configuration validation."""

    @pytest.fixture
    def task(self):
        return GLUETaskSpec("sst2")

    def test_valid_config(self, task):
        """Test that valid config passes validation."""
        config = {
            "algorithm": "eap",
            "level": "node",
            "batch_size": 4,
        }
        # Should not raise
        task.validate_discovery_config(config)

    def test_invalid_algorithm(self, task):
        """Test that invalid algorithm raises."""
        config = {
            "algorithm": "invalid",
            "level": "node",
        }
        with pytest.raises(ValueError, match="does not support algorithm"):
            task.validate_discovery_config(config)

    def test_invalid_level(self, task):
        """Test that invalid level raises."""
        config = {
            "algorithm": "eap",
            "level": "invalid",
        }
        with pytest.raises(ValueError, match="invalid 'level'"):
            task.validate_discovery_config(config)

    def test_ibcircuit_invalid_scope(self, task):
        """Test that invalid IBCircuit scope raises."""
        config = {
            "algorithm": "ibcircuit",
            "level": "node",
            "scope": "invalid",
        }
        with pytest.raises(ValueError, match="invalid 'scope'"):
            task.validate_discovery_config(config)

    def test_negative_batch_size(self, task):
        """Test that negative batch_size raises."""
        config = {
            "algorithm": "eap",
            "level": "node",
            "batch_size": -1,
        }
        with pytest.raises(ValueError, match="invalid 'batch_size'"):
            task.validate_discovery_config(config)

    def test_all_algorithms(self, task):
        """Test all supported algorithms pass validation."""
        for algo in ["eap", "eap-ig", "ibcircuit"]:
            config = {
                "algorithm": algo,
                "level": "node",
            }
            task.validate_discovery_config(config)


class TestGLUEFormatExample:
    """Test GLUE example formatting."""

    @pytest.fixture
    def sst2_task(self):
        return GLUETaskSpec("sst2")

    @pytest.fixture
    def mrpc_task(self):
        return GLUETaskSpec("mrpc")

    @pytest.fixture
    def qqp_task(self):
        return GLUETaskSpec("qqp")

    def test_format_sst2(self, sst2_task):
        """Test SST-2 example formatting."""
        item = {
            "sentence": "This movie is great!",
            "label": 1,
        }
        result = sst2_task._format_glue_example(item)
        assert result is not None
        assert "prompt" in result
        assert "answer" in result
        assert result["answer"] in ["positive", "negative"]

    def test_format_sst2_negative(self, sst2_task):
        """Test SST-2 negative label."""
        item = {
            "sentence": "This is bad.",
            "label": 0,
        }
        result = sst2_task._format_glue_example(item)
        assert result["answer"] == "negative"

    def test_format_mrpc(self, mrpc_task):
        """Test MRPC example formatting."""
        item = {
            "sentence1": "This is a test.",
            "sentence2": "This is a test.",
            "label": 1,
        }
        result = mrpc_task._format_glue_example(item)
        assert result is not None
        assert "prompt" in result
        assert result["answer"] == "paraphrase"

    def test_format_mrpc_not_paraphrase(self, mrpc_task):
        """Test MRPC non-paraphrase."""
        item = {
            "sentence1": "Sentence A",
            "sentence2": "Sentence B",
            "label": 0,
        }
        result = mrpc_task._format_glue_example(item)
        assert result["answer"] == "not_paraphrase"

    def test_format_qqp(self, qqp_task):
        """Test QQP example formatting."""
        item = {
            "question1": "What is AI?",
            "question2": "What is artificial intelligence?",
            "label": 1,
        }
        result = qqp_task._format_glue_example(item)
        assert result is not None
        assert result["answer"] == "duplicate"

    def test_format_invalid(self, sst2_task):
        """Test formatting with missing fields."""
        item = {}
        result = sst2_task._format_glue_example(item)
        # Should handle gracefully (return None or empty)
        assert result is None or isinstance(result, dict)

    def test_format_cola(self):
        """Test CoLA example formatting."""
        task = GLUETaskSpec("cola")
        item = {
            "sentence": "This is a grammatically correct sentence.",
            "label": 1,
        }
        result = task._format_glue_example(item)
        assert result["answer"] in ["acceptable", "not_acceptable"]


class TestGLUEFactory:
    """Test GLUE factory function."""

    def test_create_glue_task(self):
        """Test create_glue_task factory function."""
        task = create_glue_task("sst2")
        assert isinstance(task, GLUETaskSpec)
        assert task.task_name == "sst2"

    def test_create_glue_task_default(self):
        """Test factory defaults to sst2."""
        task = create_glue_task()
        assert task.task_name == "sst2"

    def test_create_glue_task_invalid(self):
        """Test factory with invalid task raises."""
        with pytest.raises(ValueError):
            create_glue_task("invalid")


class TestGLUEDataLoading:
    """Test GLUE data loading (mocked)."""

    @patch("circuitkit.tasks.builtins.glue.load_dataset")
    def test_load_data(self, mock_load):
        """Test data loading with mocked HF dataset."""
        task = GLUETaskSpec("sst2")

        # Mock dataset
        mock_dataset = [
            {"sentence": "Good movie", "label": 1},
            {"sentence": "Bad movie", "label": 0},
        ]
        mock_load.return_value = mock_dataset

        config = {
            "split": "validation",
            "samples_per_split": None,
        }

        # Create mock model
        mock_model = Mock()

        examples = task._get_or_load_glue_data(config, mock_model)
        assert len(examples) > 0

    @patch("circuitkit.tasks.builtins.glue.load_dataset")
    def test_load_with_sampling(self, mock_load):
        """Test data loading with sampling."""
        task = GLUETaskSpec("sst2")

        # Mock large dataset
        mock_dataset = [{"sentence": f"Sentence {i}", "label": i % 2} for i in range(100)]
        # Mock a HF Dataset: supports len(), shuffle(), select() and iteration.
        mock_dataset_obj = MagicMock()
        mock_dataset_obj.__len__.return_value = 100
        mock_dataset_obj.shuffle.return_value = mock_dataset_obj
        mock_dataset_obj.select.return_value = mock_dataset[:10]
        mock_dataset_obj.__iter__.return_value = iter(mock_dataset[:10])
        mock_load.return_value = mock_dataset_obj

        config = {
            "split": "validation",
            "samples_per_split": 10,
        }

        mock_model = Mock()

        examples = task._get_or_load_glue_data(config, mock_model)
        assert len(examples) <= 10


class TestGLUEBuildDataloader:
    """Test GLUE dataloader building."""

    def test_build_without_model_raises(self):
        """Test that building without model raises."""
        task = GLUETaskSpec("sst2")
        config = {"algorithm": "eap", "level": "node"}

        with pytest.raises(ValueError, match="requires model"):
            task.build_dataloader(None, config, "cpu")

    @patch.object(GLUETaskSpec, "_build_eap_dataloader")
    def test_build_eap(self, mock_eap):
        """Test building EAP dataloader."""
        task = GLUETaskSpec("sst2")
        config = {"algorithm": "eap", "level": "node"}
        mock_model = Mock()

        mock_eap.return_value = Mock()

        task.build_dataloader(mock_model, config, "cpu")
        mock_eap.assert_called_once()

    @patch.object(GLUETaskSpec, "_build_ibcircuit_dataloader")
    def test_build_ibcircuit(self, mock_ib):
        """Test building IBCircuit dataloader."""
        task = GLUETaskSpec("sst2")
        config = {"algorithm": "ibcircuit", "level": "node"}
        mock_model = Mock()

        mock_ib.return_value = Mock()

        task.build_dataloader(mock_model, config, "cpu")
        mock_ib.assert_called_once()
