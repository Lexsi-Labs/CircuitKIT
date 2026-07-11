"""
Auto-Schema Detector for HuggingFace Datasets

Automatically detects task type from column names and data patterns,
enabling CircuitKit to work with any HF dataset without manual configuration.

Key Components:
- SchemaAnalyzer: Inspects dataset structure and infers task type
- TaskTypeDetection: Result object with confidence scores and mappings
- Column Role Inference: Heuristics for detecting question, context, answer, etc.
- Task Type Scoring: Confidence-based detection with fallback handling
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class TaskType(str, Enum):
    """Supported task types for auto-detection."""

    QA = "qa"
    MCQ = "mcq"
    CLASSIFICATION = "classification"
    RANKING = "ranking"
    PARAPHRASE = "paraphrase"
    UNKNOWN = "unknown"


@dataclass
class TaskTypeDetection:
    """Result of task type detection."""

    task_type: TaskType
    confidence: float  # 0-1, higher = more confident
    suggested_mapping: Dict[str, str]  # column role -> actual column name
    detected_features: Dict[str, bool]  # feature -> detected
    reasoning: str  # Human-readable explanation


class SchemaAnalyzer:
    """
    Analyzes HuggingFace dataset schema and automatically detects task type.

    Uses heuristics based on:
    - Column names (question, context, answer, choices, etc.)
    - Data shapes (list vs string, cardinality)
    - Value ranges and distributions
    """

    # Pattern matching for column names
    QUESTION_PATTERNS = {
        "question",
        "query",
        "q",
        "premise",
        "sentence",
        "sentence1",
        "text_a",
        "input",
        "prompt",
        "query_text",
        "question_text",
        "instruction",
        "problem",
    }

    CONTEXT_PATTERNS = {
        "context",
        "passage",
        "document",
        "text",
        "body",
        "content",
        "document_text",
        "paragraph",
        "background",
        "evidence",
    }

    ANSWER_PATTERNS = {
        "answer",
        "answers",
        "label",
        "gold_label",
        "target",
        "answer_text",
        "response",
        "solution",
        "expected_output",
    }

    CHOICE_PATTERNS = {
        "choices",
        "options",
        "option_a",
        "option_b",
        "option_c",
        "option_d",
        "answer_choices",
        "candidates",
        "alternatives",
    }

    CORRECT_ANSWER_PATTERNS = {
        "correct_answer",
        "correct_choice",
        "correct_option",
        "correct_idx",
        "answer_idx",
        "answer_index",
        "correct_label",
        "gold_answer",
    }

    SCORE_PATTERNS = {
        "score",
        "relevance",
        "similarity",
        "rating",
        "rank",
        "relevance_score",
        "similarity_score",
        "label_score",
    }

    METADATA_PATTERNS = {"id", "idx", "example_id", "sample_id", "question_id", "pid", "uid", "qid"}

    @staticmethod
    def analyze(dataset: List[Dict[str, Any]], max_samples: int = 100) -> TaskTypeDetection:
        """
        Inspect dataset and detect task type.

        Args:
            dataset: List of example dicts from HF dataset
            max_samples: Max examples to analyze (for performance)

        Returns:
            TaskTypeDetection with detected type, confidence, and mapping
        """
        if not dataset:
            return TaskTypeDetection(
                task_type=TaskType.UNKNOWN,
                confidence=0.0,
                suggested_mapping={},
                detected_features={},
                reasoning="Empty dataset",
            )

        # Sample for analysis
        samples = dataset[: min(len(dataset), max_samples)]
        columns = set(samples[0].keys()) if samples else set()

        # Infer column roles
        inferred_roles = SchemaAnalyzer.infer_column_roles(columns, samples)

        # Detect task type and score
        task_type, confidence, detected_features, reasoning = SchemaAnalyzer.detect_task_type(
            columns, samples, inferred_roles
        )

        return TaskTypeDetection(
            task_type=task_type,
            confidence=confidence,
            suggested_mapping=inferred_roles,
            detected_features=detected_features,
            reasoning=reasoning,
        )

    @staticmethod
    def infer_column_roles(columns: set, samples: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Infer the role of each column (question, context, answer, etc.).

        Args:
            columns: Set of column names
            samples: Sample examples to analyze

        Returns:
            Dict mapping role -> column_name (only matched columns)
        """
        mapping = {}

        # Whether the dataset has an explicit choices-style column. When it
        # does, an integer-valued 'answer' column is almost certainly an MCQ
        # answer index rather than a free classification label.
        has_choice_column = any(c.lower() in SchemaAnalyzer.CHOICE_PATTERNS for c in columns)

        for col in columns:
            col_lower = col.lower()

            # Check patterns
            if col_lower in SchemaAnalyzer.QUESTION_PATTERNS:
                mapping["question"] = col
            elif col_lower in SchemaAnalyzer.CONTEXT_PATTERNS:
                mapping["context"] = col
            elif col_lower in SchemaAnalyzer.CORRECT_ANSWER_PATTERNS:
                # Check correct_answer patterns BEFORE generic answer patterns
                mapping["correct_answer_idx"] = col
            elif col_lower in SchemaAnalyzer.ANSWER_PATTERNS:
                # For 'answer' column, check if it contains text or indices.
                sample_val = None
                for s in samples:
                    if col in s and s[col] is not None:
                        sample_val = s[col]
                        break
                # An integer 'answer' alongside a choices column is an MCQ
                # answer index; otherwise it's a classification label.
                # bool is a subclass of int but is a classification label.
                if (
                    isinstance(sample_val, int)
                    and not isinstance(sample_val, bool)
                    and has_choice_column
                ):
                    mapping["correct_answer_idx"] = col
                else:
                    mapping["answer"] = col
            elif col_lower in SchemaAnalyzer.CHOICE_PATTERNS:
                mapping["choices"] = col
            elif col_lower in SchemaAnalyzer.SCORE_PATTERNS:
                mapping["score"] = col
            elif col_lower in SchemaAnalyzer.METADATA_PATTERNS:
                mapping["id"] = col

            # Infer from data type/shape
            else:
                inferred = SchemaAnalyzer._infer_from_data(col, samples)
                if inferred:
                    mapping[inferred] = col

        return mapping

    @staticmethod
    def _infer_from_data(col_name: str, samples: List[Dict[str, Any]]) -> Optional[str]:
        """
        Infer column role from data type and characteristics.

        Returns: role name or None if unknown
        """
        if not samples:
            return None

        # Get non-None sample
        sample_value = None
        for s in samples:
            if col_name in s and s[col_name] is not None:
                sample_value = s[col_name]
                break

        if sample_value is None:
            return None

        # List of strings/dicts -> likely choices or answers
        if isinstance(sample_value, list):
            if all(isinstance(x, str) for x in sample_value):
                return "answers"  # Multiple answers
            elif all(isinstance(x, dict) for x in sample_value):
                return "choices"  # List of choice objects
            return "answers"

        # Integer in small range -> likely answer index
        if isinstance(sample_value, int):
            if 0 <= sample_value < 100:
                return "correct_answer_idx"

        # Float -> likely score/similarity
        if isinstance(sample_value, float):
            if 0.0 <= sample_value <= 1.0:
                return "score"
            return "score"

        # String field -> could be text, answer, context
        if isinstance(sample_value, str):
            if len(sample_value) > 200:
                return "context"  # Longer text = context
            return None  # Generic string, check name

        return None

    @staticmethod
    def detect_task_type(
        columns: set,
        samples: List[Dict[str, Any]],
        inferred_roles: Dict[str, str],
    ) -> Tuple[TaskType, float, Dict[str, bool], str]:
        """
        Detect task type from columns and inferred roles.

        Returns:
            (task_type, confidence, detected_features, reasoning)
        """
        features = {
            "has_question": "question" in inferred_roles,
            "has_context": "context" in inferred_roles,
            "has_answer": "answer" in inferred_roles,
            "has_answers": "answers" in inferred_roles,
            "has_choices": "choices" in inferred_roles,
            "has_correct_idx": "correct_answer_idx" in inferred_roles,
            "has_score": "score" in inferred_roles,
        }

        # QA Task: (question, context, answer)
        if features["has_question"] and features["has_context"] and features["has_answer"]:
            confidence = min(1.0, 0.9)  # High confidence
            reasoning = "Detected QA pattern: question + context + answer"
            return TaskType.QA, confidence, features, reasoning

        # MCQ Task: (text/question, choices, correct_answer_idx)
        # Only if we have explicit correct_answer_idx AND choices
        if features["has_choices"] and features["has_correct_idx"]:
            confidence = 0.85
            reasoning = "Detected MCQ pattern: choices + correct answer index"
            return TaskType.MCQ, confidence, features, reasoning

        # MCQ with alternative pattern: (question, choices, answer as integer)
        if features["has_question"] and features["has_choices"] and features["has_answer"]:
            # Check if answer is an integer (likely index)
            if "answer" in inferred_roles:
                answer_col = inferred_roles["answer"]
                sample_val = None
                for s in samples:
                    if answer_col in s and s[answer_col] is not None:
                        sample_val = s[answer_col]
                        break
                if isinstance(sample_val, int):
                    confidence = 0.85
                    reasoning = "Detected MCQ pattern: question + choices + integer answer"
                    return TaskType.MCQ, confidence, features, reasoning

        # Ranking Task: (query, multiple_answers/candidates, scores)
        if features["has_answers"] or (features["has_question"] and features["has_score"]):
            confidence = 0.80
            reasoning = "Detected ranking pattern: query with multiple answers/scores"
            return TaskType.RANKING, confidence, features, reasoning

        # Paraphrase/Similarity: (text_a, text_b, similarity/label)
        if SchemaAnalyzer._detect_paraphrase(columns, samples, inferred_roles):
            confidence = 0.75
            reasoning = "Detected paraphrase pattern: two text fields + similarity label"
            return TaskType.PARAPHRASE, confidence, features, reasoning

        # Classification: (text, label)
        if features["has_question"] and features["has_answer"]:
            # Check label cardinality
            if "answer" in inferred_roles:
                answer_col = inferred_roles["answer"]
                unique_values = set()
                for s in samples[:100]:
                    if answer_col in s:
                        unique_values.add(str(s[answer_col]))
                if len(unique_values) < 100:  # Few distinct labels
                    confidence = 0.80
                    reasoning = (
                        f"Detected classification pattern: {len(unique_values)} label classes"
                    )
                    return TaskType.CLASSIFICATION, confidence, features, reasoning

        # Fallback
        return TaskType.UNKNOWN, 0.0, features, "Could not confidently detect task type"

    @staticmethod
    def _detect_paraphrase(
        columns: set,
        samples: List[Dict[str, Any]],
        inferred_roles: Dict[str, str],
    ) -> bool:
        """Check if dataset looks like paraphrase/similarity task."""
        # Need at least 2 text fields
        text_cols = [c for c in columns if SchemaAnalyzer._is_text_field(c, samples)]
        if len(text_cols) < 2:
            return False

        # Check for a similarity-like label. Classification-style labels are
        # inferred under the 'answer' role, so accept either 'score' or
        # 'answer' (in addition to a literal 'label' role, if present).
        has_similarity = (
            "score" in inferred_roles or "label" in inferred_roles or "answer" in inferred_roles
        )
        return has_similarity

    @staticmethod
    def _is_text_field(col_name: str, samples: List[Dict[str, Any]]) -> bool:
        """Check if column contains mostly text strings.

        Uses a proportional threshold so the check works for small sample
        sets (a fixed minimum count would never fire on tiny datasets).
        """
        inspected = samples[:10]
        if not inspected:
            return False
        text_count = sum(1 for s in inspected if col_name in s and isinstance(s[col_name], str))
        # Require a clear majority of inspected rows to be strings.
        return text_count >= max(1, (len(inspected) + 1) // 2)

    @staticmethod
    def detect_qa_task(columns: set, samples: List[Dict[str, Any]]) -> bool:
        """Check for question/answer/context columns (SQuAD-style)."""
        inferred = SchemaAnalyzer.infer_column_roles(columns, samples)
        return "question" in inferred and "context" in inferred and "answer" in inferred

    @staticmethod
    def detect_mcq_task(columns: set, samples: List[Dict[str, Any]]) -> bool:
        """Check for multiple choice (MMLU, GLUE-style)."""
        inferred = SchemaAnalyzer.infer_column_roles(columns, samples)
        return "choices" in inferred and "correct_answer_idx" in inferred

    @staticmethod
    def detect_classification_task(columns: set, samples: List[Dict[str, Any]]) -> bool:
        """Check for text classification (GLUE SST-2 style)."""
        inferred = SchemaAnalyzer.infer_column_roles(columns, samples)
        if not ("question" in inferred and "answer" in inferred):
            return False

        # A 'context' column means this is QA (SQuAD-style), not plain
        # classification; 'choices' means MCQ. Exclude both.
        if "context" in inferred or "choices" in inferred:
            return False

        # Check label cardinality
        if samples:
            answer_col = inferred["answer"]
            labels = set()
            for s in samples[:100]:
                if answer_col in s:
                    labels.add(str(s[answer_col]))
            return len(labels) < 100

        return False

    @staticmethod
    def detect_ranking_task(columns: set, samples: List[Dict[str, Any]]) -> bool:
        """Check for ranking/retrieval (BEiR style)."""
        inferred = SchemaAnalyzer.infer_column_roles(columns, samples)
        return ("question" in inferred or "query" in inferred) and (
            "answers" in inferred or "candidates" in inferred or "score" in inferred
        )

    @staticmethod
    def detect_paraphrase_task(columns: set, samples: List[Dict[str, Any]]) -> bool:
        """Check for paraphrase/similarity (MRPC, QQP style)."""
        inferred = SchemaAnalyzer.infer_column_roles(columns, samples)
        text_cols = [c for c in columns if SchemaAnalyzer._is_text_field(c, samples)]
        has_score = "score" in inferred or "label" in inferred

        return len(text_cols) >= 2 and has_score

    @staticmethod
    def suggest_mapping(
        dataset: List[Dict[str, Any]],
        detected_type: TaskType,
    ) -> Dict[str, str]:
        """
        Suggest GenericTaskSpec schema mapping based on detected type.

        Args:
            dataset: Raw examples from HF dataset
            detected_type: Detected task type

        Returns:
            Dict mapping GenericTaskSpec schema keys to actual columns
            Includes 'prompt' and 'answer' at minimum.
        """
        if not dataset:
            return {}

        columns = set(dataset[0].keys())
        inferred = SchemaAnalyzer.infer_column_roles(columns, dataset)

        mapping = {}

        # Prompt/Input (required)
        if "question" in inferred:
            mapping["prompt"] = inferred["question"]
        elif "query" in inferred:
            mapping["prompt"] = inferred["query"]
        else:
            # Find longest string column
            for col in columns:
                if SchemaAnalyzer._is_text_field(col, dataset):
                    mapping["prompt"] = col
                    break

        # Answer/Output (required)
        if detected_type == TaskType.QA:
            if "answer" in inferred:
                mapping["answer"] = inferred["answer"]
            if "context" in inferred:
                mapping["context"] = inferred["context"]

        elif detected_type == TaskType.MCQ:
            if "choices" in inferred:
                mapping["choices"] = inferred["choices"]
            if "correct_answer_idx" in inferred:
                mapping["correct_choice_idx"] = inferred["correct_answer_idx"]

        elif detected_type == TaskType.RANKING:
            if "answers" in inferred:
                mapping["answers"] = inferred["answers"]
            elif "answer" in inferred:
                mapping["answer"] = inferred["answer"]
            if "score" in inferred:
                mapping["score"] = inferred["score"]

        elif detected_type == TaskType.PARAPHRASE:
            # Need two text fields
            text_cols = [c for c in columns if SchemaAnalyzer._is_text_field(c, dataset)]
            if len(text_cols) >= 2:
                mapping["prompt"] = text_cols[0]
                mapping["comparison_text"] = text_cols[1]
            if "score" in inferred:
                mapping["label"] = inferred["score"]

        else:  # CLASSIFICATION or UNKNOWN
            if "answer" in inferred:
                mapping["answer"] = inferred["answer"]
            elif "label" in inferred:
                mapping["answer"] = inferred["label"]

        # Metadata
        if "id" in inferred:
            mapping["id"] = inferred["id"]

        return mapping
