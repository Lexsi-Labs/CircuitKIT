"""The flat-API error wrapper must not change the type of a HyperparameterError."""

import pytest

from circuitkit.utils.exceptions import (
    ValidationError,
    handle_errors,
    handle_exception,
)
from circuitkit.utils.hparams import HyperparameterError, HyperparameterWarning


class TestHyperparameterErrorPassthrough:
    def test_handle_exception_returns_it_unchanged(self):
        err = HyperparameterError("Invalid hyperparameter value(s):\n  - discovery.ig_steps must be >= 1, got 0.")
        assert handle_exception(err, {"operation": "discover_circuit"}) is err

    def test_an_escalated_warning_keeps_its_type(self):
        warning = HyperparameterWarning("num_examples=8 is outside the sensible range")
        assert handle_exception(warning, {"operation": "discover_circuit"}) is warning

        @handle_errors(context={"operation": "t"}, log_error=False)
        def escalated():
            raise warning

        with pytest.raises(HyperparameterWarning):
            escalated()

    def test_other_value_errors_are_still_wrapped(self):
        wrapped = handle_exception(ValueError("boom"), {"operation": "discover_circuit"})
        assert isinstance(wrapped, ValidationError)

    def test_decorator_keeps_the_type(self):
        @handle_errors(context={"operation": "t"}, log_error=False)
        def fails():
            raise HyperparameterError("bad value")

        with pytest.raises(HyperparameterError, match="bad value") as caught:
            fails()
        assert isinstance(caught.value, ValueError)

    def test_decorator_still_wraps_a_plain_value_error(self):
        @handle_errors(context={"operation": "t"}, log_error=False)
        def fails():
            raise ValueError("plain")

        with pytest.raises(ValidationError):
            fails()
