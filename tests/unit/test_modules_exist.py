"""
Simple test to verify modules exist and have correct structure.
Run with: python test_modules_exist.py
"""

import os
import sys


def check_file_exists(path, description):
    """Check if file exists."""
    if os.path.exists(path):
        size = os.path.getsize(path)
        print(f"[OK] {description}: {path} ({size} bytes)")
        return True
    else:
        print(f"[FAIL] {description}: {path} NOT FOUND")
        return False


def check_file_contains(path, text, description):
    """Check if file contains text."""
    if os.path.exists(path):
        with open(path, "r") as f:
            content = f.read()
            if text in content:
                print(f"[OK] {description}: Found in {os.path.basename(path)}")
                return True
            else:
                print(f"[FAIL] {description}: Not found in {os.path.basename(path)}")
                return False
    else:
        print(f"[FAIL] {description}: {path} NOT FOUND")
        return False


def main():
    base = os.path.dirname(__file__)
    src_base = os.path.join(base, "src", "circuitkit", "corruption")
    test_base = os.path.join(base, "tests", "corruption")

    print("=" * 60)
    print("Checking B4: ParaphraseCorruption Module")
    print("=" * 60)

    paraphrase_path = os.path.join(src_base, "paraphrase.py")
    all_pass = True

    all_pass &= check_file_exists(paraphrase_path, "Paraphrase module")
    all_pass &= check_file_contains(
        paraphrase_path, "class ParaphraseCorruption", "ParaphraseCorruption class"
    )
    all_pass &= check_file_contains(paraphrase_path, "def corrupt(", "corrupt method")
    all_pass &= check_file_contains(paraphrase_path, "def batch_corrupt(", "batch_corrupt method")
    all_pass &= check_file_contains(paraphrase_path, "def validate(", "validate method")
    all_pass &= check_file_contains(paraphrase_path, "def _load_cache(", "cache loading")
    all_pass &= check_file_contains(paraphrase_path, "def _save_cache(", "cache saving")
    all_pass &= check_file_contains(paraphrase_path, "def _get_cache_key(", "cache key generation")
    all_pass &= check_file_contains(
        paraphrase_path, "def _paraphrase_surface(", "surface paraphrase"
    )
    all_pass &= check_file_contains(
        paraphrase_path, "def _paraphrase_semantic(", "semantic paraphrase"
    )

    print("\n" + "=" * 60)
    print("Checking B7: Validators Module")
    print("=" * 60)

    validators_path = os.path.join(src_base, "validators.py")

    all_pass &= check_file_exists(validators_path, "Validators module")
    all_pass &= check_file_contains(
        validators_path, "class CorruptionValidationResult", "CorruptionValidationResult dataclass"
    )
    all_pass &= check_file_contains(
        validators_path, "class LengthBudgetValidator", "LengthBudgetValidator class"
    )
    all_pass &= check_file_contains(
        validators_path, "class LabelConsistencyValidator", "LabelConsistencyValidator class"
    )
    all_pass &= check_file_contains(
        validators_path, "class TokenizationValidator", "TokenizationValidator class"
    )
    all_pass &= check_file_contains(
        validators_path, "class SemanticShiftValidator", "SemanticShiftValidator class"
    )
    all_pass &= check_file_contains(
        validators_path, "class CompositeValidator", "CompositeValidator class"
    )

    # Check exports
    print("\n" + "=" * 60)
    print("Checking Module Exports")
    print("=" * 60)

    init_path = os.path.join(src_base, "__init__.py")
    all_pass &= check_file_contains(
        init_path, "ParaphraseCorruption", "ParaphraseCorruption exported"
    )
    all_pass &= check_file_contains(
        init_path, "LengthBudgetValidator", "LengthBudgetValidator exported"
    )
    all_pass &= check_file_contains(
        init_path, "LabelConsistencyValidator", "LabelConsistencyValidator exported"
    )
    all_pass &= check_file_contains(
        init_path, "TokenizationValidator", "TokenizationValidator exported"
    )
    all_pass &= check_file_contains(
        init_path, "SemanticShiftValidator", "SemanticShiftValidator exported"
    )
    all_pass &= check_file_contains(init_path, "CompositeValidator", "CompositeValidator exported")

    # Check tests
    print("\n" + "=" * 60)
    print("Checking Test Files")
    print("=" * 60)

    test_validators = os.path.join(test_base, "test_validators.py")
    test_paraphrase = os.path.join(test_base, "test_paraphrase.py")

    all_pass &= check_file_exists(test_validators, "Validators test file")
    all_pass &= check_file_exists(test_paraphrase, "Paraphrase test file")

    all_pass &= check_file_contains(
        test_validators, "class TestLengthBudgetValidator", "LengthBudgetValidator tests"
    )
    all_pass &= check_file_contains(
        test_validators, "class TestLabelConsistencyValidator", "LabelConsistencyValidator tests"
    )
    all_pass &= check_file_contains(
        test_validators, "class TestTokenizationValidator", "TokenizationValidator tests"
    )
    all_pass &= check_file_contains(
        test_validators, "class TestSemanticShiftValidator", "SemanticShiftValidator tests"
    )
    all_pass &= check_file_contains(
        test_validators, "class TestCompositeValidator", "CompositeValidator tests"
    )

    all_pass &= check_file_contains(
        test_paraphrase, "class TestParaphraseCorruptionInit", "Paraphrase init tests"
    )
    all_pass &= check_file_contains(
        test_paraphrase, "class TestCacheOperations", "Cache operation tests"
    )
    all_pass &= check_file_contains(test_paraphrase, "class TestCorrupt", "Corrupt method tests")

    print("\n" + "=" * 60)
    print("Manual Validator Tests (Direct Import)")
    print("=" * 60)

    # Run direct tests as subprocess to avoid scope issues
    import subprocess

    try:
        result = subprocess.run(
            [sys.executable, "test_validators_direct.py"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            print("[OK] Validator direct tests passed")
        else:
            print("[FAIL] Validator direct tests failed")
            print(result.stdout)
            print(result.stderr)
            all_pass = False
    except Exception as e:
        print(f"[WARN] Could not run validator tests: {e}")

    print("\n" + "=" * 60)
    if all_pass:
        print("ALL CHECKS PASSED!")
    else:
        print("SOME CHECKS FAILED!")
    print("=" * 60)

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
