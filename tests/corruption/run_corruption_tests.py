#!/usr/bin/env python
"""
Clean test runner for CircuitKit corruption module tests.

This script:
1. Verifies the environment and dependencies
2. Installs the package in development mode if needed
3. Runs all corruption tests with clean output
4. Generates a summary report
"""

import subprocess
import sys
from pathlib import Path
from typing import Tuple


class CorruptionTestRunner:
    """Orchestrates running corruption tests with clean output and reporting."""

    def __init__(self):
        # Go up 3 levels: run_corruption_tests.py -> corruption -> tests -> circuitkit
        self.project_root = Path(__file__).resolve().parent.parent.parent

        # The corruption tests directory is just the parent of this script
        self.corruption_tests_dir = Path(__file__).resolve().parent
        self.results = {}

    def verify_environment(self) -> bool:
        """Verify Python environment and basic dependencies."""
        print("=" * 80)
        print("ENVIRONMENT VERIFICATION")
        print("=" * 80)

        print(f"Python: {sys.version}")
        print(f"Project Root: {self.project_root}")
        print(f"Tests Directory: {self.corruption_tests_dir}")

        if not self.corruption_tests_dir.exists():
            print(f"❌ Tests directory not found: {self.corruption_tests_dir}")
            return False

        print("✓ Tests directory found")

        # Check if pytest is available
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            print(f"✓ {result.stdout.strip()}")
        except Exception as e:
            print(f"❌ pytest not available: {e}")
            return False

        return True

    def install_package(self) -> bool:
        """Install circuitkit in development mode."""
        print("\n" + "=" * 80)
        print("PACKAGE INSTALLATION")
        print("=" * 80)

        try:
            print("Installing circuitkit in development mode...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-e", str(self.project_root)],
                capture_output=True,
                text=True,
                timeout=300,
            )

            if result.returncode == 0:
                print("✓ Package installed successfully")
                return True
            else:
                print("❌ Package installation failed")
                print(result.stderr)
                return False
        except subprocess.TimeoutExpired:
            print("❌ Installation timeout (exceeded 5 minutes)")
            return False
        except Exception as e:
            print(f"❌ Installation error: {e}")
            return False

    def run_individual_test(self, test_file: Path) -> Tuple[bool, str]:
        """
        Run an individual test file.

        Args:
            test_file: Path to test file

        Returns:
            Tuple of (success: bool, output: str)
        """
        test_name = test_file.stem
        print(f"\n{'─' * 80}")
        print(f"Running: {test_name}")
        print("─" * 80)

        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    str(test_file),
                    "-v",
                    "--tb=short",
                    "-x",  # Stop on first failure
                    "--color=yes",
                ],
                capture_output=False,  # Show output in real-time
                text=True,
                timeout=300,
            )

            success = result.returncode == 0
            return success, test_name

        except subprocess.TimeoutExpired:
            print("❌ Test timeout (exceeded 5 minutes)")
            return False, test_name
        except Exception as e:
            print(f"❌ Test execution error: {e}")
            return False, test_name

    def run_all_tests(self) -> bool:
        """Run all corruption tests via pytest."""
        print("\n" + "=" * 80)
        print("RUNNING ALL CORRUPTION TESTS")
        print("=" * 80)

        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    str(self.corruption_tests_dir),
                    "-v",
                    "--tb=short",
                    "--color=yes",
                    "-ra",  # Show summary of all test outcomes
                    "--durations=10",  # Show 10 slowest tests
                ],
                text=True,
                timeout=600,
            )

            return result.returncode == 0

        except subprocess.TimeoutExpired:
            print("❌ Test suite timeout (exceeded 10 minutes)")
            return False
        except Exception as e:
            print(f"❌ Test execution error: {e}")
            return False

    def run_with_coverage(self) -> bool:
        """Run tests with coverage report."""
        print("\n" + "=" * 80)
        print("RUNNING TESTS WITH COVERAGE")
        print("=" * 80)

        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    str(self.corruption_tests_dir),
                    "--cov=circuitkit.corruption",
                    "--cov-report=term-missing",
                    "--cov-report=html",
                    "-v",
                    "--tb=short",
                    "--color=yes",
                ],
                text=True,
                timeout=600,
            )

            if result.returncode == 0:
                print("\n✓ Coverage report generated: htmlcov/index.html")

            return result.returncode == 0

        except subprocess.TimeoutExpired:
            print("❌ Coverage test timeout")
            return False
        except Exception as e:
            print(f"⚠ Coverage report failed (tests may still have passed): {e}")
            # Don't fail entirely if coverage fails
            return True

    def generate_summary(self, all_passed: bool, coverage_available: bool) -> None:
        """Generate final summary report."""
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)

        if all_passed:
            print("✓ ALL TESTS PASSED")
        else:
            print("❌ SOME TESTS FAILED - Review output above for details")

        if coverage_available:
            print("✓ Coverage report available at: htmlcov/index.html")

        print("=" * 80)

    def run(self, install: bool = True, coverage: bool = False) -> int:
        """
        Execute the complete test pipeline.

        Args:
            install: Whether to install package before testing
            coverage: Whether to generate coverage report

        Returns:
            Exit code (0 = success, 1 = failure)
        """
        # Step 1: Verify environment
        if not self.verify_environment():
            print("\n❌ Environment verification failed")
            return 1

        # Step 2: Install package
        if install:
            if not self.install_package():
                print("\n⚠ Package installation failed, attempting to continue anyway...")

        # Step 3: Run all tests
        all_passed = self.run_all_tests()

        # Step 4: Run with coverage if requested
        coverage_available = False
        if coverage and all_passed:
            coverage_available = self.run_with_coverage()

        # Step 5: Summary
        self.generate_summary(all_passed, coverage_available)

        return 0 if all_passed else 1


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run CircuitKit corruption module tests")
    parser.add_argument("--no-install", action="store_true", help="Skip package installation")
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    parser.add_argument(
        "--skip-coverage", action="store_true", help="Skip coverage report even if tests pass"
    )

    args = parser.parse_args()

    runner = CorruptionTestRunner()
    exit_code = runner.run(
        install=not args.no_install, coverage=args.coverage and not args.skip_coverage
    )

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
