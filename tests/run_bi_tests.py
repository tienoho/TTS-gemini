#!/usr/bin/env python3
"""
Business Intelligence Test Runner
Comprehensive test suite for BI functionality
"""

import pytest
import sys
import os
from datetime import datetime
import argparse

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_bi_tests():
    """Run all BI-related tests."""
    print("=" * 60)
    print("BUSINESS INTELLIGENCE TEST SUITE")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Test files to run
    test_files = [
        'tests/test_bi_service.py',
        'tests/test_bi_analytics.py',
        'tests/test_bi_dashboard.py',
        'tests/test_bi_reporting.py',
        'tests/test_bi_api.py'
    ]

    # Check if test files exist
    missing_files = []
    for test_file in test_files:
        if not os.path.exists(test_file):
            missing_files.append(test_file)

    if missing_files:
        print("❌ Missing test files:")
        for file in missing_files:
            print(f"   - {file}")
        print()
        return False

    print("📋 Test Files Found:")
    for test_file in test_files:
        print(f"   ✓ {test_file}")
    print()

    # Run tests with coverage
    print("🚀 Running Tests...")
    print("-" * 40)

    # Configure pytest
    pytest_args = [
        '--verbose',
        '--tb=short',
        '--strict-markers',
        '--disable-warnings',
        '--cov=utils.bi_service',
        '--cov=utils.bi_analytics',
        '--cov=utils.bi_dashboard',
        '--cov=utils.bi_reporting',
        '--cov=routes.business_intelligence',
        '--cov=models.business_intelligence',
        '--cov-report=term-missing',
        '--cov-report=html:htmlcov',
        '--cov-fail-under=80'
    ]

    # Add test files
    pytest_args.extend(test_files)

    # Run pytest
    exit_code = pytest.main(pytest_args)

    print()
    print("=" * 60)
    print("TEST RESULTS")
    print("=" * 60)

    if exit_code == 0:
        print("✅ All tests passed!")
        print("🎉 Business Intelligence system is ready for production!")
        return True
    else:
        print("❌ Some tests failed!")
        print("🔧 Please review the test output above and fix any issues.")
        return False


def run_specific_test(test_name):
    """Run a specific test module."""
    print(f"Running specific test: {test_name}")
    print("-" * 40)

    test_file = f'tests/test_{test_name}.py'

    if not os.path.exists(test_file):
        print(f"❌ Test file not found: {test_file}")
        return False

    pytest_args = [
        '--verbose',
        '--tb=short',
        test_file
    ]

    exit_code = pytest.main(pytest_args)

    return exit_code == 0


def show_test_coverage():
    """Show test coverage report."""
    coverage_file = 'htmlcov/index.html'

    if os.path.exists(coverage_file):
        print("📊 Coverage report generated: htmlcov/index.html")
        print("   Open in browser to view detailed coverage report")
    else:
        print("⚠️  No coverage report found")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Business Intelligence Test Runner')
    parser.add_argument('--test', help='Run specific test module (e.g., bi_service, bi_analytics)')
    parser.add_argument('--coverage', action='store_true', help='Show coverage report after tests')

    args = parser.parse_args()

    if args.test:
        success = run_specific_test(args.test)
        if args.coverage:
            show_test_coverage()
        sys.exit(0 if success else 1)

    # Run all tests
    success = run_bi_tests()

    if args.coverage:
        show_test_coverage()

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()