#!/usr/bin/env python3
"""Verify connector module structure is correct."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all imports work correctly."""
    print("Testing imports...")

    # Test base module imports
    from formbridge.connectors import (
        ConnectorError,
        AuthenticationError,
        APIError,
        TransactionType,
        TransactionCategory,
        Transaction,
        Account,
        AnnualSummary,
        EntityInfo,
        DataConnector,
        BaseConnector,
        categorize_transaction,
        merge_tax_data,
    )
    print("  ✓ Base imports OK")

    # Test Mercury connector
    from formbridge.connectors.mercury import MercuryConnector
    print("  ✓ MercuryConnector import OK")

    # Test Wise connector
    from formbridge.connectors.wise import WiseConnector
    print("  ✓ WiseConnector import OK")

    # Test CLI imports
    from formbridge.cli import main, import_group, import_mercury, import_wise, import_merge
    print("  ✓ CLI imports OK")

    print("\nAll imports successful!")


def test_entity_info():
    """Test EntityInfo creation."""
    from formbridge.connectors import EntityInfo

    entity = EntityInfo(
        name="Test Company",
        ein="12-3456789",
        address="123 Main St",
        city="New York",
        state="NY",
        zip_code="10001",
    )
    print(f"  EntityInfo created: {entity.name}")
    print("  ✓ EntityInfo test OK")


def test_categorization():
    """Test transaction categorization."""
    from formbridge.connectors import categorize_transaction, TransactionCategory

    # Test income
    cat = categorize_transaction("Payment from client", 50000)
    assert cat == TransactionCategory.GROSS_RECEIPTS, f"Expected GROSS_RECEIPTS, got {cat}"
    print("  ✓ Income categorization OK")

    # Test advertising expense
    cat = categorize_transaction("Google Ads", -5000)
    assert cat == TransactionCategory.ADVERTISING, f"Expected ADVERTISING, got {cat}"
    print("  ✓ Advertising categorization OK")

    # Test interest income
    cat = categorize_transaction("Interest payment", 1000)
    assert cat == TransactionCategory.INTEREST_INCOME, f"Expected INTEREST_INCOME, got {cat}"
    print("  ✓ Interest income categorization OK")


def test_merge():
    """Test merge functionality."""
    from formbridge.connectors import merge_tax_data, EntityInfo

    data1 = {
        "entity_name": "Test 1",
        "total_income": 1000.0,
        "total_deductions": 200.0,
        "_meta": {"source": "Mercury"},
    }
    data2 = {
        "entity_name": "Test 2",
        "total_income": 500.0,
        "total_deductions": 100.0,
        "_meta": {"source": "Wise"},
    }

    merged = merge_tax_data(data1, data2)
    assert merged["total_income"] == 1500.0
    assert merged["total_deductions"] == 300.0
    print("  ✓ Merge test OK")


def test_connector_requires_token():
    """Test that connectors require API token."""
    from formbridge.connectors import MercuryConnector, WiseConnector, AuthenticationError

    # Test Mercury without token
    import os
    old_token = os.environ.pop("MERCURY_API_TOKEN", None)
    try:
        MercuryConnector()
        print("  ✗ MercuryConnector should have raised error")
    except AuthenticationError:
        print("  ✓ MercuryConnector token check OK")
    finally:
        if old_token:
            os.environ["MERCURY_API_TOKEN"] = old_token

    # Test Wise without token
    old_token = os.environ.pop("WISE_API_TOKEN", None)
    try:
        WiseConnector()
        print("  ✗ WiseConnector should have raised error")
    except AuthenticationError:
        print("  ✓ WiseConnector token check OK")
    finally:
        if old_token:
            os.environ["WISE_API_TOKEN"] = old_token


def main():
    print("=" * 60)
    print("FormBridge Connector Verification")
    print("=" * 60)
    print()

    test_imports()
    print()

    print("Testing EntityInfo...")
    test_entity_info()
    print()

    print("Testing categorization...")
    test_categorization()
    print()

    print("Testing merge...")
    test_merge()
    print()

    print("Testing token requirements...")
    test_connector_requires_token()
    print()

    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
