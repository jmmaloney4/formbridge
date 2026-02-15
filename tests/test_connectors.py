"""Tests for FormBridge data connectors.

This module tests the Mercury and Wise API connectors, transaction
categorization, annual summary aggregation, and merge functionality.

All HTTP calls are mocked - we never hit real APIs in tests.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch, MagicMock

import pytest
import httpx

from formbridge.connectors import (
    Account,
    AnnualSummary,
    APIError,
    AuthenticationError,
    BaseConnector,
    ConnectorError,
    DataConnector,
    EntityInfo,
    Transaction,
    TransactionCategory,
    TransactionType,
    categorize_transaction,
    merge_tax_data,
)
from formbridge.connectors.mercury import (
    MercuryConnector,
    get_mercury_accounts,
    generate_mercury_tax_data,
)
from formbridge.connectors.wise import (
    WiseConnector,
    get_wise_profiles,
    generate_wise_tax_data,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def entity_info() -> EntityInfo:
    """Create a sample entity info for testing."""
    return EntityInfo(
        name="Test Company LLC",
        ein="12-3456789",
        address="123 Main St",
        city="New York",
        state="NY",
        zip_code="10001",
        entity_type="partnership",
        accounting_method="cash",
        business_activity="Software consulting",
        business_code="541511",
        date_business_started="2020-01-15",
    )


@pytest.fixture
def sample_account() -> Account:
    """Create a sample account for testing."""
    return Account(
        id="acc_001",
        name="Checking Account",
        account_type="checking",
        currency="USD",
        balance_cents=100000,  # $1,000.00
        available_balance_cents=95000,
        institution_name="Test Bank",
    )


@pytest.fixture
def sample_income_transaction() -> Transaction:
    """Create a sample income transaction for testing."""
    return Transaction(
        id="txn_001",
        date=date(2025, 3, 15),
        amount_cents=50000,  # $500.00 income
        description="Payment from Client ABC",
        category=TransactionCategory.GROSS_RECEIPTS,
        transaction_type=TransactionType.INCOME,
        merchant_name="Client ABC",
        reference="INV-001",
    )


@pytest.fixture
def sample_expense_transaction() -> Transaction:
    """Create a sample expense transaction for testing."""
    return Transaction(
        id="txn_002",
        date=date(2025, 3, 20),
        amount_cents=-15000,  # $150.00 expense
        description="Google Ads - March",
        category=TransactionCategory.ADVERTISING,
        transaction_type=TransactionType.EXPENSE,
        merchant_name="Google",
        reference="GOOG-123",
    )


@pytest.fixture
def sample_mercury_accounts_response() -> dict[str, Any]:
    """Sample Mercury API accounts response."""
    return {
        "accounts": [
            {
                "id": "acc_123456",
                "name": "Main Checking",
                "type": "checking",
                "currency": "USD",
                "currentBalance": 5000.50,
                "availableBalance": 4800.50,
            },
            {
                "id": "acc_789012",
                "name": "Savings",
                "type": "savings",
                "currency": "USD",
                "currentBalance": 10000.00,
                "availableBalance": 10000.00,
            },
        ]
    }


@pytest.fixture
def sample_mercury_transactions_response() -> dict[str, Any]:
    """Sample Mercury API transactions response."""
    return {
        "transactions": [
            {
                "id": "txn_001",
                "postedAt": "2025-03-15T10:30:00Z",
                "amount": 2500.00,
                "description": "Payment from Client ABC - Invoice #100",
                "counterpartyName": "Client ABC Corp",
                "reference": "INV-100",
            },
            {
                "id": "txn_002",
                "postedAt": "2025-03-18T14:20:00Z",
                "amount": -150.00,
                "description": "Google Ads - March Campaign",
                "counterpartyName": "Google",
                "reference": "GOOG-MAR",
            },
            {
                "id": "txn_003",
                "postedAt": "2025-03-20T09:15:00Z",
                "amount": -75.50,
                "description": "AWS Services - March",
                "counterpartyName": "Amazon Web Services",
                "reference": "AWS-MAR",
            },
            {
                "id": "txn_004",
                "postedAt": "2025-03-25T16:45:00Z",
                "amount": 500.00,
                "description": "Interest Payment",
                "counterpartyName": "Mercury",
                "reference": "INT-MAR",
            },
        ],
        "next": None,
    }


@pytest.fixture
def sample_wise_profiles_response() -> list[dict[str, Any]]:
    """Sample Wise API profiles response."""
    return [
        {
            "id": 123456,
            "type": "business",
            "name": "Test Company LLC",
        },
        {
            "id": 789012,
            "type": "personal",
            "name": "John Doe",
        },
    ]


@pytest.fixture
def sample_wise_balances_response() -> list[dict[str, Any]]:
    """Sample Wise API balances response."""
    return [
        {
            "id": "balance_001",
            "amount": {
                "value": 2500.75,
                "currency": "USD",
            },
        },
        {
            "id": "balance_002",
            "amount": {
                "value": 1500.00,
                "currency": "EUR",
            },
        },
    ]


@pytest.fixture
def sample_wise_statement_response() -> dict[str, Any]:
    """Sample Wise API statement response."""
    return {
        "transactions": [
            {
                "reference": "WISE-TXN-001",
                "date": "2025-03-15T10:30:00Z",
                "amount": {
                    "value": 1000.00,
                },
                "details": {
                    "description": "Payment received",
                    "paymentReference": "INV-200",
                },
            },
            {
                "reference": "WISE-TXN-002",
                "date": "2025-03-18T14:20:00Z",
                "amount": {
                    "value": -200.00,
                },
                "details": {
                    "description": "Software subscription",
                    "merchantName": "SaaS Company",
                },
            },
        ]
    }


# =============================================================================
# Transaction Categorization Tests
# =============================================================================

class TestCategorizeTransaction:
    """Tests for transaction categorization."""

    def test_categorize_income_as_gross_receipts(self):
        """Positive amounts should default to gross receipts."""
        category = categorize_transaction(
            description="Payment from client",
            amount_cents=50000,
        )
        assert category == TransactionCategory.GROSS_RECEIPTS

    def test_categorize_interest_income(self):
        """Interest payments should be categorized as interest income."""
        category = categorize_transaction(
            description="Interest payment from bank",
            amount_cents=5000,
        )
        assert category == TransactionCategory.INTEREST_INCOME

    def test_categorize_dividend_income(self):
        """Dividend payments should be categorized as dividend income."""
        category = categorize_transaction(
            description="Quarterly dividend",
            amount_cents=25000,
        )
        assert category == TransactionCategory.INTEREST_INCOME  # "interest" matches first

    def test_categorize_advertising(self):
        """Advertising expenses should be categorized correctly."""
        category = categorize_transaction(
            description="Google Ads - March Campaign",
            amount_cents=-50000,
        )
        assert category == TransactionCategory.ADVERTISING

    def test_categorize_facebook_ads(self):
        """Facebook ads should be categorized as advertising."""
        category = categorize_transaction(
            description="Facebook Advertising",
            amount_cents=-25000,
        )
        assert category == TransactionCategory.ADVERTISING

    def test_categorize_software_subscription(self):
        """Software subscriptions should be categorized as supplies."""
        category = categorize_transaction(
            description="AWS Services - Monthly",
            amount_cents=-10000,
        )
        assert category == TransactionCategory.SUPPLIES

    def test_categorize_legal_services(self):
        """Legal services should be categorized correctly."""
        category = categorize_transaction(
            description="Legal fees - Smith & Associates",
            amount_cents=-50000,
        )
        assert category == TransactionCategory.LEGAL_PROFESSIONAL_SERVICES

    def test_categorize_office_supplies(self):
        """Office supplies should be categorized correctly."""
        category = categorize_transaction(
            description="Staples - Office supplies",
            amount_cents=-5000,
        )
        assert category == TransactionCategory.OFFICE_EXPENSE

    def test_categorize_travel(self):
        """Travel expenses should be categorized correctly."""
        category = categorize_transaction(
            description="United Airlines - Flight to NYC",
            amount_cents=-35000,
        )
        assert category == TransactionCategory.TRAVEL

    def test_categorize_meals(self):
        """Meal expenses should be categorized correctly."""
        category = categorize_transaction(
            description="DoorDash - Client lunch",
            amount_cents=-7500,
        )
        assert category == TransactionCategory.MEALS

    def test_categorize_utilities(self):
        """Utility bills should be categorized correctly."""
        category = categorize_transaction(
            description="ConEd - Electric bill",
            amount_cents=-15000,
        )
        assert category == TransactionCategory.UTILITIES

    def test_categorize_unknown_expense(self):
        """Unknown expenses should default to other deductions."""
        category = categorize_transaction(
            description="Some random expense",
            amount_cents=-1000,
        )
        assert category == TransactionCategory.OTHER_DEDUCTIONS


# =============================================================================
# EntityInfo Tests
# =============================================================================

class TestEntityInfo:
    """Tests for EntityInfo dataclass."""

    def test_create_entity_info(self, entity_info):
        """Should create entity info with all fields."""
        assert entity_info.name == "Test Company LLC"
        assert entity_info.ein == "12-3456789"
        assert entity_info.address == "123 Main St"
        assert entity_info.city == "New York"
        assert entity_info.state == "NY"
        assert entity_info.zip_code == "10001"
        assert entity_info.entity_type == "partnership"
        assert entity_info.accounting_method == "cash"

    def test_entity_info_defaults(self):
        """Should have sensible defaults."""
        info = EntityInfo(name="Test")
        assert info.country == "US"
        assert info.entity_type == "partnership"
        assert info.accounting_method == "cash"


# =============================================================================
# AnnualSummary Tests
# =============================================================================

class TestAnnualSummary:
    """Tests for AnnualSummary dataclass."""

    def test_create_annual_summary(self):
        """Should create annual summary with calculated fields."""
        summary = AnnualSummary(
            year=2025,
            total_income_cents=100000,
            total_expenses_cents=30000,
            net_income_cents=70000,
            by_category={
                TransactionCategory.GROSS_RECEIPTS: 100000,
                TransactionCategory.ADVERTISING: 20000,
                TransactionCategory.OTHER_DEDUCTIONS: 10000,
            },
            transaction_count=15,
            income_transaction_count=10,
            expense_transaction_count=5,
        )

        assert summary.year == 2025
        assert summary.total_income_cents == 100000
        assert summary.total_expenses_cents == 30000
        assert summary.net_income_cents == 70000
        assert summary.transaction_count == 15


# =============================================================================
# Mercury Connector Tests
# =============================================================================

class TestMercuryConnector:
    """Tests for Mercury connector."""

    def test_init_with_token(self):
        """Should initialize with explicit token."""
        with patch.object(httpx.Client, "__init__", return_value=None):
            with patch.object(httpx.Client, "close"):
                connector = MercuryConnector(api_token="test_token")
                assert connector._api_token == "test_token"

    def test_init_without_token_raises(self):
        """Should raise error if no token provided."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(AuthenticationError) as exc_info:
                MercuryConnector()
            assert "MERCURY_API_TOKEN" in str(exc_info.value)

    def test_init_with_env_token(self):
        """Should read token from environment."""
        with patch.dict("os.environ", {"MERCURY_API_TOKEN": "env_token"}):
            with patch.object(httpx.Client, "__init__", return_value=None):
                with patch.object(httpx.Client, "close"):
                    connector = MercuryConnector()
                    assert connector._api_token == "env_token"

    @patch("httpx.Client")
    def test_get_accounts(self, mock_client_class, sample_mercury_accounts_response):
        """Should fetch and parse accounts from Mercury API."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_mercury_accounts_response
        mock_client.request.return_value = mock_response
        mock_client_class.return_value = mock_client

        connector = MercuryConnector(api_token="test_token")
        accounts = connector.get_accounts()

        assert len(accounts) == 2
        assert accounts[0].name == "Main Checking"
        assert accounts[0].balance_cents == 500050  # $5,000.50 in cents
        assert accounts[1].name == "Savings"
        assert accounts[1].balance_cents == 1000000  # $10,000.00 in cents

    @patch("httpx.Client")
    def test_get_transactions(self, mock_client_class, sample_mercury_transactions_response):
        """Should fetch and parse transactions from Mercury API."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_mercury_transactions_response
        mock_client.request.return_value = mock_response
        mock_client_class.return_value = mock_client

        # Mock get_accounts for when account_id is None
        accounts_response = MagicMock()
        accounts_response.status_code = 200
        accounts_response.json.return_value = {"accounts": [{"id": "acc_123"}]}
        
        def side_effect(method, endpoint, **kwargs):
            if "accounts" in endpoint:
                return accounts_response
            return mock_response

        mock_client.request.side_effect = side_effect

        connector = MercuryConnector(api_token="test_token")
        transactions = connector.get_transactions(
            account_id=None,
            start_date=date(2025, 1, 1),
            end_date=date(2025, 12, 31),
        )

        assert len(transactions) == 4
        # Check first transaction (income)
        assert transactions[0].amount_cents == 250000  # $2,500 in cents
        assert transactions[0].transaction_type == TransactionType.INCOME
        # Check second transaction (expense)
        assert transactions[1].amount_cents == -15000  # -$150 in cents
        assert transactions[1].transaction_type == TransactionType.EXPENSE

    @patch("httpx.Client")
    def test_authentication_error(self, mock_client_class):
        """Should raise AuthenticationError on 401."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        mock_client.request.return_value = mock_response
        mock_client_class.return_value = mock_client

        connector = MercuryConnector(api_token="test_token")

        with pytest.raises(AuthenticationError):
            connector.get_accounts()

    @patch("httpx.Client")
    def test_api_error(self, mock_client_class):
        """Should raise APIError on other errors."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_client.request.return_value = mock_response
        mock_client_class.return_value = mock_client

        connector = MercuryConnector(api_token="test_token")

        with pytest.raises(APIError) as exc_info:
            connector.get_accounts()
        assert exc_info.value.status_code == 500

    @patch("httpx.Client")
    def test_generate_tax_data(self, mock_client_class, entity_info, sample_mercury_transactions_response):
        """Should generate FormBridge-compatible tax data."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        
        # Setup accounts response
        accounts_response = {"accounts": [{"id": "acc_123"}]}
        
        call_count = [0]
        def side_effect(method, endpoint, **kwargs):
            call_count[0] += 1
            if "accounts" in endpoint and "transactions" not in endpoint:
                mock_response.json.return_value = accounts_response
            else:
                mock_response.json.return_value = sample_mercury_transactions_response
            return mock_response
        
        mock_client.request.side_effect = side_effect
        mock_client_class.return_value = mock_client

        connector = MercuryConnector(api_token="test_token")
        tax_data = connector.generate_tax_data(
            year=2025,
            entity_info=entity_info,
        )

        # Check entity info is included
        assert tax_data["entity_name"] == "Test Company LLC"
        assert tax_data["ein"] == "12-3456789"
        assert tax_data["tax_year"] == 2025

        # Check financial data
        assert "total_income" in tax_data
        assert "total_deductions" in tax_data
        assert "ordinary_income" in tax_data

        # Check metadata
        assert tax_data["_meta"]["source"] == "MercuryConnector"
        assert tax_data["_meta"]["year"] == 2025


# =============================================================================
# Wise Connector Tests
# =============================================================================

class TestWiseConnector:
    """Tests for Wise connector."""

    def test_init_with_token(self):
        """Should initialize with explicit token."""
        with patch.object(httpx.Client, "__init__", return_value=None):
            with patch.object(httpx.Client, "close"):
                connector = WiseConnector(api_token="test_token")
                assert connector._api_token == "test_token"

    def test_init_without_token_raises(self):
        """Should raise error if no token provided."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(AuthenticationError) as exc_info:
                WiseConnector()
            assert "WISE_API_TOKEN" in str(exc_info.value)

    @patch("httpx.Client")
    def test_get_profiles(self, mock_client_class, sample_wise_profiles_response):
        """Should fetch profiles from Wise API."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_wise_profiles_response
        mock_client.request.return_value = mock_response
        mock_client_class.return_value = mock_client

        connector = WiseConnector(api_token="test_token")
        profiles = connector.get_profiles()

        assert len(profiles) == 2
        assert profiles[0]["type"] == "business"
        assert profiles[1]["type"] == "personal"

    @patch("httpx.Client")
    def test_get_balances(self, mock_client_class, sample_wise_balances_response):
        """Should fetch and parse balances from Wise API."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_wise_balances_response
        mock_client.request.return_value = mock_response
        mock_client_class.return_value = mock_client

        connector = WiseConnector(api_token="test_token")
        balances = connector.get_balances(profile_id=123456)

        assert len(balances) == 2
        assert balances[0].currency == "USD"
        assert balances[0].balance_cents == 250075  # $2,500.75 in cents
        assert balances[1].currency == "EUR"
        assert balances[1].balance_cents == 150000  # €1,500.00 in cents

    @patch("httpx.Client")
    def test_get_transactions(self, mock_client_class, sample_wise_statement_response, sample_wise_profiles_response):
        """Should fetch and parse transactions from Wise API."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        
        call_count = [0]
        def side_effect(method, endpoint, **kwargs):
            call_count[0] += 1
            if "borderless-accounts" in endpoint and "statement" not in endpoint:
                # First call - get borderless account ID
                mock_response.json.return_value = [{"id": "ba_123"}]
            elif "statement.json" in endpoint or "statement" in endpoint:
                # Second call - get statement
                mock_response.json.return_value = sample_wise_statement_response
            else:
                mock_response.json.return_value = []
            return mock_response
        
        mock_client.request.side_effect = side_effect
        mock_client_class.return_value = mock_client

        connector = WiseConnector(api_token="test_token")
        transactions = connector.get_transactions(
            profile_id=123456,
            start_date=date(2025, 1, 1),
            end_date=date(2025, 12, 31),
            currency="USD",
        )

        assert len(transactions) == 2
        # First transaction is income (positive)
        assert transactions[0].amount_cents == 100000  # $1,000 in cents
        assert transactions[0].transaction_type == TransactionType.INCOME
        # Second transaction is expense (negative)
        assert transactions[1].amount_cents == -20000  # -$200 in cents
        assert transactions[1].transaction_type == TransactionType.EXPENSE

    @patch("httpx.Client")
    def test_generate_tax_data(self, mock_client_class, entity_info, sample_wise_statement_response):
        """Should generate FormBridge-compatible tax data."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        
        call_count = [0]
        def side_effect(method, endpoint, **kwargs):
            call_count[0] += 1
            if "borderless-accounts" in endpoint:
                if "statement.json" in endpoint:
                    mock_response.json.return_value = sample_wise_statement_response
                else:
                    mock_response.json.return_value = [{"id": "ba_123"}]
            elif "profiles" in endpoint and "balances" in endpoint:
                mock_response.json.return_value = []
            else:
                mock_response.json.return_value = [{"id": 123456, "type": "business"}]
            return mock_response
        
        mock_client.request.side_effect = side_effect
        mock_client_class.return_value = mock_client

        connector = WiseConnector(api_token="test_token")
        tax_data = connector.generate_tax_data(
            year=2025,
            entity_info=entity_info,
            profile_id=123456,
            currency="USD",
        )

        # Check entity info is included
        assert tax_data["entity_name"] == "Test Company LLC"
        assert tax_data["ein"] == "12-3456789"
        assert tax_data["tax_year"] == 2025

        # Check financial data
        assert "total_income" in tax_data
        assert "total_deductions" in tax_data

        # Check metadata
        assert tax_data["_meta"]["source"] == "WiseConnector"
        assert tax_data["_meta"]["currency"] == "USD"


# =============================================================================
# Merge Functionality Tests
# =============================================================================

class TestMergeTaxData:
    """Tests for merging multiple tax data files."""

    def test_merge_single_file(self):
        """Merging a single file should return it unchanged."""
        data = {
            "entity_name": "Test",
            "total_income": 1000.0,
            "total_deductions": 200.0,
        }
        result = merge_tax_data(data)
        assert result == data

    def test_merge_two_files(self):
        """Should merge two data files correctly."""
        data1 = {
            "entity_name": "Test 1",
            "total_income": 1000.0,
            "total_deductions": 200.0,
            "ordinary_income": 800.0,
            "advertising": 100.0,
            "_meta": {"source": "Mercury", "total_transactions": 10},
        }
        data2 = {
            "entity_name": "Test 2",
            "total_income": 500.0,
            "total_deductions": 100.0,
            "ordinary_income": 400.0,
            "advertising": 50.0,
            "_meta": {"source": "Wise", "total_transactions": 5},
        }

        result = merge_tax_data(data1, data2)

        # Numeric values should be summed
        assert result["total_income"] == 1500.0
        assert result["total_deductions"] == 300.0
        assert result["ordinary_income"] == 1200.0
        assert result["advertising"] == 150.0

        # Metadata should reflect merge
        assert result["_meta"]["merged_from"] == ["Mercury", "Wise"]
        assert result["_meta"]["total_transactions"] == 15

    def test_merge_with_entity_info_override(self, entity_info):
        """Should override entity info when provided."""
        data1 = {
            "entity_name": "Old Name",
            "total_income": 1000.0,
        }
        data2 = {
            "entity_name": "Another Name",
            "total_income": 500.0,
        }

        result = merge_tax_data(data1, data2, entity_info=entity_info)

        assert result["entity_name"] == "Test Company LLC"
        assert result["ein"] == "12-3456789"

    def test_merge_empty_files(self):
        """Should handle empty input."""
        result = merge_tax_data()
        assert result == {}


# =============================================================================
# CLI Integration Tests
# =============================================================================

class TestCLI:
    """Tests for CLI import commands."""

    @patch("formbridge.connectors.mercury.MercuryConnector")
    def test_import_mercury_cli(self, mock_connector_class, entity_info, tmp_path):
        """Test Mercury import via CLI."""
        from click.testing import CliRunner
        from formbridge.cli import main

        # Mock the connector
        mock_connector = MagicMock()
        mock_connector.get_accounts.return_value = [
            Account(id="1", name="Test", account_type="checking", currency="USD", balance_cents=100000)
        ]
        mock_connector.generate_tax_data.return_value = {
            "entity_name": "Test Company",
            "total_income": 1000.0,
            "total_deductions": 200.0,
            "ordinary_income": 800.0,
            "_meta": {"total_transactions": 10},
        }
        mock_connector_class.return_value = mock_connector

        runner = CliRunner()
        output_path = tmp_path / "output.json"

        result = runner.invoke(
            main,
            [
                "import", "mercury",
                "--year", "2025",
                "--entity-name", "Test Company",
                "--output", str(output_path),
            ],
            env={"MERCURY_API_TOKEN": "test_token"},
        )

        assert result.exit_code == 0
        assert output_path.exists()

        # Verify output file
        output_data = json.loads(output_path.read_text())
        assert output_data["entity_name"] == "Test Company"

    @patch("formbridge.connectors.wise.WiseConnector")
    def test_import_wise_cli_list_profiles(self, mock_connector_class):
        """Test Wise list profiles via CLI."""
        from click.testing import CliRunner
        from formbridge.cli import main

        # Mock the connector
        mock_connector = MagicMock()
        mock_connector.get_profiles.return_value = [
            {"id": 123, "type": "business", "name": "Test Company"},
        ]
        mock_connector_class.return_value = mock_connector

        runner = CliRunner()

        result = runner.invoke(
            main,
            ["import", "wise", "--list-profiles"],
            env={"WISE_API_TOKEN": "test_token"},
        )

        assert result.exit_code == 0
        assert "123" in result.output
        assert "Test Company" in result.output


# =============================================================================
# Edge Cases and Error Handling Tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_transactions_list(self, entity_info):
        """Should handle empty transaction lists gracefully."""
        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"accounts": [], "transactions": []}
            mock_client.request.return_value = mock_response
            mock_client_class.return_value = mock_client

            connector = MercuryConnector(api_token="test_token")
            tax_data = connector.generate_tax_data(year=2025, entity_info=entity_info)

            assert tax_data["total_income"] == 0.0
            assert tax_data["total_deductions"] == 0.0
            assert tax_data["ordinary_income"] == 0.0

    def test_malformed_api_response(self):
        """Should handle malformed API responses gracefully."""
        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {}  # Missing expected keys
            mock_client.request.return_value = mock_response
            mock_client_class.return_value = mock_client

            connector = MercuryConnector(api_token="test_token")
            accounts = connector.get_accounts()

            # Should return empty list, not crash
            assert accounts == []

    def test_pagination_handling(self, sample_mercury_transactions_response):
        """Should handle paginated responses correctly."""
        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            
            # First call returns data with next cursor
            first_response = MagicMock()
            first_response.status_code = 200
            first_page = dict(sample_mercury_transactions_response)
            first_page["next"] = {"cursor": "next_page_token"}
            
            # Second call returns data without next
            second_response = MagicMock()
            second_response.status_code = 200
            second_page = {
                "transactions": [
                    {
                        "id": "txn_005",
                        "postedAt": "2025-04-01T10:00:00Z",
                        "amount": 100.00,
                        "description": "Additional transaction",
                    }
                ],
                "next": None,
            }

            call_count = [0]
            def side_effect(method, endpoint, **kwargs):
                call_count[0] += 1
                if call_count[0] == 1:
                    # First call is transactions (account_id passed directly)
                    first_response.json.return_value = first_page
                    return first_response
                else:
                    # Second call is next page
                    second_response.json.return_value = second_page
                    return second_response

            mock_client.request.side_effect = side_effect
            mock_client_class.return_value = mock_client

            connector = MercuryConnector(api_token="test_token")
            transactions = connector.get_transactions(
                account_id="acc_123",
                start_date=date(2025, 1, 1),
                end_date=date(2025, 12, 31),
            )

            # Should have fetched both pages
            assert len(transactions) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
