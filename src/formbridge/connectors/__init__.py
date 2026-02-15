"""Data connectors for FormBridge.

This module provides connectors for importing financial data from various
sources (Mercury, Wise, etc.) and generating FormBridge-compatible JSON
for tax form filling.

Example usage:
    >>> from formbridge.connectors import MercuryConnector, WiseConnector
    >>> 
    >>> # Mercury connector
    >>> mercury = MercuryConnector(api_token="...")
    >>> accounts = mercury.get_accounts()
    >>> tax_data = mercury.generate_tax_data(2025, entity_info)
    >>> 
    >>> # Wise connector
    >>> wise = WiseConnector(api_token="...")
    >>> profiles = wise.get_profiles()
    >>> tax_data = wise.generate_tax_data(profile_id, 2025, "USD", entity_info)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Any, Protocol

from pydantic import BaseModel, Field


class ConnectorError(Exception):
    """Base error for connector operations."""
    pass


class AuthenticationError(ConnectorError):
    """Error for authentication failures."""
    pass


class APIError(ConnectorError):
    """Error for API failures."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        response_body: str | None = None
    ):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class TransactionType(str, Enum):
    """Type of financial transaction."""
    INCOME = "income"
    EXPENSE = "expense"
    TRANSFER = "transfer"


class TransactionCategory(str, Enum):
    """Category for tax reporting."""
    GROSS_RECEIPTS = "gross_receipts"
    COST_OF_GOODS_SOLD = "cost_of_goods_sold"
    ADVERTISING = "advertising"
    CAR_TRUCK_EXPENSES = "car_truck_expenses"
    COMMISSIONS_FEES = "commissions_fees"
    CONTRACT_LABOR = "contract_labor"
    DEPLETION = "depletion"
    EMPLOYEE_BENEFIT_PROGRAMS = "employee_benefit_programs"
    INSURANCE = "insurance"
    INTEREST_PAID = "interest_paid"
    LEGAL_PROFESSIONAL_SERVICES = "legal_professional_services"
    OFFICE_EXPENSE = "office_expense"
    PENSION_PROFIT_SHARING = "pension_profit_sharing"
    RENT_LEASE = "rent_lease"
    REPAIRS_MAINTENANCE = "repairs_maintenance"
    SUPPLIES = "supplies"
    TAXES_LICENSES = "taxes_licenses"
    TRAVEL = "travel"
    MEALS = "meals"
    UTILITIES = "utilities"
    WAGES = "wages"
    OTHER_DEDUCTIONS = "other_deductions"
    INTEREST_INCOME = "interest_income"
    DIVIDEND_INCOME = "dividend_income"
    OTHER_INCOME = "other_income"
    UNCATEGORIZED = "uncategorized"


@dataclass
class Transaction:
    """A financial transaction."""
    id: str
    date: date
    amount_cents: int  # Positive for income, negative for expense
    description: str
    category: TransactionCategory
    transaction_type: TransactionType
    merchant_name: str | None = None
    reference: str | None = None
    raw_data: dict[str, Any] | None = None


@dataclass
class Account:
    """A financial account."""
    id: str
    name: str
    account_type: str
    currency: str
    balance_cents: int
    available_balance_cents: int | None = None
    institution_name: str | None = None
    raw_data: dict[str, Any] | None = None


@dataclass
class AnnualSummary:
    """Annual financial summary for tax purposes."""
    year: int
    total_income_cents: int
    total_expenses_cents: int
    net_income_cents: int
    by_category: dict[TransactionCategory, int]  # amounts in cents
    transaction_count: int
    income_transaction_count: int
    expense_transaction_count: int


@dataclass
class EntityInfo:
    """Entity information for tax forms."""
    name: str
    ein: str | None = None
    address: str | None = None
    city: str | None = None
    state: str | None = None
    zip_code: str | None = None
    country: str = "US"
    entity_type: str = "partnership"  # "corporation", "partnership", "sole_proprietorship"
    tax_year: int | None = None
    accounting_method: str = "cash"  # "cash" or "accrual"
    business_activity: str | None = None
    business_code: str | None = None
    date_business_started: str | None = None  # ISO date format


class DataConnector(Protocol):
    """Protocol for data connectors.

    All connectors (Mercury, Wise, etc.) implement this interface
    to ensure consistent behavior.
    """

    def get_accounts(self) -> list[Account]:
        """Get all accounts with current balances."""
        ...

    def get_transactions(
        self,
        account_id: str | None,
        start_date: date,
        end_date: date
    ) -> list[Transaction]:
        """Get transactions in date range.

        Args:
            account_id: Specific account ID or None for all accounts
            start_date: Start of date range (inclusive)
            end_date: End of date range (inclusive)

        Returns:
            List of transactions in the date range
        """
        ...

    def get_annual_summary(
        self,
        year: int,
        account_id: str | None = None
    ) -> AnnualSummary:
        """Get annual financial summary.

        Args:
            year: Tax year
            account_id: Specific account ID or None for all accounts

        Returns:
            Annual summary with totals by category
        """
        ...

    def generate_tax_data(
        self,
        year: int,
        entity_info: EntityInfo,
        account_id: str | None = None
    ) -> dict[str, Any]:
        """Generate FormBridge-compatible JSON for tax filing.

        Args:
            year: Tax year
            entity_info: Entity information (name, EIN, address, etc.)
            account_id: Specific account ID or None for all accounts

        Returns:
            JSON dict compatible with formbridge fill --data
        """
        ...


class BaseConnector(ABC):
    """Base class for data connectors with common functionality."""

    def __init__(
        self,
        api_token: str | None = None,
        base_url: str | None = None,
        timeout: float = 30.0
    ):
        """Initialize the connector.

        Args:
            api_token: API token for authentication
            base_url: Base URL for the API (optional, uses default)
            timeout: Request timeout in seconds
        """
        self._api_token = api_token
        self._base_url = base_url
        self._timeout = timeout

    @abstractmethod
    def get_accounts(self) -> list[Account]:
        """Get all accounts with current balances."""
        pass

    @abstractmethod
    def get_transactions(
        self,
        account_id: str | None,
        start_date: date,
        end_date: date
    ) -> list[Transaction]:
        """Get transactions in date range."""
        pass

    def get_annual_summary(
        self,
        year: int,
        account_id: str | None = None
    ) -> AnnualSummary:
        """Get annual financial summary for tax purposes.

        Args:
            year: Tax year
            account_id: Specific account ID or None for all accounts

        Returns:
            AnnualSummary with totals and breakdowns
        """
        from datetime import date as date_type

        start_date = date_type(year, 1, 1)
        end_date = date_type(year, 12, 31)

        transactions = self.get_transactions(account_id, start_date, end_date)

        total_income = 0
        total_expenses = 0
        by_category: dict[TransactionCategory, int] = {}
        income_count = 0
        expense_count = 0

        for txn in transactions:
            if txn.amount_cents > 0:
                total_income += txn.amount_cents
                income_count += 1
            else:
                total_expenses += abs(txn.amount_cents)
                expense_count += 1

            # Track by category
            if txn.category not in by_category:
                by_category[txn.category] = 0
            by_category[txn.category] += abs(txn.amount_cents)

        return AnnualSummary(
            year=year,
            total_income_cents=total_income,
            total_expenses_cents=total_expenses,
            net_income_cents=total_income - total_expenses,
            by_category=by_category,
            transaction_count=len(transactions),
            income_transaction_count=income_count,
            expense_transaction_count=expense_count,
        )

    def generate_tax_data(
        self,
        year: int,
        entity_info: EntityInfo,
        account_id: str | None = None
    ) -> dict[str, Any]:
        """Generate FormBridge-compatible JSON for tax filing.

        This generates a data JSON compatible with formbridge fill --data,
        containing entity information and financial data organized for
        Form 1120 (C-corp) or Form 1065 (partnership).

        Args:
            year: Tax year
            entity_info: Entity information
            account_id: Specific account ID or None for all accounts

        Returns:
            JSON dict ready for formbridge fill
        """
        summary = self.get_annual_summary(year, account_id)

        # Helper to convert cents to dollars
        def to_dollars(cents: int) -> float:
            return cents / 100.0

        # Base entity info
        data: dict[str, Any] = {
            # Entity information
            "entity_name": entity_info.name,
            "ein": entity_info.ein,
            "address": entity_info.address,
            "city": entity_info.city,
            "state": entity_info.state,
            "zip_code": entity_info.zip_code,
            "country": entity_info.country,
            "entity_type": entity_info.entity_type,
            "tax_year": year,
            "accounting_method": entity_info.accounting_method,
            "business_activity": entity_info.business_activity,
            "business_code": entity_info.business_code,
            "date_business_started": entity_info.date_business_started,

            # Financial summary
            "gross_receipts": to_dollars(
                summary.by_category.get(TransactionCategory.GROSS_RECEIPTS, 0)
            ),
            "total_income": to_dollars(summary.total_income_cents),
            "total_deductions": to_dollars(summary.total_expenses_cents),
            "ordinary_income": to_dollars(summary.net_income_cents),

            # Deduction categories (Form 1120 / 1065 line items)
            "cost_of_goods_sold": to_dollars(
                summary.by_category.get(TransactionCategory.COST_OF_GOODS_SOLD, 0)
            ),
            "advertising": to_dollars(
                summary.by_category.get(TransactionCategory.ADVERTISING, 0)
            ),
            "car_truck_expenses": to_dollars(
                summary.by_category.get(TransactionCategory.CAR_TRUCK_EXPENSES, 0)
            ),
            "commissions_fees": to_dollars(
                summary.by_category.get(TransactionCategory.COMMISSIONS_FEES, 0)
            ),
            "contract_labor": to_dollars(
                summary.by_category.get(TransactionCategory.CONTRACT_LABOR, 0)
            ),
            "depletion": to_dollars(
                summary.by_category.get(TransactionCategory.DEPLETION, 0)
            ),
            "employee_benefit_programs": to_dollars(
                summary.by_category.get(TransactionCategory.EMPLOYEE_BENEFIT_PROGRAMS, 0)
            ),
            "insurance": to_dollars(
                summary.by_category.get(TransactionCategory.INSURANCE, 0)
            ),
            "interest_paid": to_dollars(
                summary.by_category.get(TransactionCategory.INTEREST_PAID, 0)
            ),
            "legal_professional_services": to_dollars(
                summary.by_category.get(TransactionCategory.LEGAL_PROFESSIONAL_SERVICES, 0)
            ),
            "office_expense": to_dollars(
                summary.by_category.get(TransactionCategory.OFFICE_EXPENSE, 0)
            ),
            "pension_profit_sharing": to_dollars(
                summary.by_category.get(TransactionCategory.PENSION_PROFIT_SHARING, 0)
            ),
            "rent_lease": to_dollars(
                summary.by_category.get(TransactionCategory.RENT_LEASE, 0)
            ),
            "repairs_maintenance": to_dollars(
                summary.by_category.get(TransactionCategory.REPAIRS_MAINTENANCE, 0)
            ),
            "supplies": to_dollars(
                summary.by_category.get(TransactionCategory.SUPPLIES, 0)
            ),
            "taxes_licenses": to_dollars(
                summary.by_category.get(TransactionCategory.TAXES_LICENSES, 0)
            ),
            "travel": to_dollars(
                summary.by_category.get(TransactionCategory.TRAVEL, 0)
            ),
            "meals": to_dollars(
                summary.by_category.get(TransactionCategory.MEALS, 0)
            ),
            "utilities": to_dollars(
                summary.by_category.get(TransactionCategory.UTILITIES, 0)
            ),
            "wages": to_dollars(
                summary.by_category.get(TransactionCategory.WAGES, 0)
            ),
            "other_deductions": to_dollars(
                summary.by_category.get(TransactionCategory.OTHER_DEDUCTIONS, 0)
            ),

            # Income categories
            "interest_income": to_dollars(
                summary.by_category.get(TransactionCategory.INTEREST_INCOME, 0)
            ),
            "dividend_income": to_dollars(
                summary.by_category.get(TransactionCategory.DIVIDEND_INCOME, 0)
            ),
            "other_income": to_dollars(
                summary.by_category.get(TransactionCategory.OTHER_INCOME, 0)
            ),

            # Metadata
            "_meta": {
                "source": self.__class__.__name__,
                "year": year,
                "total_transactions": summary.transaction_count,
                "income_transactions": summary.income_transaction_count,
                "expense_transactions": summary.expense_transaction_count,
            }
        }

        return data


def categorize_transaction(
    description: str,
    amount_cents: int,
    transaction_type: str | None = None
) -> TransactionCategory:
    """Categorize a transaction based on description and type.

    This is a simple keyword-based categorizer. In production, this
    could be enhanced with ML or user-defined rules.

    Args:
        description: Transaction description
        amount_cents: Amount in cents (positive = income, negative = expense)
        transaction_type: Optional type hint from the API

    Returns:
        TransactionCategory for the transaction
    """
    desc_lower = description.lower()

    # Income patterns
    if amount_cents > 0:
        if any(kw in desc_lower for kw in ["interest", "int payment", "dividend"]):
            return TransactionCategory.INTEREST_INCOME
        if any(kw in desc_lower for kw in ["dividend"]):
            return TransactionCategory.DIVIDEND_INCOME
        # Default income to gross receipts
        return TransactionCategory.GROSS_RECEIPTS

    # Expense patterns
    if any(kw in desc_lower for kw in ["advertising", "marketing", "google ads", "facebook", "linkedin ads"]):
        return TransactionCategory.ADVERTISING

    if any(kw in desc_lower for kw in ["uber", "lyft", "gas station", "auto repair", "car wash"]):
        return TransactionCategory.CAR_TRUCK_EXPENSES

    if any(kw in desc_lower for kw in ["legal", "attorney", "lawyer", "counsel"]):
        return TransactionCategory.LEGAL_PROFESSIONAL_SERVICES

    if any(kw in desc_lower for kw in ["commission", "fee", "stripe", "paypal fee"]):
        return TransactionCategory.COMMISSIONS_FEES

    if any(kw in desc_lower for kw in ["contractor", "freelance", "upwork", "fiverr"]):
        return TransactionCategory.CONTRACT_LABOR

    if any(kw in desc_lower for kw in ["insurance", "geico", "state farm"]):
        return TransactionCategory.INSURANCE

    if any(kw in desc_lower for kw in ["interest payment", "loan payment", "mortgage"]):
        return TransactionCategory.INTEREST_PAID

    if any(kw in desc_lower for kw in ["office", "staples", "office depot", "paper", "printing"]):
        return TransactionCategory.OFFICE_EXPENSE

    if any(kw in desc_lower for kw in ["rent", "lease"]):
        return TransactionCategory.RENT_LEASE

    if any(kw in desc_lower for kw in ["repair", "maintenance", "hvac", "plumbing"]):
        return TransactionCategory.REPAIRS_MAINTENANCE

    if any(kw in desc_lower for kw in ["supplies", "amazon", "amazon web services", "aws"]):
        return TransactionCategory.SUPPLIES

    if any(kw in desc_lower for kw in ["tax", "license", "permit", "filing fee"]):
        return TransactionCategory.TAXES_LICENSES

    if any(kw in desc_lower for kw in ["airline", "hotel", "flight", "airbnb", "travel"]):
        return TransactionCategory.TRAVEL

    if any(kw in desc_lower for kw in ["restaurant", "cafe", "coffee", "doordash", "ubereats", "grubhub"]):
        return TransactionCategory.MEALS

    if any(kw in desc_lower for kw in ["electric", "water", "gas bill", "internet", "phone", "utility"]):
        return TransactionCategory.UTILITIES

    if any(kw in desc_lower for kw in ["payroll", "salary", "wage", "wages"]):
        return TransactionCategory.WAGES

    # Default expense to other deductions
    return TransactionCategory.OTHER_DEDUCTIONS


def merge_tax_data(
    *data_files: dict[str, Any],
    entity_info: EntityInfo | None = None
) -> dict[str, Any]:
    """Merge multiple tax data JSONs into one.

    This is useful when combining data from multiple sources
    (e.g., Mercury + Wise).

    Args:
        *data_files: Tax data JSONs to merge
        entity_info: Optional entity info to use (overrides any in data files)

    Returns:
        Merged tax data JSON
    """
    if not data_files:
        return {}

    if len(data_files) == 1:
        return data_files[0]

    # Start with the first file as base
    merged = dict(data_files[0])

    # Merge numeric values (add them)
    numeric_fields = [
        "gross_receipts", "total_income", "total_deductions", "ordinary_income",
        "cost_of_goods_sold", "advertising", "car_truck_expenses",
        "commissions_fees", "contract_labor", "depletion",
        "employee_benefit_programs", "insurance", "interest_paid",
        "legal_professional_services", "office_expense", "pension_profit_sharing",
        "rent_lease", "repairs_maintenance", "supplies", "taxes_licenses",
        "travel", "meals", "utilities", "wages", "other_deductions",
        "interest_income", "dividend_income", "other_income",
    ]

    for data in data_files[1:]:
        for field in numeric_fields:
            if field in data:
                merged[field] = merged.get(field, 0) + data[field]

    # Override entity info if provided
    if entity_info:
        merged.update({
            "entity_name": entity_info.name,
            "ein": entity_info.ein,
            "address": entity_info.address,
            "city": entity_info.city,
            "state": entity_info.state,
            "zip_code": entity_info.zip_code,
            "country": entity_info.country,
            "entity_type": entity_info.entity_type,
            "accounting_method": entity_info.accounting_method,
            "business_activity": entity_info.business_activity,
            "business_code": entity_info.business_code,
            "date_business_started": entity_info.date_business_started,
        })

    # Merge metadata
    meta_list = [d.get("_meta", {}) for d in data_files if d.get("_meta")]
    if meta_list:
        merged["_meta"] = {
            "merged_from": [m.get("source", "unknown") for m in meta_list],
            "total_transactions": sum(m.get("total_transactions", 0) for m in meta_list),
        }

    return merged


# Import connector classes (lazy to avoid circular imports)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from formbridge.connectors.mercury import MercuryConnector
    from formbridge.connectors.wise import WiseConnector


def __getattr__(name: str):
    """Lazy import of connector classes."""
    if name == "MercuryConnector":
        from formbridge.connectors.mercury import MercuryConnector
        return MercuryConnector
    if name == "WiseConnector":
        from formbridge.connectors.wise import WiseConnector
        return WiseConnector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Errors
    "ConnectorError",
    "AuthenticationError",
    "APIError",
    # Enums
    "TransactionType",
    "TransactionCategory",
    # Data classes
    "Transaction",
    "Account",
    "AnnualSummary",
    "EntityInfo",
    # Protocols and base classes
    "DataConnector",
    "BaseConnector",
    # Connectors (lazy loaded)
    "MercuryConnector",
    "WiseConnector",
    # Utilities
    "categorize_transaction",
    "merge_tax_data",
]
