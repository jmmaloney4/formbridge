"""Wise API connector for FormBridge.

This module provides a connector for the Wise (formerly TransferWise) API,
allowing import of account data and transactions for tax form filling.

Wise API docs: https://docs.wise.com/api-docs/

Key endpoints:
- GET /v1/profiles - List profiles
- GET /v4/profiles/{profileId}/balances - Get balances
- GET /v3/profiles/{profileId}/borderless-accounts/{accountId}/statement.json - Get statement

Example usage:
    >>> from formbridge.connectors import WiseConnector, EntityInfo
    >>> 
    >>> connector = WiseConnector(api_token="...")
    >>> profiles = connector.get_profiles()
    >>> balances = connector.get_balances(profile_id)
    >>> transactions = connector.get_transactions(profile_id, start_date, end_date, "USD")
    >>> tax_data = connector.generate_tax_data(profile_id, 2025, "USD", entity_info)
"""

from __future__ import annotations

import logging
import os
from datetime import date, datetime
from typing import Any

import httpx

from . import (
    Account,
    AnnualSummary,
    APIError,
    AuthenticationError,
    BaseConnector,
    ConnectorError,
    EntityInfo,
    Transaction,
    TransactionCategory,
    TransactionType,
    categorize_transaction,
)

logger = logging.getLogger(__name__)

# Wise API base URL
WISE_API_BASE = "https://api.transferwise.com"


class WiseConnector(BaseConnector):
    """Connector for Wise (TransferWise) API.

    Provides access to Wise profiles, balances, and transactions for
    generating tax data for FormBridge.

    API tokens should be stored in the WISE_API_TOKEN environment variable.
    """

    def __init__(
        self,
        api_token: str | None = None,
        base_url: str | None = None,
        timeout: float = 30.0
    ):
        """Initialize the Wise connector.

        Args:
            api_token: Wise API token (reads from WISE_API_TOKEN env var if not provided)
            base_url: API base URL (defaults to https://api.transferwise.com)
            timeout: Request timeout in seconds
        """
        # Get token from environment if not provided
        token = api_token or os.environ.get("WISE_API_TOKEN")
        if not token:
            raise AuthenticationError(
                "Wise API token required. Set WISE_API_TOKEN environment variable "
                "or pass api_token parameter."
            )

        super().__init__(
            api_token=token,
            base_url=base_url or WISE_API_BASE,
            timeout=timeout
        )

        self._client = httpx.Client(
            base_url=self._base_url,
            timeout=self._timeout,
            headers={
                "Authorization": f"Bearer {self._api_token}",
                "Accept": "application/json",
            }
        )

        # Cache for profile data
        self._profiles_cache: list[dict[str, Any]] | None = None

    def __del__(self):
        """Clean up HTTP client."""
        if hasattr(self, "_client"):
            self._client.close()

    def _make_request(
        self,
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any] | list[Any]:
        """Make an API request.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (e.g., "/v1/profiles")
            params: Query parameters

        Returns:
            Response JSON

        Raises:
            APIError: On API failure
            AuthenticationError: On authentication failure
        """
        try:
            response = self._client.request(method, endpoint, params=params)

            if response.status_code == 401:
                raise AuthenticationError(
                    "Wise API authentication failed. Check your API token."
                )

            if response.status_code == 403:
                raise AuthenticationError(
                    "Wise API access forbidden. Check your API token permissions."
                )

            if response.status_code >= 400:
                raise APIError(
                    f"Wise API error: {response.status_code}",
                    status_code=response.status_code,
                    response_body=response.text
                )

            return response.json()

        except httpx.TimeoutException as e:
            raise APIError(f"Wise API timeout: {e}") from e
        except httpx.RequestError as e:
            raise APIError(f"Wise API request error: {e}") from e

    def get_profiles(self) -> list[dict[str, Any]]:
        """Get all Wise profiles (personal and business).

        Wise separates accounts into profiles. You'll need a profile ID
        to fetch balances and transactions.

        Returns:
            List of profile dicts with keys: id, type, name, etc.

        Raises:
            APIError: On API failure
            AuthenticationError: On auth failure
        """
        if self._profiles_cache is not None:
            return self._profiles_cache

        profiles = self._make_request("GET", "/v1/profiles")
        if isinstance(profiles, list):
            self._profiles_cache = profiles
            return profiles
        return []

    def get_profile(self, profile_id: int | str) -> dict[str, Any] | None:
        """Get a specific profile by ID.

        Args:
            profile_id: Wise profile ID

        Returns:
            Profile dict or None if not found
        """
        profiles = self.get_profiles()
        for profile in profiles:
            if str(profile.get("id")) == str(profile_id):
                return profile
        return None

    def get_balances(self, profile_id: int | str) -> list[Account]:
        """Get all currency balances for a profile.

        Args:
            profile_id: Wise profile ID

        Returns:
            List of Account objects (one per currency)

        Raises:
            APIError: On API failure
            AuthenticationError: On auth failure
        """
        # Wise uses /v4/profiles/{profileId}/balances?types=STANDARD
        response = self._make_request(
            "GET",
            f"/v4/profiles/{profile_id}/balances",
            params={"types": "STANDARD"}
        )

        if not isinstance(response, list):
            return []

        accounts = []
        for balance in response:
            # Each balance has a currency and amount
            # Wise returns amounts in the smallest currency unit (cents for USD)
            amount = balance.get("amount", {})
            value = amount.get("value", 0)
            currency = amount.get("currency", "USD")

            # Wise returns amounts as decimals, convert to cents
            # Note: Some currencies have different decimal places
            # For simplicity, we assume 2 decimal places (most common)
            amount_cents = int(value * 100) if value else 0

            account = Account(
                id=str(balance.get("id", "")),
                name=f"Wise {currency}",
                account_type="multi_currency",
                currency=currency,
                balance_cents=amount_cents,
                available_balance_cents=amount_cents,
                institution_name="Wise",
                raw_data=balance,
            )
            accounts.append(account)

        return accounts

    def get_borderless_account_id(self, profile_id: int | str) -> str | None:
        """Get the borderless account ID for a profile.

        Wise stores transactions under a borderless account,
        which is separate from the profile.

        Args:
            profile_id: Wise profile ID

        Returns:
            Borderless account ID or None
        """
        try:
            response = self._make_request(
                "GET",
                f"/v1/borderless-accounts",
                params={"profileId": str(profile_id)}
            )

            if isinstance(response, list) and response:
                return str(response[0].get("id", ""))
            return None
        except APIError:
            return None

    def get_transactions(
        self,
        profile_id: str | int | None,
        start_date: date,
        end_date: date,
        currency: str = "USD"
    ) -> list[Transaction]:
        """Get transactions from a Wise profile.

        Args:
            profile_id: Wise profile ID (required)
            start_date: Start of date range (inclusive)
            end_date: End of date range (inclusive)
            currency: Currency code (e.g., "USD")

        Returns:
            List of Transaction objects

        Raises:
            APIError: On API failure
            AuthenticationError: On auth failure
            ConnectorError: If profile_id not provided
        """
        if profile_id is None:
            raise ConnectorError("profile_id is required for Wise transactions")

        # Get the borderless account ID
        account_id = self.get_borderless_account_id(profile_id)
        if not account_id:
            logger.warning(f"No borderless account found for profile {profile_id}")
            return []

        # Format dates for Wise API (ISO 8601 with time)
        interval_start = f"{start_date.isoformat()}T00:00:00Z"
        interval_end = f"{end_date.isoformat()}T23:59:59Z"

        # Get statement from Wise
        response = self._make_request(
            "GET",
            f"/v3/profiles/{profile_id}/borderless-accounts/{account_id}/statement.json",
            params={
                "currency": currency,
                "intervalStart": interval_start,
                "intervalEnd": interval_end,
            }
        )

        if not isinstance(response, dict):
            return []

        transactions_data = response.get("transactions", [])
        transactions = []

        for txn in transactions_data:
            # Parse date
            date_str = txn.get("date", "")
            if date_str:
                try:
                    txn_date = datetime.fromisoformat(date_str.replace("Z", "+00:00")).date()
                except (ValueError, TypeError):
                    txn_date = start_date
            else:
                txn_date = start_date

            # Get amount and type
            amount_details = txn.get("amount", {})
            value = amount_details.get("value", 0)

            # Wise returns amount as decimal with sign
            # Negative = debit (expense), Positive = credit (income)
            amount_cents = int(value * 100) if value else 0

            if amount_cents > 0:
                txn_type = TransactionType.INCOME
            else:
                txn_type = TransactionType.EXPENSE

            # Get description
            details = txn.get("details", {})
            description = (
                details.get("description") or
                details.get("paymentReference") or
                details.get("merchantName") or
                txn.get("reference", "")
            )

            # Categorize
            category = categorize_transaction(
                description=description,
                amount_cents=amount_cents,
            )

            transaction = Transaction(
                id=txn.get("reference", ""),
                date=txn_date,
                amount_cents=amount_cents,
                description=description,
                category=category,
                transaction_type=txn_type,
                merchant_name=details.get("merchantName"),
                reference=txn.get("reference"),
                raw_data=txn,
            )
            transactions.append(transaction)

        # Sort by date
        transactions.sort(key=lambda t: t.date)

        return transactions

    def get_accounts(self) -> list[Account]:
        """Get all accounts across all profiles.

        This implements the DataConnector protocol by aggregating
        balances from all profiles.

        Returns:
            List of Account objects
        """
        profiles = self.get_profiles()
        all_accounts = []

        for profile in profiles:
            profile_id = profile.get("id")
            if profile_id:
                balances = self.get_balances(profile_id)
                all_accounts.extend(balances)

        return all_accounts

    def get_annual_summary(
        self,
        year: int,
        profile_id: int | str | None = None,
        currency: str = "USD"
    ) -> AnnualSummary:
        """Get annual financial summary for a specific profile/currency.

        Args:
            year: Tax year
            profile_id: Wise profile ID (uses first profile if not provided)
            currency: Currency code

        Returns:
            AnnualSummary with totals and breakdowns
        """
        from datetime import date as date_type

        start_date = date_type(year, 1, 1)
        end_date = date_type(year, 12, 31)

        # Use first profile if not specified
        if profile_id is None:
            profiles = self.get_profiles()
            if not profiles:
                return AnnualSummary(
                    year=year,
                    total_income_cents=0,
                    total_expenses_cents=0,
                    net_income_cents=0,
                    by_category={},
                    transaction_count=0,
                    income_transaction_count=0,
                    expense_transaction_count=0,
                )
            profile_id = profiles[0].get("id")

        transactions = self.get_transactions(
            str(profile_id),
            start_date,
            end_date,
            currency
        )

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
        profile_id: int | str | None = None,
        currency: str = "USD",
    ) -> dict[str, Any]:
        """Generate FormBridge-compatible JSON for tax filing.

        Args:
            year: Tax year
            entity_info: Entity information
            profile_id: Wise profile ID (uses first profile if not provided)
            currency: Currency code

        Returns:
            JSON dict compatible with formbridge fill --data
        """
        # Use first profile if not specified
        if profile_id is None:
            profiles = self.get_profiles()
            if profiles:
                profile_id = profiles[0].get("id")

        # Use parent class implementation with custom summary
        from datetime import date as date_type

        start_date = date_type(year, 1, 1)
        end_date = date_type(year, 12, 31)

        transactions = self.get_transactions(
            str(profile_id) if profile_id else None,
            start_date,
            end_date,
            currency
        )

        # Build summary
        total_income = 0
        total_expenses = 0
        by_category: dict[TransactionCategory, int] = {}

        for txn in transactions:
            if txn.amount_cents > 0:
                total_income += txn.amount_cents
            else:
                total_expenses += abs(txn.amount_cents)

            if txn.category not in by_category:
                by_category[txn.category] = 0
            by_category[txn.category] += abs(txn.amount_cents)

        summary = AnnualSummary(
            year=year,
            total_income_cents=total_income,
            total_expenses_cents=total_expenses,
            net_income_cents=total_income - total_expenses,
            by_category=by_category,
            transaction_count=len(transactions),
            income_transaction_count=sum(1 for t in transactions if t.amount_cents > 0),
            expense_transaction_count=sum(1 for t in transactions if t.amount_cents <= 0),
        )

        # Helper to convert cents to dollars
        def to_dollars(cents: int) -> float:
            return cents / 100.0

        # Build tax data
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
                by_category.get(TransactionCategory.GROSS_RECEIPTS, 0)
            ),
            "total_income": to_dollars(summary.total_income_cents),
            "total_deductions": to_dollars(summary.total_expenses_cents),
            "ordinary_income": to_dollars(summary.net_income_cents),

            # Deduction categories
            "cost_of_goods_sold": to_dollars(
                by_category.get(TransactionCategory.COST_OF_GOODS_SOLD, 0)
            ),
            "advertising": to_dollars(
                by_category.get(TransactionCategory.ADVERTISING, 0)
            ),
            "car_truck_expenses": to_dollars(
                by_category.get(TransactionCategory.CAR_TRUCK_EXPENSES, 0)
            ),
            "commissions_fees": to_dollars(
                by_category.get(TransactionCategory.COMMISSIONS_FEES, 0)
            ),
            "contract_labor": to_dollars(
                by_category.get(TransactionCategory.CONTRACT_LABOR, 0)
            ),
            "depletion": to_dollars(
                by_category.get(TransactionCategory.DEPLETION, 0)
            ),
            "employee_benefit_programs": to_dollars(
                by_category.get(TransactionCategory.EMPLOYEE_BENEFIT_PROGRAMS, 0)
            ),
            "insurance": to_dollars(
                by_category.get(TransactionCategory.INSURANCE, 0)
            ),
            "interest_paid": to_dollars(
                by_category.get(TransactionCategory.INTEREST_PAID, 0)
            ),
            "legal_professional_services": to_dollars(
                by_category.get(TransactionCategory.LEGAL_PROFESSIONAL_SERVICES, 0)
            ),
            "office_expense": to_dollars(
                by_category.get(TransactionCategory.OFFICE_EXPENSE, 0)
            ),
            "pension_profit_sharing": to_dollars(
                by_category.get(TransactionCategory.PENSION_PROFIT_SHARING, 0)
            ),
            "rent_lease": to_dollars(
                by_category.get(TransactionCategory.RENT_LEASE, 0)
            ),
            "repairs_maintenance": to_dollars(
                by_category.get(TransactionCategory.REPAIRS_MAINTENANCE, 0)
            ),
            "supplies": to_dollars(
                by_category.get(TransactionCategory.SUPPLIES, 0)
            ),
            "taxes_licenses": to_dollars(
                by_category.get(TransactionCategory.TAXES_LICENSES, 0)
            ),
            "travel": to_dollars(
                by_category.get(TransactionCategory.TRAVEL, 0)
            ),
            "meals": to_dollars(
                by_category.get(TransactionCategory.MEALS, 0)
            ),
            "utilities": to_dollars(
                by_category.get(TransactionCategory.UTILITIES, 0)
            ),
            "wages": to_dollars(
                by_category.get(TransactionCategory.WAGES, 0)
            ),
            "other_deductions": to_dollars(
                by_category.get(TransactionCategory.OTHER_DEDUCTIONS, 0)
            ),

            # Income categories
            "interest_income": to_dollars(
                by_category.get(TransactionCategory.INTEREST_INCOME, 0)
            ),
            "dividend_income": to_dollars(
                by_category.get(TransactionCategory.DIVIDEND_INCOME, 0)
            ),
            "other_income": to_dollars(
                by_category.get(TransactionCategory.OTHER_INCOME, 0)
            ),

            # Metadata
            "_meta": {
                "source": "WiseConnector",
                "year": year,
                "profile_id": str(profile_id) if profile_id else None,
                "currency": currency,
                "total_transactions": summary.transaction_count,
                "income_transactions": summary.income_transaction_count,
                "expense_transactions": summary.expense_transaction_count,
            }
        }

        return data


# Convenience functions

def get_wise_profiles(api_token: str | None = None) -> list[dict[str, Any]]:
    """Get all Wise profiles.

    Args:
        api_token: Wise API token (reads from env if not provided)

    Returns:
        List of profile dicts
    """
    connector = WiseConnector(api_token=api_token)
    return connector.get_profiles()


def get_wise_balances(
    profile_id: int | str,
    api_token: str | None = None,
) -> list[Account]:
    """Get Wise balances for a profile.

    Args:
        profile_id: Wise profile ID
        api_token: Wise API token

    Returns:
        List of Account objects
    """
    connector = WiseConnector(api_token=api_token)
    return connector.get_balances(profile_id)


def get_wise_transactions(
    profile_id: int | str,
    start_date: date,
    end_date: date,
    currency: str = "USD",
    api_token: str | None = None,
) -> list[Transaction]:
    """Get Wise transactions.

    Args:
        profile_id: Wise profile ID
        start_date: Start date
        end_date: End date
        currency: Currency code
        api_token: Wise API token

    Returns:
        List of Transaction objects
    """
    connector = WiseConnector(api_token=api_token)
    return connector.get_transactions(profile_id, start_date, end_date, currency)


def generate_wise_tax_data(
    year: int,
    entity_info: EntityInfo,
    profile_id: int | str | None = None,
    currency: str = "USD",
    api_token: str | None = None,
) -> dict[str, Any]:
    """Generate tax data from Wise.

    Args:
        year: Tax year
        entity_info: Entity information
        profile_id: Wise profile ID (uses first if not provided)
        currency: Currency code
        api_token: Wise API token

    Returns:
        Tax data JSON for FormBridge
    """
    connector = WiseConnector(api_token=api_token)
    return connector.generate_tax_data(year, entity_info, profile_id, currency)


__all__ = [
    "WiseConnector",
    "get_wise_profiles",
    "get_wise_balances",
    "get_wise_transactions",
    "generate_wise_tax_data",
]
