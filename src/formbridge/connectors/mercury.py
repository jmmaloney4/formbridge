"""Mercury API connector for FormBridge.

This module provides a connector for the Mercury banking API,
allowing import of account data and transactions for tax form filling.

Mercury API docs: https://docs.mercury.com/reference/

Key endpoints:
- GET /api/v1/accounts - List accounts
- GET /api/v1/account/{id}/transactions - Get transactions

Example usage:
    >>> from formbridge.connectors import MercuryConnector, EntityInfo
    >>> 
    >>> connector = MercuryConnector(api_token="...")
    >>> accounts = connector.get_accounts()
    >>> transactions = connector.get_transactions(account_id, start_date, end_date)
    >>> tax_data = connector.generate_tax_data(2025, entity_info)
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

# Mercury API base URL
MERCURY_API_BASE = "https://api.mercury.com"


class MercuryConnector(BaseConnector):
    """Connector for Mercury banking API.

    Provides access to Mercury accounts and transactions for
    generating tax data for FormBridge.

    API tokens should be stored in the MERCURY_API_TOKEN environment variable.
    """

    def __init__(
        self,
        api_token: str | None = None,
        base_url: str | None = None,
        timeout: float = 30.0
    ):
        """Initialize the Mercury connector.

        Args:
            api_token: Mercury API token (reads from MERCURY_API_TOKEN env var if not provided)
            base_url: API base URL (defaults to https://api.mercury.com)
            timeout: Request timeout in seconds
        """
        # Get token from environment if not provided
        token = api_token or os.environ.get("MERCURY_API_TOKEN")
        if not token:
            raise AuthenticationError(
                "Mercury API token required. Set MERCURY_API_TOKEN environment variable "
                "or pass api_token parameter."
            )

        super().__init__(
            api_token=token,
            base_url=base_url or MERCURY_API_BASE,
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

    def __del__(self):
        """Clean up HTTP client."""
        if hasattr(self, "_client"):
            self._client.close()

    def _make_request(
        self,
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Make an API request.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (e.g., "/api/v1/accounts")
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
                    "Mercury API authentication failed. Check your API token."
                )

            if response.status_code == 403:
                raise AuthenticationError(
                    "Mercury API access forbidden. Check your API token permissions."
                )

            if response.status_code >= 400:
                raise APIError(
                    f"Mercury API error: {response.status_code}",
                    status_code=response.status_code,
                    response_body=response.text
                )

            return response.json()

        except httpx.TimeoutException as e:
            raise APIError(f"Mercury API timeout: {e}") from e
        except httpx.RequestError as e:
            raise APIError(f"Mercury API request error: {e}") from e

    def _paginated_request(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
        items_key: str = "transactions",
        max_pages: int = 100,
    ) -> list[dict[str, Any]]:
        """Make a paginated API request.

        Mercury uses cursor-based pagination.

        Args:
            endpoint: API endpoint
            params: Query parameters
            items_key: Key in response containing items
            max_pages: Maximum pages to fetch (safety limit)

        Returns:
            List of all items across all pages
        """
        all_items = []
        params = params or {}
        page = 0

        while page < max_pages:
            response = self._make_request("GET", endpoint, params=params)
            items = response.get(items_key, [])

            if not items:
                break

            all_items.extend(items)

            # Check for next page cursor
            # Mercury pagination varies by endpoint
            # Some use offset, some use cursor
            if "next" not in response or not response["next"]:
                break

            # For cursor-based pagination
            if "cursor" in response.get("next", {}):
                params["cursor"] = response["next"]["cursor"]
            # For offset-based pagination
            elif "offset" in response:
                params["offset"] = len(all_items)
            else:
                break

            page += 1

        return all_items

    def get_accounts(self) -> list[Account]:
        """Get all Mercury accounts with balances.

        Returns:
            List of Account objects

        Raises:
            APIError: On API failure
            AuthenticationError: On auth failure
        """
        response = self._make_request("GET", "/api/v1/accounts")
        accounts_data = response.get("accounts", [])

        accounts = []
        for acc in accounts_data:
            # Mercury returns amounts in dollars as floats
            # Convert to cents for internal storage
            balance = acc.get("currentBalance", 0)
            available = acc.get("availableBalance", balance)

            account = Account(
                id=acc.get("id", ""),
                name=acc.get("name", "Unknown"),
                account_type=acc.get("type", "checking"),
                currency=acc.get("currency", "USD"),
                balance_cents=int(balance * 100) if balance else 0,
                available_balance_cents=int(available * 100) if available else None,
                institution_name="Mercury",
                raw_data=acc,
            )
            accounts.append(account)

        return accounts

    def get_transactions(
        self,
        account_id: str | None,
        start_date: date,
        end_date: date
    ) -> list[Transaction]:
        """Get transactions from Mercury accounts.

        Args:
            account_id: Specific account ID or None for all accounts
            start_date: Start of date range (inclusive)
            end_date: End of date range (inclusive)

        Returns:
            List of Transaction objects

        Raises:
            APIError: On API failure
            AuthenticationError: On auth failure
        """
        if account_id:
            account_ids = [account_id]
        else:
            # Get all accounts
            accounts = self.get_accounts()
            account_ids = [acc.id for acc in accounts]

        all_transactions = []

        for acc_id in account_ids:
            # Mercury API uses YYYY-MM-DD date format
            params = {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
            }

            transactions_data = self._paginated_request(
                f"/api/v1/account/{acc_id}/transactions",
                params=params,
                items_key="transactions",
            )

            for txn in transactions_data:
                # Parse date from ISO format
                txn_date = datetime.fromisoformat(
                    txn.get("postedAt", txn.get("createdAt", ""))[:10]
                ).date()

                # Mercury returns amounts in dollars as floats
                # Negative for debits (expenses), positive for credits (income)
                amount = txn.get("amount", 0)
                amount_cents = int(amount * 100)

                # Determine transaction type based on amount direction
                if amount_cents > 0:
                    txn_type = TransactionType.INCOME
                else:
                    txn_type = TransactionType.EXPENSE

                # Categorize transaction
                description = txn.get("description", "")
                category = categorize_transaction(
                    description=description,
                    amount_cents=amount_cents,
                )

                transaction = Transaction(
                    id=txn.get("id", ""),
                    date=txn_date,
                    amount_cents=amount_cents,
                    description=description,
                    category=category,
                    transaction_type=txn_type,
                    merchant_name=txn.get("counterpartyName"),
                    reference=txn.get("reference"),
                    raw_data=txn,
                )
                all_transactions.append(transaction)

        # Sort by date
        all_transactions.sort(key=lambda t: t.date)

        return all_transactions

    def get_account_by_id(self, account_id: str) -> Account | None:
        """Get a specific account by ID.

        Args:
            account_id: Mercury account ID

        Returns:
            Account object or None if not found
        """
        try:
            response = self._make_request("GET", f"/api/v1/account/{account_id}")
            acc = response.get("account", response)

            if not acc:
                return None

            balance = acc.get("currentBalance", 0)
            available = acc.get("availableBalance", balance)

            return Account(
                id=acc.get("id", ""),
                name=acc.get("name", "Unknown"),
                account_type=acc.get("type", "checking"),
                currency=acc.get("currency", "USD"),
                balance_cents=int(balance * 100) if balance else 0,
                available_balance_cents=int(available * 100) if available else None,
                institution_name="Mercury",
                raw_data=acc,
            )
        except APIError:
            return None


# Convenience functions

def get_mercury_accounts(api_token: str | None = None) -> list[Account]:
    """Get all Mercury accounts.

    Args:
        api_token: Mercury API token (reads from env if not provided)

    Returns:
        List of Account objects
    """
    connector = MercuryConnector(api_token=api_token)
    return connector.get_accounts()


def get_mercury_transactions(
    account_id: str | None,
    start_date: date,
    end_date: date,
    api_token: str | None = None,
) -> list[Transaction]:
    """Get Mercury transactions.

    Args:
        account_id: Specific account or None for all
        start_date: Start date
        end_date: End date
        api_token: Mercury API token

    Returns:
        List of Transaction objects
    """
    connector = MercuryConnector(api_token=api_token)
    return connector.get_transactions(account_id, start_date, end_date)


def generate_mercury_tax_data(
    year: int,
    entity_info: EntityInfo,
    account_id: str | None = None,
    api_token: str | None = None,
) -> dict[str, Any]:
    """Generate tax data from Mercury.

    Args:
        year: Tax year
        entity_info: Entity information
        account_id: Specific account or None for all
        api_token: Mercury API token

    Returns:
        Tax data JSON for FormBridge
    """
    connector = MercuryConnector(api_token=api_token)
    return connector.generate_tax_data(year, entity_info, account_id)


__all__ = [
    "MercuryConnector",
    "get_mercury_accounts",
    "get_mercury_transactions",
    "generate_mercury_tax_data",
]
