# FormBridge Data Connectors

This document describes the data connectors added to FormBridge for importing
financial data from external sources and generating tax-ready JSON.

## Files Created/Modified

### New Files

1. **`src/formbridge/connectors/__init__.py`**
   - Base connector protocol and abstract class
   - Data models: `Transaction`, `Account`, `AnnualSummary`, `EntityInfo`
   - Enums: `TransactionType`, `TransactionCategory`
   - Exceptions: `ConnectorError`, `AuthenticationError`, `APIError`
   - Utilities: `categorize_transaction()`, `merge_tax_data()`

2. **`src/formbridge/connectors/mercury.py`**
   - `MercuryConnector` class for Mercury banking API
   - Methods: `get_accounts()`, `get_transactions()`, `get_annual_summary()`, `generate_tax_data()`
   - Convenience functions: `get_mercury_accounts()`, `get_mercury_transactions()`, `generate_mercury_tax_data()`

3. **`src/formbridge/connectors/wise.py`**
   - `WiseConnector` class for Wise (TransferWise) API
   - Methods: `get_profiles()`, `get_balances()`, `get_transactions()`, `get_annual_summary()`, `generate_tax_data()`
   - Convenience functions: `get_wise_profiles()`, `get_wise_balances()`, `get_wise_transactions()`, `generate_wise_tax_data()`

4. **`tests/test_connectors.py`**
   - Comprehensive tests for all connectors
   - Mocked HTTP calls (never hits real APIs)
   - Tests for: categorization, aggregation, tax data generation, merge functionality, CLI commands

### Modified Files

1. **`src/formbridge/cli.py`**
   - Added `import` command group with subcommands:
     - `formbridge import mercury` - Import from Mercury
     - `formbridge import wise` - Import from Wise
     - `formbridge import merge` - Merge multiple JSON files

## CLI Usage

### Mercury Import

```bash
# Set API token
export MERCURY_API_TOKEN='your-api-token'

# Import data for a tax year
formbridge import mercury --year 2025 \
    --entity-name "My Company LLC" \
    --ein "12-3456789" \
    --address "123 Main St" \
    --city "New York" \
    --state "NY" \
    --zip "10001" \
    --output mycompany-2025.json
```

### Wise Import

```bash
# Set API token
export WISE_API_TOKEN='your-api-token'

# List available profiles
formbridge import wise --list-profiles

# Import data for a specific profile
formbridge import wise --profile 123456 --year 2025 --currency USD \
    --entity-name "My Company LLC" \
    --output wise-2025.json
```

### Merge Multiple Sources

```bash
# Combine Mercury and Wise data
formbridge import merge mercury-2025.json wise-2025.json \
    --output combined-2025.json
```

## Generated JSON Structure

The generated JSON is compatible with `formbridge fill --data`:

```json
{
  "entity_name": "My Company LLC",
  "ein": "12-3456789",
  "address": "123 Main St",
  "city": "New York",
  "state": "NY",
  "zip_code": "10001",
  "entity_type": "partnership",
  "tax_year": 2025,
  "accounting_method": "cash",
  "business_activity": "Software consulting",
  "business_code": "541511",

  "gross_receipts": 50000.00,
  "total_income": 55000.00,
  "total_deductions": 15000.00,
  "ordinary_income": 40000.00,

  "advertising": 5000.00,
  "legal_professional_services": 2000.00,
  "office_expense": 1000.00,
  "supplies": 3000.00,
  "travel": 1500.00,
  "meals": 500.00,
  "other_deductions": 2000.00,

  "_meta": {
    "source": "MercuryConnector",
    "year": 2025,
    "total_transactions": 127,
    "income_transactions": 45,
    "expense_transactions": 82
  }
}
```

## Transaction Categories

Transactions are automatically categorized for tax reporting:

| Category | Description |
|----------|-------------|
| `gross_receipts` | Revenue from sales/services |
| `advertising` | Marketing and ads (Google, Facebook, etc.) |
| `car_truck_expenses` | Vehicle-related costs |
| `commissions_fees` | Payment processing fees |
| `contract_labor` | Freelancer/contractor payments |
| `insurance` | Business insurance |
| `interest_paid` | Interest on loans |
| `legal_professional_services` | Legal, accounting, consulting |
| `office_expense` | Office supplies and equipment |
| `rent_lease` | Office rent, equipment leases |
| `supplies` | AWS, software, general supplies |
| `taxes_licenses` | Business licenses, permits |
| `travel` | Flights, hotels, transportation |
| `meals` | Business meals, client lunches |
| `utilities` | Electric, phone, internet |
| `wages` | Employee payroll |
| `other_deductions` | Uncategorized expenses |
| `interest_income` | Bank interest |
| `dividend_income` | Investment dividends |

## API Configuration

API tokens are read from environment variables only (never hardcoded):

- `MERCURY_API_TOKEN` - Mercury API token
- `WISE_API_TOKEN` - Wise API token

Get your tokens from:
- Mercury: https://app.mercury.com/settings/api
- Wise: https://wise.com/user/settings/api-tokens

## Running Tests

```bash
cd /home/admin/clawd/formbridge
source .venv/bin/activate

# Run connector tests
pytest tests/test_connectors.py -v

# Run all tests (verify no regressions)
pytest tests/ -v
```

## Implementation Notes

1. **Amounts in cents internally** - All monetary amounts are stored as integers (cents) to avoid floating point issues. Converted to dollars only for output.

2. **Pagination handling** - Both connectors handle paginated API responses to fetch all transactions.

3. **Date handling** - ISO 8601 format throughout (YYYY-MM-DD).

4. **Error handling** - Clear error messages for authentication failures, API errors, and missing data.

5. **Type safety** - Full type hints for all functions and methods.

6. **Mocked tests** - All tests mock HTTP calls; no real API calls during testing.
