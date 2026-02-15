#!/bin/bash
# Run connector tests

set -e

cd /home/admin/clawd/formbridge
source .venv/bin/activate

echo "Testing imports..."
python -c "from formbridge.connectors import MercuryConnector, WiseConnector, EntityInfo, TransactionCategory; print('Imports OK')"

echo ""
echo "Running connector tests..."
pytest tests/test_connectors.py -v --tb=short

echo ""
echo "Running all tests to check for regressions..."
pytest tests/ -v --tb=short -q 2>&1 | tail -30
