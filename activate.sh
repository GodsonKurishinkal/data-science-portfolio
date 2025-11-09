#!/bin/bash

# Quick activation script for data-science-portfolio virtual environment
# Usage: source activate.sh (or . activate.sh)

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Data Science Portfolio Environment${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if .venv exists
if [ ! -d ".venv" ]; then
    echo ""
    echo "⚠️  Virtual environment not found!"
    echo "Creating .venv with Python 3.11.14..."
    python3.11 -m venv .venv
    if [ $? -eq 0 ]; then
        echo "✓ Virtual environment created successfully"
    else
        echo "✗ Failed to create virtual environment"
        return 1
    fi
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Verify activation
if [ "$VIRTUAL_ENV" != "" ]; then
    echo -e "${GREEN}✓ Virtual environment activated${NC}"
    echo ""
    echo "Python version: $(python --version)"
    echo "Python location: $(which python)"
    echo "Pip version: $(pip --version | cut -d' ' -f1-2)"
    echo ""
    echo -e "${BLUE}Available projects:${NC}"
    echo "  • project-001-demand-forecasting-system"
    echo "  • project-002-inventory-optimization-engine"
    echo ""
    echo -e "${GREEN}Ready to work! 🚀${NC}"
    echo "To deactivate: ${YELLOW}deactivate${NC}"
else
    echo -e "${YELLOW}⚠ Failed to activate virtual environment${NC}"
fi

echo ""
