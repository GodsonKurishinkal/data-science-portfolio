#!/bin/bash
# Setup script for Data Science Portfolio
# This script sets up the virtual environment and installs dependencies

set -e  # Exit on error

echo "=================================="
echo "Data Science Portfolio Setup"
echo "=================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo -e "${BLUE}Checking Python version...${NC}"
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

if [[ $(echo $PYTHON_VERSION | cut -d. -f1) -lt 3 ]] || [[ $(echo $PYTHON_VERSION | cut -d. -f2) -lt 8 ]]; then
    echo -e "${YELLOW}Warning: Python 3.8+ is recommended${NC}"
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo -e "${BLUE}Creating virtual environment...${NC}"
    python3 -m venv .venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment already exists${NC}"
fi

# Activate virtual environment
echo -e "${BLUE}Activating virtual environment...${NC}"
source .venv/bin/activate

# Upgrade pip
echo -e "${BLUE}Upgrading pip...${NC}"
pip install --upgrade pip --quiet
echo -e "${GREEN}✓ pip upgraded${NC}"

# Install project 001 dependencies
if [ -f "project-001-demand-forecasting-system/requirements.txt" ]; then
    echo -e "${BLUE}Installing Project 001 dependencies...${NC}"
    pip install -r project-001-demand-forecasting-system/requirements.txt --quiet
    echo -e "${GREEN}✓ Project 001 dependencies installed${NC}"
fi

# Install project 001 in development mode
if [ -f "project-001-demand-forecasting-system/setup.py" ]; then
    echo -e "${BLUE}Installing Project 001 package...${NC}"
    pip install -e project-001-demand-forecasting-system/ --quiet
    echo -e "${GREEN}✓ Project 001 package installed${NC}"
fi

# Display installed packages
echo ""
echo -e "${BLUE}Installed packages:${NC}"
pip list | head -20

echo ""
echo "=================================="
echo -e "${GREEN}✓ Setup Complete!${NC}"
echo "=================================="
echo ""
echo "To activate the virtual environment manually:"
echo "  source .venv/bin/activate"
echo ""
echo "To run Project 001 demo:"
echo "  cd project-001-demand-forecasting-system"
echo "  python demo.py"
echo ""
echo "To deactivate:"
echo "  deactivate"
echo ""
