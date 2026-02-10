#!/bin/bash
# BEANS Dataset Download Script
# =============================
# Quick setup and download for BEANS benchmark datasets
# (excluding CBI which requires Kaggle registration)

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}BEANS Dataset Downloader${NC}"
echo -e "${GREEN}========================================${NC}"

# Default output directory
OUTPUT_DIR="${1:-./beans_data}"

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is required but not installed.${NC}"
    exit 1
fi

# Install required Python packages
echo -e "\n${YELLOW}Installing required Python packages...${NC}"
pip install --quiet requests tqdm zenodo_get 2>/dev/null || {
    echo -e "${YELLOW}Warning: Could not install all packages. Continuing...${NC}"
}

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Download datasets
echo -e "\n${GREEN}Starting downloads...${NC}"
python3 download_beans_datasets.py --output-dir "$OUTPUT_DIR"

# Summary
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Download process complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "Data directory: $OUTPUT_DIR"
echo -e "\nDatasets that can be auto-downloaded:"
echo -e "  - ESC-50 (environmental sounds)"
echo -e "  - Speech Commands (spoken words)"
echo -e "\nDatasets requiring manual download:"
echo -e "  - Watkins (marine mammals) - see instructions in $OUTPUT_DIR/watkins/"
echo -e "  - Bats (Egyptian fruit bats) - large dataset, see $OUTPUT_DIR/bats/"
echo -e "  - HumBugDB (mosquitoes) - use zenodo_get or manual download"
echo -e "  - Dogs (dog barks) - may require author contact"
