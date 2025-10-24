#!/bin/bash
# Test script to verify Google Drive upload with rclone

set -e  # Exit on any error

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "============================================================"
echo "  Google Drive Upload Test"
echo "============================================================"
echo ""

# Configuration
GDRIVE_REMOTE="gdrive"
GDRIVE_FOLDER="MonteCarloSimulations"
TEST_FILE="gdrive_test_$(date +%Y%m%d_%H%M%S).txt"

# Create a test file
echo -e "${BLUE}[INFO]${NC} Creating test file: ${TEST_FILE}"
echo "This is a test file created at $(date)" > "${TEST_FILE}"
echo "Hostname: $(hostname)" >> "${TEST_FILE}"
echo "User: $(whoami)" >> "${TEST_FILE}"
echo "Working directory: $(pwd)" >> "${TEST_FILE}"
echo "" >> "${TEST_FILE}"
echo "If you can see this file in Google Drive, rclone is working correctly!" >> "${TEST_FILE}"

echo -e "${GREEN}[SUCCESS]${NC} Test file created"
echo ""

# Check if rclone is available
if ! command -v rclone &> /dev/null; then
    echo -e "${RED}[ERROR]${NC} rclone is not installed or not in PATH"
    exit 1
fi

# Check if Google Drive remote is configured
if ! rclone listremotes | grep -q "${GDRIVE_REMOTE}:"; then
    echo -e "${RED}[ERROR]${NC} rclone remote '${GDRIVE_REMOTE}' is not configured"
    echo "Please run: rclone config"
    exit 1
fi

echo -e "${BLUE}[INFO]${NC} rclone remote '${GDRIVE_REMOTE}' found"
echo ""

# List current remotes
echo -e "${BLUE}[INFO]${NC} Available rclone remotes:"
rclone listremotes
echo ""

# Test listing Google Drive root
echo -e "${BLUE}[INFO]${NC} Testing Google Drive access (listing root folder)..."
if rclone lsd "${GDRIVE_REMOTE}:" 2>&1; then
    echo -e "${GREEN}[SUCCESS]${NC} Successfully accessed Google Drive"
    echo ""
else
    echo -e "${RED}[ERROR]${NC} Failed to access Google Drive"
    exit 1
fi

# Upload test file
echo -e "${BLUE}[INFO]${NC} Uploading test file to ${GDRIVE_REMOTE}:${GDRIVE_FOLDER}/"
rclone copy "${TEST_FILE}" "${GDRIVE_REMOTE}:${GDRIVE_FOLDER}/" -v

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}[SUCCESS]${NC} Test file uploaded successfully!"
    echo ""
    echo "File location in Google Drive:"
    echo "  ${GDRIVE_FOLDER}/${TEST_FILE}"
    echo ""
    echo "Check your Google Drive at:"
    echo "  https://drive.google.com/"
    echo ""

    # Clean up local test file
    echo -e "${BLUE}[INFO]${NC} Cleaning up local test file..."
    rm "${TEST_FILE}"
    echo -e "${GREEN}[SUCCESS]${NC} Local test file removed"
    echo ""

    echo "============================================================"
    echo -e "${GREEN}  Google Drive upload is working correctly!${NC}"
    echo "============================================================"
else
    echo ""
    echo -e "${RED}[ERROR]${NC} Upload failed"
    echo "Keeping local test file: ${TEST_FILE}"
    exit 1
fi
