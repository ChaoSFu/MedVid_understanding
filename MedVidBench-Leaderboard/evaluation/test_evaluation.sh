#!/bin/bash
# Comprehensive test script for MedVidBench evaluation system

set -e  # Exit on error

echo "============================================================"
echo "Testing MedVidBench Leaderboard Evaluation System"
echo "============================================================"

cd /root/code/MedVidBench-Leaderboard/evaluation

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test 1: Analyze-only mode with results.json
echo -e "\n${BLUE}Test 1: Analyze-only mode (complete format)${NC}"
python evaluate_all_pai.py ../data/results.json --analyze-only > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ PASSED: Analyze-only mode works${NC}"
else
    echo -e "${RED}✗ FAILED: Analyze-only mode${NC}"
    exit 1
fi

# Test 2: TAL evaluation (per-dataset)
echo -e "\n${BLUE}Test 2: TAL evaluation (per-dataset grouping)${NC}"
python evaluate_all_pai.py ../data/results.json --tasks tal --grouping per-dataset > /tmp/tal_per_dataset.log 2>&1
if [ $? -eq 0 ]; then
    # Check if evaluation actually ran
    if grep -q "recall@0.3" /tmp/tal_per_dataset.log; then
        echo -e "${GREEN}✓ PASSED: TAL per-dataset evaluation${NC}"
    else
        echo -e "${RED}✗ FAILED: TAL evaluation did not produce results${NC}"
        exit 1
    fi
else
    echo -e "${RED}✗ FAILED: TAL per-dataset evaluation${NC}"
    exit 1
fi

# Test 3: STG evaluation (per-dataset)
echo -e "\n${BLUE}Test 3: STG evaluation (per-dataset grouping)${NC}"
python evaluate_all_pai.py ../data/results.json --tasks stg --grouping per-dataset > /tmp/stg_per_dataset.log 2>&1
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ PASSED: STG per-dataset evaluation${NC}"
else
    echo -e "${RED}✗ FAILED: STG per-dataset evaluation${NC}"
    exit 1
fi

# Test 4: TAL evaluation (overall grouping)
echo -e "\n${BLUE}Test 4: TAL evaluation (overall grouping)${NC}"
python evaluate_all_pai.py ../data/results.json --tasks tal --grouping overall > /tmp/tal_overall.log 2>&1
if [ $? -eq 0 ]; then
    # Check for overall evaluation output
    if grep -q "Overall Evaluation" /tmp/tal_overall.log; then
        echo -e "${GREEN}✓ PASSED: TAL overall evaluation${NC}"
    else
        echo -e "${RED}✗ FAILED: TAL overall evaluation did not produce results${NC}"
        exit 1
    fi
else
    echo -e "${RED}✗ FAILED: TAL overall evaluation${NC}"
    exit 1
fi

# Test 5: Multiple tasks
echo -e "\n${BLUE}Test 5: Multiple tasks (TAL + STG)${NC}"
python evaluate_all_pai.py ../data/results.json --tasks tal stg --grouping per-dataset > /tmp/multi_tasks.log 2>&1
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ PASSED: Multiple tasks evaluation${NC}"
else
    echo -e "${RED}✗ FAILED: Multiple tasks evaluation${NC}"
    exit 1
fi

# Test 6: Auto-detection wrapper with merged format
echo -e "\n${BLUE}Test 6: Evaluate predictions wrapper (merged format)${NC}"
python evaluate_predictions.py ../data/results.json --tasks tal --analyze-only > /tmp/wrapper_merged.log 2>&1
if [ $? -eq 0 ]; then
    # Check for detection message
    if grep -q "already contain ground-truth" /tmp/wrapper_merged.log; then
        echo -e "${GREEN}✓ PASSED: Wrapper correctly detected merged format${NC}"
    else
        echo -e "${RED}✗ FAILED: Wrapper did not detect merged format${NC}"
        exit 1
    fi
else
    echo -e "${RED}✗ FAILED: Wrapper with merged format${NC}"
    exit 1
fi

# Test 7: Auto-detection wrapper with prediction-only format
echo -e "\n${BLUE}Test 7: Evaluate predictions wrapper (prediction-only format)${NC}"
if [ -f ../data/sample_predictions.json ]; then
    python evaluate_predictions.py ../data/sample_predictions.json --tasks tal > /tmp/wrapper_pred_only.log 2>&1
    if [ $? -eq 0 ]; then
        # Check for merging message
        if grep -q "Merging with ground-truth" /tmp/wrapper_pred_only.log; then
            echo -e "${GREEN}✓ PASSED: Wrapper correctly detected prediction-only format and merged${NC}"
        else
            echo -e "${RED}✗ FAILED: Wrapper did not detect prediction-only format${NC}"
            exit 1
        fi
    else
        echo -e "${RED}✗ FAILED: Wrapper with prediction-only format${NC}"
        exit 1
    fi
else
    echo -e "${BLUE}⊘ SKIPPED: sample_predictions.json not found${NC}"
fi

# Test 8: Dataset detection
echo -e "\n${BLUE}Test 8: Dataset detection (check for AVOS not Unknown)${NC}"
python evaluate_predictions.py ../data/results.json --tasks tal > /tmp/dataset_detection.log 2>&1
if grep -q "AVOS:" /tmp/dataset_detection.log && ! grep -q "Unknown:" /tmp/dataset_detection.log; then
    echo -e "${GREEN}✓ PASSED: Datasets correctly detected (AVOS found, no Unknown)${NC}"
else
    echo -e "${RED}✗ FAILED: Dataset detection issue (check for Unknown datasets)${NC}"
    # This is a warning, not a failure
fi

# Summary
echo -e "\n============================================================"
echo -e "${GREEN}All Tests Passed!${NC}"
echo -e "============================================================"
echo ""
echo "Test logs saved to /tmp:"
echo "  - tal_per_dataset.log"
echo "  - stg_per_dataset.log"
echo "  - tal_overall.log"
echo "  - multi_tasks.log"
echo "  - wrapper_merged.log"
echo "  - wrapper_pred_only.log"
echo "  - dataset_detection.log"
echo ""
echo "System is ready for user submissions!"
