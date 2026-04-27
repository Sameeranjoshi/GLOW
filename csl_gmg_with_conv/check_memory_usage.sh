#!/bin/bash
# Script to check actual memory usage from compiled ELF files using cs-readelf
#
# Usage: ./check_memory_usage.sh <elf_dir> [tile_x] [tile_y]
# Example: ./check_memory_usage.sh out_vcycle
# Example: ./check_memory_usage.sh out_vcycle 1 1

ELF_DIR=${1:-"out_vcycle"}
TILE_X=${2:-"0"}
TILE_Y=${3:-"0"}
SUMMARY=${4:-"true"}
if [ "$SUMMARY" = "true" ]; then
    echo "Summary mode enabled"
    SUMMARY_MODE="--summary"
else
    SUMMARY_MODE=""
fi
echo "=============================================="
echo "Checking Memory Usage from ELF Files"
echo "=============================================="

# Check if cs-readelf or cs_readelf exists
CS_READELF=""
if command -v cs_readelf &> /dev/null; then
    CS_READELF="cs_readelf"
else
    echo "ERROR: cs-readelf not found"
    echo "Please run this from the project root where cs_readelf wrapper exists, or:"
    echo "  - Add Cerebras SDK bin to PATH"
    echo "  - Create a wrapper script cs_readelf in current directory"
    exit 1
fi

echo "Using tool: $CS_READELF"
echo "ELF Directory: $ELF_DIR"
echo "Checking PE at tile ($TILE_X, $TILE_Y)"

# Find a sample PE ELF file
SAMPLE_ELF=$(find "$ELF_DIR/bin" -name "out_*.elf" | head -1)

if [ -z "$SAMPLE_ELF" ]; then
    echo "ERROR: No ELF files found in $ELF_DIR/bin/"
    exit 1
fi

echo "Sample PE ELF: $SAMPLE_ELF"

# Use cs-readelf to get memory size per PE
# Capture the full output
MEM_OUTPUT=$($CS_READELF -m "$SAMPLE_ELF" 2>&1 | sed 's/\x1b\[[0-9;]*m//g')

# Try to extract actual memory usage in bytes from MEM_OUTPUT (works if cs-readelf prints as:
#   "(X, Y): 33976 bytes"
# )
TOTAL_SIZE=$(echo "$MEM_OUTPUT" | grep -Eo '([0-9]+)[[:space:]]+bytes' | head -1 | awk '{print $1}')

echo ""
# Parse symbols to categorize by type (FUNC=code, OBJECT=data)
# Strip ANSI color codes so awk patterns match cleanly
SYMBOLS_OUTPUT=$($CS_READELF --symbols "$SAMPLE_ELF" 2>&1 | sed 's/\x1b\[[0-9;]*m//g')

# Extract function symbols (code)
FUNC_SYMBOLS=$(echo "$SYMBOLS_OUTPUT" | awk '/FUNC/ && $3 > 0 {print $3}')
CODE_SIZE=$(echo "$FUNC_SYMBOLS" | awk '{sum += $1} END {print sum+0}')

# Extract object symbols (data)
OBJECT_SYMBOLS=$(echo "$SYMBOLS_OUTPUT" | awk '/OBJECT/ && $3 > 0 {print $3}')
DATA_SIZE=$(echo "$OBJECT_SYMBOLS" | awk '{sum += $1} END {print sum+0}')

# Count symbols
FUNC_COUNT=$(echo "$FUNC_SYMBOLS" | grep -c '[0-9]')
OBJECT_COUNT=$(echo "$OBJECT_SYMBOLS" | grep -c '[0-9]')

echo "  Code (FUNC symbols):      $CODE_SIZE bytes ($(awk "BEGIN {printf \"%.2f\", $CODE_SIZE/1024}") KB) - $FUNC_COUNT functions"
echo "  Data (OBJECT symbols):    $DATA_SIZE bytes ($(awk "BEGIN {printf \"%.2f\", $DATA_SIZE/1024}") KB) - $OBJECT_COUNT objects"
echo ""
ACCOUNTED=$((CODE_SIZE + DATA_SIZE))
echo "  Total accounted:          $ACCOUNTED bytes ($(awk "BEGIN {printf \"%.2f\", $ACCOUNTED/1024}") KB)"
if [ -n "$TOTAL_SIZE" ] && [ "$TOTAL_SIZE" -gt 0 ]; then
    OVERHEAD=$((TOTAL_SIZE - ACCOUNTED))
    echo "  Overhead (stack/other):   $OVERHEAD bytes ($(awk "BEGIN {printf \"%.2f\", $OVERHEAD/1024}") KB)"
fi
echo "=============================================="

if [ "$SUMMARY" = "false" ]; then
    echo ""
    echo "=============================================="
    echo "Top 10 Largest Data Symbols:"
    echo "=============================================="
    SYMBOL_DATA=$($CS_READELF --symbols "$SAMPLE_ELF" 2>&1 | sed 's/\x1b\[[0-9;]*m//g' | awk '/OBJECT/ && $3 > 0 {print $3, $NF}')
    if [ -n "$SYMBOL_DATA" ]; then
        echo "$SYMBOL_DATA" | sort -k1,1nr | head -20 | awk '{printf "  %-40s %8s bytes\n", $2, $1}'

        # Calculate total data symbol size
        TOTAL_DATA_SYMBOLS=$(echo "$SYMBOL_DATA" | awk '{sum += $1} END {print sum}')
        if [ -n "$TOTAL_DATA_SYMBOLS" ] && [ "$TOTAL_DATA_SYMBOLS" -gt 0 ]; then
            echo ""
            echo "Total from all data symbols (OBJECT type): $TOTAL_DATA_SYMBOLS bytes ($(awk "BEGIN {printf \"%.2f\", $TOTAL_DATA_SYMBOLS/1024}") KB)"
        fi
    else
        echo "  No OBJECT symbols found."
    fi

    echo ""
    echo "=============================================="
    echo "Top 10 Largest Code Symbols:"
    echo "=============================================="
    SYMBOL_CODE=$($CS_READELF --symbols "$SAMPLE_ELF" 2>&1 | sed 's/\x1b\[[0-9;]*m//g' | awk '/FUNC/ && $3 > 0 {print $3, $NF}')
    if [ -n "$SYMBOL_CODE" ]; then
        echo "$SYMBOL_CODE" | sort -k1,1nr | head -20 | awk '{printf "  %-40s %8s bytes\n", $2, $1}'

        TOTAL_CODE_SYMBOLS=$(echo "$SYMBOL_CODE" | awk '{sum += $1} END {print sum}')
        if [ -n "$TOTAL_CODE_SYMBOLS" ] && [ "$TOTAL_CODE_SYMBOLS" -gt 0 ]; then
            echo ""
            echo "Total from all code symbols (FUNC type): $TOTAL_CODE_SYMBOLS bytes ($(awk "BEGIN {printf \"%.2f\", $TOTAL_CODE_SYMBOLS/1024}") KB)"
        fi
    else
        echo "  No FUNC symbols found."
    fi
fi


# echo "=============================================="
# echo ""
# echo "Useful commands:"
# echo "  # Show all symbols with sizes and banks:"
# echo "    $CS_READELF --symbols --tile $TILE_X,$TILE_Y $SAMPLE_ELF"
# echo ""
# echo "  # Show fabric size:"
# echo "    $CS_READELF -s $SAMPLE_ELF"
# echo ""
# echo "  # Show memory usage in words instead of bytes:"
# echo "    $CS_READELF -m -u word --tile $TILE_X,$TILE_Y $SAMPLE_ELF"
# echo ""
# echo "  # Visualize which PEs have code/data:"
# echo "    $CS_READELF --visualize $SAMPLE_ELF"



