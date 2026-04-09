#!/bin/bash
# Download all LEMON EEG subjects from S3 and fix .vhdr file references
# Usage: bash scripts/download_all.sh

set -e
DATA_DIR="data/lemon"
S3_BASE="s3://fcp-indi/data/Projects/INDI/MPI-LEMON/EEG_MPILMBB_LEMON/EEG_Raw_BIDS_ID"

# Get list of all subjects
echo "Listing subjects on S3..."
SUBJECTS=$(aws s3 ls "$S3_BASE/" --no-sign-request | grep "PRE" | awk '{print $2}' | tr -d '/')

TOTAL=$(echo "$SUBJECTS" | wc -l | tr -d ' ')
echo "Found $TOTAL subjects"

COUNT=0
SKIPPED=0
for SUB in $SUBJECTS; do
    COUNT=$((COUNT + 1))
    VHDR="$DATA_DIR/$SUB/RSEEG/$SUB.vhdr"

    # Skip if already downloaded and fixed
    if [ -f "$VHDR" ] && grep -q "DataFile=$SUB.eeg" "$VHDR" 2>/dev/null; then
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    echo "[$COUNT/$TOTAL] Downloading $SUB..."
    mkdir -p "$DATA_DIR/$SUB/RSEEG"
    aws s3 sync "$S3_BASE/$SUB/RSEEG/" "$DATA_DIR/$SUB/RSEEG/" --no-sign-request --quiet

    # Fix .vhdr internal references (old ID -> BIDS ID)
    if [ -f "$VHDR" ]; then
        OLD_DATA=$(grep "^DataFile=" "$VHDR" | cut -d= -f2)
        OLD_MARKER=$(grep "^MarkerFile=" "$VHDR" | cut -d= -f2)
        if [ "$OLD_DATA" != "$SUB.eeg" ]; then
            sed -i '' "s|^DataFile=.*|DataFile=$SUB.eeg|" "$VHDR"
            sed -i '' "s|^MarkerFile=.*|MarkerFile=$SUB.vmrk|" "$VHDR"
            echo "  Fixed refs: $OLD_DATA -> $SUB.eeg"
        fi
    fi
done

echo ""
echo "Done! Downloaded: $((COUNT - SKIPPED)), Skipped: $SKIPPED, Total: $TOTAL"
echo "Data directory: $DATA_DIR"
