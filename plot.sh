# Check if the correct number of arguments are provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <data_file_path> <record_file_path> <output_path>"
    exit 1
fi

DATA_FILE_PATH="$1"
RECORD_FILE_PATH="$2"
OUtPUT_DIR="$3"

# Run plot.py 
# It will report the plots that we use in the report and slides.
python plot.py \
  --data_file "$DATA_FILE_PATH" \
  --record_file "$RECORD_FILE_PATH" \
  --output_dir "$OUTPUT_DIR"