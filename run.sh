# This shell script is for executing the prediction process, which includes one step.
# Check if the correct number of arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <predict_file_path> <reference_path>"
    exit 1
fi

INPUT_PATH="$1"
REFERENCE_PATH="$2"

# Run predict.py 
# It will report the matraces
python predict.py \
  --predict_file "$INPUT_PATH" \
  --reference_file "$REFERENCE_PATH"