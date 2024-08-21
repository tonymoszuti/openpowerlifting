URL="https://openpowerlifting.gitlab.io/opl-csv/files/openipf-latest.zip"
ZIP_FILE="/workspace/runtime/postgresql/data/openipf-latest.zip"
UNZIP_DIR="/workspace/runtime/postgresql/data"

# Download the ZIP file
echo "Downloading $ZIP_FILE..."
curl -o "$ZIP_FILE" -L "$URL"

# Check if the download was successful
if [ $? -ne 0 ]; then
    echo "Failed to download $ZIP_FILE"
    exit 1
fi

echo "Download complete."

# Create the directory to unzip the files
mkdir -p "$UNZIP_DIR"

# Unzip the data file
unzip -l "$ZIP_FILE" | grep ".csv" | awk '{print $4}' | while read csv_file; do
    echo "Extracting $csv_file from $ZIP_FILE..."
    unzip -j "$ZIP_FILE" "$csv_file" -d "$UNZIP_DIR"
done

# Check if the unzip was successful
if [ $? -ne 0 ]; then
    echo "Failed to unzip $ZIP_FILE"
    exit 1
fi

echo "Unzip complete."

# Remove the ZIP file after unzipping
rm "$ZIP_FILE"

echo "Done!"