#!/bin/sh
pip install -e ./openpowerlifting/[dev]
./runtime/scripts/download_data.sh
