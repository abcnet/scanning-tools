#!/bin/bash
cd "$(dirname "$0")"
python3 pdf_image_processor.py "$@"
status=$?
echo
read -r -p "按回车键关闭..."
exit $status
