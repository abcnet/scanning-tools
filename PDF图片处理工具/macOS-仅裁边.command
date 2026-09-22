#!/bin/bash
cd "$(dirname "$0")"
python3 pdf_image_processor.py --mode border-only --force-process "$@"
status=$?
echo
read -r -p "按 Return 关闭窗口……"
exit $status
