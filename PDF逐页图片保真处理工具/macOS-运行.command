#!/bin/bash
cd "$(dirname "$0")"
if [ "$#" -gt 0 ]; then
    python3 pdf_image_processor.py "$@"
else
    pdf_path=$(osascript -e 'POSIX path of (choose file with prompt "选择需要处理的 PDF" of type {"com.adobe.pdf"})')
    if [ -n "$pdf_path" ]; then
        python3 pdf_image_processor.py "$pdf_path"
    fi
fi
status=$?
echo
read -r -p "按回车键关闭..."
exit $status
