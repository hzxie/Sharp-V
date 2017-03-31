#!/bin/bash 
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <locale>"
    exit 1
fi

locale=$1
domain="messages"

locale_dir="../locales/${locale}/LC_MESSAGES"
po_file="${locale_dir}/${domain}.po"
mo_file="${locale_dir}/${domain}.mo"

# create .mo file from .po
msgfmt ${po_file} --output-file=${mo_file}
