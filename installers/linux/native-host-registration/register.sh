#!/bin/sh
set -eu

if [ "$#" -ne 1 ]; then
    echo "usage: register.sh /absolute/path/to/hsk-manga-native-host" >&2
    exit 2
fi
host_path=$1
case "$host_path" in
    /*) ;;
    *) echo "native host path must be absolute" >&2; exit 2 ;;
esac
if [ "$(printf '%s' "$host_path" | LC_ALL=C tr -d '\001-\037\177')" != "$host_path" ]; then
    echo "native host path must not contain control characters" >&2
    exit 2
fi
if [ ! -f "$host_path" ] || [ ! -x "$host_path" ] || [ "$(basename "$host_path")" != "hsk-manga-native-host" ]; then
    echo "native host executable is missing or has the wrong name" >&2
    exit 2
fi

manifest_directory="${HOME}/.mozilla/native-messaging-hosts"
manifest_path="${manifest_directory}/local.hskify.hsk_manga.json"
escaped_path=$(printf '%s' "$host_path" | sed 's/\\/\\\\/g; s/"/\\"/g')
umask 077
mkdir -p "$manifest_directory"
{
    printf '%s\n' '{'
    printf '%s\n' '  "name": "local.hskify.hsk_manga",'
    printf '%s\n' '  "description": "Hskify local browser companion",'
    printf '  "path": "%s",\n' "$escaped_path"
    printf '%s\n' '  "type": "stdio",'
    printf '%s\n' '  "allowed_extensions": ["hsk-manga-translator@local.hskify"]'
    printf '%s\n' '}'
} > "$manifest_path"
chmod 600 "$manifest_path"
printf '%s\n' "$manifest_path"
