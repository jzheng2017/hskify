#!/bin/sh
set -eu

repository_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
temporary_root=$(mktemp -d "${TMPDIR:-/tmp}/hsk-manga-registration.XXXXXX")
trap 'rm -rf "$temporary_root"' EXIT HUP INT TERM

unicode_directory="${temporary_root}/翻訳ツール"
host_path="${unicode_directory}/hsk-manga-native-host"
mkdir -p "$unicode_directory"
printf '%s\n' '#!/bin/sh' 'exit 0' > "$host_path"
chmod 700 "$host_path"
directory_host="${temporary_root}/directory/hsk-manga-native-host"
mkdir -p "$directory_host"
chmod 700 "$directory_host"

for platform in linux macos; do
    test_home="${temporary_root}/home-${platform}"
    mkdir -p "$test_home"
    register_script="${repository_root}/installers/${platform}/native-host-registration/register.sh"
    manifest_path=$(HOME="$test_home" sh "$register_script" "$host_path")
    test -f "$manifest_path"
    grep -F "\"path\": \"${host_path}\"," "$manifest_path" > /dev/null
    unregister_script="${repository_root}/installers/${platform}/native-host-registration/unregister.sh"
    HOME="$test_home" sh "$unregister_script"
    test ! -e "$manifest_path"

    directory_error="${temporary_root}/${platform}-directory-error.txt"
    if HOME="$test_home" sh "$register_script" "$directory_host" > /dev/null 2> "$directory_error"; then
        echo "${platform} registration accepted a directory as the native host executable" >&2
        exit 1
    fi
    grep -F 'native host executable is missing or has the wrong name' "$directory_error" > /dev/null
    test ! -e "$manifest_path"

    control_path="${temporary_root}/control
path/hsk-manga-native-host"
    error_output="${temporary_root}/${platform}-control-error.txt"
    if HOME="$test_home" sh "$register_script" "$control_path" > /dev/null 2> "$error_output"; then
        echo "${platform} registration accepted a control character in the host path" >&2
        exit 1
    fi
    grep -F 'native host path must not contain control characters' "$error_output" > /dev/null
done

printf '%s\n' "native-host registration path checks passed"
