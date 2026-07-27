# Firefox native-host registration utility (Linux)

These scripts preserve per-user registration behavior for development and
future packaging work. The current Hskify performance product is Windows
CUDA-only on an RTX 4080 SUPER; Linux execution and performance are not
supported or measured.

`register.sh` accepts the absolute native-host path and writes a mode-0600
manifest to `~/.mozilla/native-messaging-hosts`. It permits only
`hsk-manga-translator@local.hskify`. `unregister.sh` removes only that manifest.

`sh installers/test-native-host-registration.sh` exercises path validation and
isolated registration logic. Passing it does not establish a supported Linux
runtime or browser benchmark.
