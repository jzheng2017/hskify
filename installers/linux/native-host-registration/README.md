# Firefox native-host registration (Linux)

The per-user installer invokes `register.sh` with the absolute installed native
host path. It writes a mode-0600 manifest to
`~/.mozilla/native-messaging-hosts`. The manifest allows only the permanent
Firefox add-on ID.

`unregister.sh` removes only that manifest. Main application cache/state cleanup
belongs to the companion uninstaller.

From the repository root, run
`sh installers/test-native-host-registration.sh` to exercise both Unix
registration scripts with a valid non-ASCII UTF-8 executable path and a
control-character rejection case. It also exercises unregistration and rejects
a directory with the native-host filename. The check redirects `HOME` to a
temporary directory and does not modify the user's Firefox profile.
