# Firefox native-host registration (Linux)

The per-user installer invokes `register.sh` with the absolute installed native
host path. It writes a mode-0600 manifest to
`~/.mozilla/native-messaging-hosts`. The manifest allows only the permanent
Firefox add-on ID.

`unregister.sh` removes only that manifest. Main application cache/state cleanup
belongs to the companion uninstaller.
