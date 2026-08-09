# Firefox native-host registration (Windows)

The per-user installer runs `Register-NativeHost.ps1` with the absolute path to
`hskify-native-host.exe`. The script writes the manifest under the current
user's local application-data directory and registers its absolute path at:

```text
HKCU\Software\Mozilla\NativeMessagingHosts\local.hskify.browser
```

The manifest permits exactly
`hskify@local.hskify`. Uninstall runs
`Unregister-NativeHost.ps1`; daemon cache/state cleanup belongs to the main
companion uninstaller.

From the repository root, run
`powershell -File installers/test-native-host-registration.ps1` to exercise
registration and unregistration against isolated temporary manifest and
registry paths. The regression also proves that a directory named
`hskify-native-host.exe` is rejected before installer state changes.
