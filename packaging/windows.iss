#ifndef AppVersion
  #error AppVersion must be supplied from auto_reframe_core.version
#endif
#ifndef BuildRoot
  #error BuildRoot is required
#endif
#ifndef OutputRoot
  #error OutputRoot is required
#endif
[Setup]
AppId={{2E21E5E6-EF04-4E5A-95B4-798C07A96B9A}
AppName=Auto Reframe Videos
AppVersion={#AppVersion}
AppPublisher=minhung1126
DefaultDirName={autopf}\Auto Reframe Videos
DefaultGroupName=Auto Reframe Videos
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0.19045
OutputDir={#OutputRoot}
OutputBaseFilename=auto-reframe-videos-v{#AppVersion}-windows-x64-Setup
SetupIconFile={#SourcePath}\..\build\desktop\app.ico
UninstallDisplayIcon={app}\Auto Reframe Videos.exe
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
WizardSizePercent=120
CloseApplications=yes
CloseApplicationsFilter=Auto Reframe Videos.exe
RestartApplications=no
AppMutex=AutoReframeVideosDesktop
LicenseFile={#SourcePath}\..\LICENSE
[LangOptions]
DialogFontName=Microsoft JhengHei UI
DialogFontSize=10
[Files]
Source: "{#BuildRoot}\Auto Reframe Videos\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
[Icons]
Name: "{group}\Auto Reframe Videos"; Filename: "{app}\Auto Reframe Videos.exe"
Name: "{autodesktop}\Auto Reframe Videos"; Filename: "{app}\Auto Reframe Videos.exe"; Tasks: desktopicon
[Tasks]
Name: desktopicon; Description: "建立桌面捷徑"; Flags: unchecked
[Run]
Filename: "{app}\Auto Reframe Videos.exe"; Description: "啟動 Auto Reframe Videos"; Flags: nowait postinstall skipifsilent runasoriginaluser
; No user-data delete entries. Upgrades replace only {app}.

[Code]
procedure InitializeWizard;
begin
  { Scaled pixel minimums leave space around native checkbox glyphs at high DPI.
    Native Windows screenshots at 100/125/150/200% are still required. }
  WizardForm.TasksList.MinItemHeight := ScaleY(28);
  WizardForm.RunList.MinItemHeight := ScaleY(28);
end;
