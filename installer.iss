; installer.iss  —— 适配 dist\EggInspector\ 与 EggInspector.exe

#define MyAppName "Egg Inspector"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "Miller Zhou"
#define MyAppExeName "EggInspector.exe"

[Setup]
; 全局信息
AppId={{A3C6425D-0AC1-4205-9EF2-1C7E0C59CB10}}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}

; 安装位置与开始菜单
DefaultDirName={pf}\{#MyAppName}
DefaultGroupName={#MyAppName}

; 输出安装包到仓库根目录，文件名与 workflow 上传路径一致
OutputDir=.
OutputBaseFilename=EggInspector_Setup

; 其他常用设置
Compression=lzma
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64
DisableDirPage=no
DisableProgramGroupPage=no
; 如果不想要求管理员权限，可改为：PrivilegesRequired=lowest
; PrivilegesRequired=admin

[Files]
; 把 PyInstaller 生成的 onedir 全量复制到 {app}
Source: "dist\EggInspector\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
; 开始菜单快捷方式
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
; 可选：卸载程序快捷方式
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
; 桌面图标（默认不勾选）
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop icon"; GroupDescription: "Additional icons:"; Flags: unchecked

[Run]
; 安装完成后给个复选框：是否立即启动
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent
