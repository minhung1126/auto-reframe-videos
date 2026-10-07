# Build on the target OS/CPU; PyInstaller is not a cross compiler.
import os
from pathlib import Path
if os.name == "nt":
    from PyInstaller.utils.win32.versioninfo import FixedFileInfo, StringFileInfo, StringTable, StringStruct, VarFileInfo, VarStruct, VSVersionInfo

root = Path(SPECPATH).parent
import sys
sys.path.insert(0, str(root))
from auto_reframe_core.version import VERSION, __version__
from auto_reframe_core.runtime_paths import APP_NAME

vendor = Path(os.environ['ARV_VENDOR_DIR']).resolve()
target = os.environ['ARV_TARGET']
metadata = Path(os.environ.get('ARV_METADATA_DIR', root / 'build' / 'desktop')).resolve()
datas = [(str(root / 'config.json.example'), '.'),
         (str(root / 'fonts' / 'NotoSerifTC.ttf'), 'fonts'),
         (str(root / 'fonts' / 'LICENSE'), 'fonts'),
         (str(root / 'LICENSE'), '.'),
         (str(root / 'THIRD_PARTY_NOTICES.md'), '.'),
         (str(vendor / 'licenses'), 'licenses'),
         (str(vendor / 'provenance.json'), '.'),
         (str(metadata / 'build-info.json'), '.'),
         (str(metadata / 'runtime-licenses'), 'licenses/runtime')]
import tkinterdnd2
dnd_platform = {"windows-x64": "win-x64", "macos-arm64": "osx-arm64", "macos-x64": "osx-x64"}.get(target, "linux-x64")
dnd_root = Path(tkinterdnd2.__file__).parent / "tkdnd" / dnd_platform
datas.append((str(dnd_root), "tkinterdnd2/tkdnd/" + dnd_platform))
import certifi
datas.append((certifi.where(), 'certs'))
binaries = [(str(p), 'bin') for p in (vendor / 'bin').iterdir()]
# Relocatable Python distributions can load Tcl/Tk from their private prefix
# even when the system loader cannot resolve _tkinter's dependencies for the hook.
if sys.platform == 'linux':
    private_lib = Path(sys.base_prefix) / 'lib'
    for pattern in ('libtcl*.so*', 'libtk*.so*'):
        binaries.extend((str(path), '.') for path in private_lib.glob(pattern) if path.is_file())
icon = metadata / ('app.ico' if target.startswith('windows') else 'app.icns')
version_info = None
if target.startswith('windows'):
    version_info = VSVersionInfo(
        ffi=FixedFileInfo(filevers=(*VERSION, 0), prodvers=(*VERSION, 0), mask=0x3f,
                         flags=0, OS=0x40004, fileType=1, subtype=0, date=(0, 0)),
        kids=[StringFileInfo([StringTable('040904B0', [
            StringStruct('FileDescription', APP_NAME),
            StringStruct('ProductName', APP_NAME),
            StringStruct('FileVersion', __version__),
            StringStruct('ProductVersion', __version__),
            StringStruct('LegalCopyright', 'All Rights Reserved'),
        ])]), VarFileInfo([VarStruct('Translation', [1033, 1200])])])
a = Analysis([str(root / 'auto_reframe_core' / '__main__.py')], pathex=[str(root)],
             binaries=binaries, datas=datas,
             hiddenimports=['auto_reframe_core.gui', 'tkinter', 'tkinter.filedialog',
                            'auto_reframe_core.desktop_smoke', 'tkinterdnd2'],
             excludes=['pytest', 'notebook', 'IPython'], noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name=APP_NAME, console=False,
          icon=str(icon), version=version_info, target_arch='arm64' if target.endswith('arm64') else 'x86_64',
          codesign_identity=None, entitlements_file=None)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name=APP_NAME)
if target.startswith('macos'):
    app = BUNDLE(coll, name=APP_NAME + '.app', icon=str(icon),
                 bundle_identifier='com.minhung1126.auto-reframe-videos',
                 info_plist={'CFBundleShortVersionString': __version__,
                             'CFBundleVersion': __version__,
                             'LSMinimumSystemVersion': '13.0',
                             'NSHighResolutionCapable': True})
