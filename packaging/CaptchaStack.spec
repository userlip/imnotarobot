from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

datas, binaries, hiddenimports = [], [], []
for pkg in ("ddddocr", "onnxruntime", "rumps", "WebKit", "patchright"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h
hiddenimports += collect_submodules("uvicorn") + ["captcha_stack"]
hiddenimports += collect_submodules("captcha_stack")
# mcp.cli exits on import without its optional CLI extras, so it is left out
hiddenimports += collect_submodules("mcp", filter=lambda name: not name.startswith("mcp.cli"))
datas += collect_data_files("mcp")
datas += collect_data_files("captcha_stack")

a = Analysis(["app_main.py"], pathex=[".."], datas=datas, binaries=binaries, hiddenimports=hiddenimports,
             excludes=["tkinter", "matplotlib", "pytest"])
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="YaNeRobot", console=False, argv_emulation=False)
coll = COLLECT(exe, a.binaries, a.datas, name="YaNeRobot")
app = BUNDLE(coll, name="Я не робот.app", icon="icon.icns", bundle_identifier="app.captcha-stack",
             info_plist={"LSUIElement": True, "NSAppTransportSecurity": {"NSAllowsLocalNetworking": True}, "CFBundleShortVersionString": "0.3.0", "CFBundleDisplayName": "Я не робот", "CFBundleName": "Я не робот",
                         "NSHumanReadableCopyright": "ziganshin"})
