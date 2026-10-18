from PyInstaller.utils.hooks import collect_all, collect_submodules

datas, binaries, hiddenimports = [], [], []
for pkg in ("ddddocr", "onnxruntime", "rumps"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h
hiddenimports += collect_submodules("uvicorn") + collect_submodules("keyring.backends") + ["captcha_stack"]
hiddenimports += collect_submodules("captcha_stack")

a = Analysis(["app_main.py"], pathex=[".."], datas=datas, binaries=binaries, hiddenimports=hiddenimports,
             excludes=["tkinter", "matplotlib", "pytest"])
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="Captcha Stack", console=False, argv_emulation=False)
coll = COLLECT(exe, a.binaries, a.datas, name="Captcha Stack")
app = BUNDLE(coll, name="Captcha Stack.app", icon="icon.icns", bundle_identifier="app.captcha-stack",
             info_plist={"LSUIElement": True, "CFBundleShortVersionString": "0.2.0",
                         "NSHumanReadableCopyright": "ziganshin"})
