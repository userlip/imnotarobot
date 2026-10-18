import subprocess
import threading
from pathlib import Path

import rumps

from .control import Controller

ICON = Path(__file__).with_name("assets") / "menubar.png"


def install_edit_menu() -> None:
    """Gives the app an Edit menu: without one, Cmd+V/C/X/A do nothing in its windows (menu bar apps have no menu)."""
    from AppKit import NSApplication, NSMenu, NSMenuItem

    app = NSApplication.sharedApplication()
    main = app.mainMenu() or NSMenu.alloc().init()
    edit = NSMenu.alloc().initWithTitle_("Edit")
    for title, action, key in [("Undo", "undo:", "z"), ("Redo", "redo:", "Z"), ("Cut", "cut:", "x"),
                               ("Copy", "copy:", "c"), ("Paste", "paste:", "v"), ("Select All", "selectAll:", "a")]:
        edit.addItemWithTitle_action_keyEquivalent_(title, action, key)
    window = NSMenu.alloc().initWithTitle_("Window")
    window.addItemWithTitle_action_keyEquivalent_("Close", "performClose:", "w")
    for menu in (edit, window):
        holder = NSMenuItem.alloc().init()
        holder.setSubmenu_(menu)
        main.addItem_(holder)
    app.setMainMenu_(main)


def clipboard_text() -> str:
    return subprocess.run(["pbpaste"], capture_output=True, text=True, check=False).stdout.strip()


def copy_to_clipboard(text: str) -> None:
    subprocess.run(["pbcopy"], input=text.encode(), check=False)


class SettingsWindow:
    """A native window showing the settings page served by the app itself."""

    def __init__(self, url: str):
        self.url = url
        self.window = None

    def show(self) -> None:
        from AppKit import (NSApplication, NSBackingStoreBuffered, NSMakeRect, NSWindow, NSWindowStyleMaskClosable,
                            NSWindowStyleMaskMiniaturizable, NSWindowStyleMaskResizable, NSWindowStyleMaskTitled)
        from Foundation import NSURL, NSURLRequest
        from WebKit import WKWebView, WKWebViewConfiguration

        if self.window is None:
            style = (NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskResizable
                     | NSWindowStyleMaskMiniaturizable)
            w = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
                NSMakeRect(0, 0, 900, 780), style, NSBackingStoreBuffered, False)
            w.setTitle_("Captcha Stack")
            w.setReleasedWhenClosed_(False)
            w.setMinSize_((560, 420))
            w.center()
            web = WKWebView.alloc().initWithFrame_configuration_(w.contentView().bounds(),
                                                                 WKWebViewConfiguration.alloc().init())
            web.setAutoresizingMask_(2 | 16)  # width and height follow the window
            w.setContentView_(web)
            web.loadRequest_(NSURLRequest.requestWithURL_(NSURL.URLWithString_(self.url)))
            self.window = w
        NSApplication.sharedApplication().activateIgnoringOtherApps_(True)
        self.window.makeKeyAndOrderFront_(None)


class MenuBarApp(rumps.App):
    def __init__(self, controller: Controller):
        # template image: macOS recolours it for light and dark menu bars like system icons
        super().__init__("Captcha Stack", icon=str(ICON), template=True, quit_button=None)
        self.c = controller
        self.window = SettingsWindow(f"{controller.address}/ui?t={controller.ui_token}")
        self._started = False
        self.rebuild()
        rumps.Timer(self.tick, 2).start()
        threading.Thread(target=self._check_human, daemon=True).start()

    def _check_human(self) -> None:
        if self.c.cfg.human_api_key:
            self.c.check_human()

    def rebuild(self) -> None:
        c = self.c
        status = rumps.MenuItem("⏸ На паузе — продолжить" if c.service.paused else "● Работает — поставить на паузу",
                                callback=self.toggle_pause)
        self.today = rumps.MenuItem(c.today_line())
        items = [status, self.today, None, rumps.MenuItem("Открыть Captcha Stack…", callback=self.open_window, key=","),
                 None]
        if not c.cfg.keys:
            items.append(rumps.MenuItem("Облачных моделей нет — добавить", callback=self.open_window))
        for entry in c.cfg.keys:
            enabled = sum(1 for _, on in c.key_models(entry.name) if on)
            mark = "✓" if enabled else "○"
            items.append(rumps.MenuItem(f"{mark} {entry.name}: включено моделей {enabled}", callback=self.open_window))
        items.append(rumps.MenuItem(c.human_line(), callback=self.open_window))
        items += [rumps.MenuItem("Добавить ключ из буфера обмена", callback=self.add_key_from_clipboard), None,
                  rumps.MenuItem(f"Скопировать адрес {c.address}", callback=self.copy_address),
                  rumps.MenuItem("Выход", callback=self.quit, key="q")]
        self.menu.clear()
        self.menu = items

    def tick(self, _=None) -> None:
        if not self._started:
            self._started = True
            install_edit_menu()
            if not self.c.cfg.keys and not self.c.cfg.human_api_key:
                self.open_window()
        self.today.title = self.c.today_line()
        job = self.c.bench
        self.title = f"{job['done']}/{job['total']}" if job.get("running") else None
        signature = (self.c.service.paused, self.c.human_line(), tuple(
            (k.name, sum(1 for _, on in self.c.key_models(k.name) if on)) for k in self.c.cfg.keys))
        if signature != getattr(self, "_signature", None):
            self._signature = signature
            self.rebuild()

    def open_window(self, _=None) -> None:
        self.window.show()

    def toggle_pause(self, _) -> None:
        self.c.set_paused(not self.c.service.paused)
        rumps.notification("Captcha Stack", "", "Решатель на паузе" if self.c.service.paused else "Решатель снова работает")
        self.rebuild()

    def add_key_from_clipboard(self, _) -> None:
        secret = clipboard_text()
        if not secret or " " in secret or len(secret) < 12:
            rumps.alert("В буфере нет ключа", "Скопируйте ключ API и выберите пункт ещё раз.")
            return
        try:
            entry = self.c.add_key(secret, "")
        except Exception as e:
            rumps.alert("Ключ не подошёл", f"{e}\n\nЕсли это ключ посредника, добавьте его в окне и укажите адрес.")
            return
        n = len(self.c.key_models(entry.name))
        rumps.alert("Ключ добавлен", f"{entry.name}: моделей с картинками — {n}. Отметьте нужные в окне Captcha Stack.")
        self.rebuild()
        self.open_window()

    def copy_address(self, _) -> None:
        copy_to_clipboard(self.c.address)
        rumps.notification("Captcha Stack", "Адрес скопирован", self.c.address)

    def quit(self, _) -> None:
        self.c.stop_server()
        rumps.quit_application()


def main() -> None:
    controller = Controller()
    controller.start_server()
    MenuBarApp(controller).run()


if __name__ == "__main__":
    main()
