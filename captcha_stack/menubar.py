import subprocess
import threading
import webbrowser
from pathlib import Path

import rumps

from . import bench
from .control import Controller

ICON = Path(__file__).with_name("assets") / "menubar.png"


def install_edit_menu() -> None:
    """Gives the app an Edit menu: without one, Cmd+V/C/X/A do nothing in its dialogs (menu bar apps have no menu)."""
    from AppKit import NSApplication, NSMenu, NSMenuItem

    app = NSApplication.sharedApplication()
    main = app.mainMenu() or NSMenu.alloc().init()
    edit = NSMenu.alloc().initWithTitle_("Edit")
    for title, action, key in [("Undo", "undo:", "z"), ("Cut", "cut:", "x"), ("Copy", "copy:", "c"),
                               ("Paste", "paste:", "v"), ("Select All", "selectAll:", "a")]:
        edit.addItemWithTitle_action_keyEquivalent_(title, action, key)
    holder = NSMenuItem.alloc().init()
    holder.setSubmenu_(edit)
    main.addItem_(holder)
    app.setMainMenu_(main)


def clipboard_text() -> str:
    return subprocess.run(["pbpaste"], capture_output=True, text=True, check=False).stdout.strip()


def ask(title: str, message: str, default: str = "", secure: bool = False) -> str | None:
    w = rumps.Window(message=message, title=title, default_text=default, ok="Сохранить", cancel="Отмена",
                     dimensions=(320, 24), secure=secure)
    r = w.run()
    return r.text.strip() if r.clicked else None


class MenuBarApp(rumps.App):
    def __init__(self, controller: Controller):
        # template image: macOS recolours it for light and dark menu bars like system icons
        super().__init__("captcha-stack", icon=str(ICON), template=True, quit_button=None)
        self.c = controller
        self.bench_job: dict = {}
        self._edit_menu_ready = False
        self.rebuild()
        rumps.Timer(self.refresh, 5).start()

    def rebuild(self, _=None) -> None:
        c = self.c
        self.menu.clear()
        self.status = rumps.MenuItem("На паузе" if c.service.paused else "Работает", callback=self.toggle_pause)
        self.status.state = 0 if c.service.paused else 1
        self.today = rumps.MenuItem(c.today_line())
        layers = rumps.MenuItem("Слои")
        for layer_id, title, on in c.layer_items():
            item = rumps.MenuItem(title, callback=lambda _, i=layer_id: self._do(c.toggle_layer, i))
            item.state = int(on)
            layers.add(item)
        models = rumps.MenuItem("Облачные модели")
        for entry in c.cfg.keys:
            sub = rumps.MenuItem(f"{entry.name}  ({entry.hint})")
            listed = c.key_models(entry.name)
            for model_id, on in listed[:60]:
                item = rumps.MenuItem(model_id, callback=lambda _, k=entry.name, m=model_id: self._do(c.toggle_key_model, k, m))
                item.state = int(on)
                sub.add(item)
            if not entry.listable or not listed:
                sub.add(rumps.MenuItem("Вписать модель вручную…", callback=lambda _, k=entry.name: self.manual_model(k)))
            sub.add(rumps.separator)
            sub.add(rumps.MenuItem("Сравнить все модели ключа…", callback=lambda _, k=entry.name: self.compare(k)))
            sub.add(rumps.MenuItem("Обновить список моделей", callback=lambda _, k=entry.name: self._do(c.refresh_key, k)))
            sub.add(rumps.MenuItem("Удалить ключ", callback=lambda _, k=entry.name: self._do(c.remove_key, k)))
            models.add(sub)
        if c.cfg.keys:
            models.add(rumps.separator)
        models.add(rumps.MenuItem("Добавить ключ из буфера обмена", callback=self.add_key_from_clipboard))
        models.add(rumps.MenuItem("Добавить ключ…", callback=self.add_key))
        models.add(rumps.MenuItem("Сравнить включённые модели…", callback=lambda _: self.compare(None)))
        human = rumps.MenuItem(f"Люди: {c.cfg.human_provider}")
        human.add(rumps.MenuItem("Ключ…", callback=self.human_key))
        human.add(rumps.MenuItem("Сервис: RuCaptcha / 2Captcha…", callback=self.human_provider))
        human.add(rumps.MenuItem(f"Лимит в день: ${c.cfg.human_daily_limit_usd:g}…", callback=self.human_limit))
        human.add(rumps.MenuItem("Показать баланс", callback=lambda _: rumps.alert("Люди", c.balance_line())))
        login = rumps.MenuItem("Запускать при входе", callback=self.toggle_login)
        login.state = int(c.starts_at_login())
        self.menu = [
            self.status, self.today, None, layers, models, human, None,
            rumps.MenuItem(f"Адрес для программ: {c.address}", callback=self.copy_address),
            rumps.MenuItem("Статистика в браузере", callback=lambda _: webbrowser.open(f"{c.address}/stats")),
            login, None, rumps.MenuItem("Выход", callback=self.quit),
        ]

    def refresh(self, _=None) -> None:
        if not self._edit_menu_ready:
            install_edit_menu()
            self._edit_menu_ready = True
        self.today.title = self.c.today_line()
        job = self.bench_job
        if job.get("running"):
            self.title = f"{job['done']}/{job['total']}"
        elif job.get("finished"):
            self.bench_job = {}
            self.title = None
            if job.get("error"):
                rumps.alert("Сравнение не удалось", job["error"])
                return
            webbrowser.open(job["report"].as_uri())
            best = job["best"]
            if best and rumps.alert("Сравнение готово", f"Лучшая: {best}. Поставить её первой среди моделей?",
                                    ok="Поставить", cancel="Не надо") == 1:
                self._do(self.c.use_best, best)

    def add_key_from_clipboard(self, _) -> None:
        secret = clipboard_text()
        if not secret or " " in secret or len(secret) < 12:
            rumps.alert("В буфере нет ключа", "Скопируйте ключ API и выберите пункт ещё раз.")
            return
        self._add_key(secret, "")

    def add_key(self, _) -> None:
        secret = ask("Добавить ключ", "Ключ API — поставщик определится сам", secure=True)
        if not secret:
            return
        base = ask("Добавить ключ", "Адрес посредника, если это не сам поставщик (например https://reseller.example/v1). "
                                    "Пусто — определить самому")
        if base is None:
            return
        self._add_key(secret, base)

    def _add_key(self, secret: str, base: str) -> None:
        try:
            entry = self.c.add_key(secret, base)
        except Exception as e:
            rumps.alert("Ключ не подошёл", str(e))
            return
        n = sum(1 for _ in self.c.key_models(entry.name))
        rumps.alert("Ключ добавлен", f"{entry.name}: моделей с картинками — {n}. Отметьте нужные в меню "
                                     "«Облачные модели» или запустите сравнение.")
        self.rebuild()

    def manual_model(self, key_name: str) -> None:
        model_id = ask("Модель", f"Точное название модели у {key_name}")
        if model_id:
            entry = self.c.key(key_name)
            entry.models.append([model_id, True, 0.0, 0.0])
            self._do(self.c.toggle_key_model, key_name, model_id)

    def compare(self, key_name) -> None:
        if self.bench_job.get("running"):
            rumps.alert("Сравнение уже идёт", "Дождитесь конца — прогресс виден рядом со значком.")
            return
        specs = self.c.bench_specs(key_name)
        if not specs:
            rumps.alert("Нечего сравнивать", "Нет моделей с картинками или нет ключа.")
            return
        runs, usd = self.c.bench_estimate(specs)
        price = f"примерно ${usd:.2f}" if usd is not None else "неизвестна — поставщик не публикует цены"
        if rumps.alert("Сравнить модели", f"Моделей: {len(specs)}, решений: {runs}. Цена {price}. Запустить?",
                       ok="Запустить", cancel="Отмена") != 1:
            return
        job = {"running": True, "done": 0, "total": runs}
        self.bench_job = job

        def work():
            try:
                report = self.c.run_bench(specs, progress=lambda d, t: job.update(done=d, total=t))
                import json
                results = json.loads(report.with_suffix(".json").read_text())
                ranked = bench.ranking(results)
                job["best"] = ranked[0] if ranked and results[ranked[0]]["accuracy"] > 0 else None
                job["report"] = report
            except Exception as e:
                job["error"] = str(e)
            job.update(running=False, finished=True)

        threading.Thread(target=work, daemon=True).start()

    def _do(self, fn, *args) -> None:
        try:
            fn(*args)
        except Exception as e:
            rumps.alert("Не получилось", str(e))
        self.rebuild()

    def toggle_pause(self, _) -> None:
        self.c.set_paused(not self.c.service.paused)
        self.rebuild()

    def human_key(self, _) -> None:
        key = ask("Люди", f"Ключ {self.c.cfg.human_provider}", secure=True)
        if key is not None:
            self._do(self.c.set_human, None, key)

    def human_provider(self, _) -> None:
        p = ask("Люди", "rucaptcha или 2captcha", self.c.cfg.human_provider)
        if p is not None:
            self._do(self.c.set_human, p.lower())

    def human_limit(self, _) -> None:
        v = ask("Люди", "Сколько долларов в день можно тратить (0 — без лимита)", f"{self.c.cfg.human_daily_limit_usd:g}")
        if v is not None:
            try:
                limit = float(v.replace(",", "."))
            except ValueError:
                rumps.alert("Не получилось", "Нужно число")
                return
            self._do(self.c.set_human, None, None, limit)

    def toggle_login(self, _) -> None:
        self._do(self.c.set_start_at_login, not self.c.starts_at_login())

    def copy_address(self, _) -> None:
        subprocess.run(["pbcopy"], input=self.c.address.encode(), check=False)
        rumps.notification("captcha-stack", "Адрес скопирован", self.c.address)

    def quit(self, _) -> None:
        self.c.stop_server()
        rumps.quit_application()


def main() -> None:
    controller = Controller()
    controller.start_server()
    MenuBarApp(controller).run()


if __name__ == "__main__":
    main()
