import subprocess
import webbrowser

import rumps

from .control import Controller
from .providers import PRESETS

PROVIDER_HINT = "Поставщик: " + ", ".join(PRESETS)


def ask(title: str, message: str, default: str = "", secure: bool = False) -> str | None:
    w = rumps.Window(message=message, title=title, default_text=default, ok="Сохранить", cancel="Отмена",
                     dimensions=(320, 24), secure=secure)
    r = w.run()
    return r.text.strip() if r.clicked else None


class MenuBarApp(rumps.App):
    def __init__(self, controller: Controller):
        super().__init__("captcha-stack", title="🧩", quit_button=None)
        self.c = controller
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
        for name, line in c.model_lines():
            sub = rumps.MenuItem(line)
            sub.add(rumps.MenuItem("Сменить ключ…", callback=lambda _, n=name: self.change_key(n)))
            sub.add(rumps.MenuItem("Удалить", callback=lambda _, n=name: self._do(c.remove_model, n)))
            models.add(sub)
        models.add(rumps.separator)
        models.add(rumps.MenuItem("Добавить модель…", callback=self.add_model))
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
        self.today.title = self.c.today_line()

    def _do(self, fn, *args) -> None:
        try:
            fn(*args)
        except Exception as e:
            rumps.alert("Не получилось", str(e))
        self.rebuild()

    def toggle_pause(self, _) -> None:
        self.c.set_paused(not self.c.service.paused)
        self.rebuild()

    def add_model(self, _) -> None:
        provider = ask("Новая модель", PROVIDER_HINT, "dashscope")
        if provider is None:
            return
        default_model = PRESETS.get(provider, {}).get("model", "")
        model = ask("Новая модель", f"Название модели у {provider} (например qwen3-vl-plus, gemini-2.5-flash)", default_model)
        if model is None:
            return
        key = "" if PRESETS.get(provider, {}).get("keyless") else ask("Новая модель", f"Ключ API {provider}", secure=True)
        if key is None:
            return
        base_url = ask("Новая модель", "Адрес API (для custom; иначе оставьте пустым)") if provider == "custom" else ""
        if base_url is None:
            return
        name = ask("Новая модель", "Короткое имя в меню", provider)
        if name is None:
            return
        self._do(self.c.add_model, name, provider, model, key, base_url)

    def change_key(self, name: str) -> None:
        key = ask(f"Ключ для {name}", "Новый ключ API", secure=True)
        if key is not None:
            self._do(self.c.set_model_key, name, key)

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
