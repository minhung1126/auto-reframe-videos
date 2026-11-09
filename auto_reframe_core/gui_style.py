"""Shared desktop presentation helpers."""

import tkinter as tk
from tkinter import font as tkfont, ttk
from auto_reframe_core.gui_constants import CJK_EDITOR_FONT_CANDIDATES
from auto_reframe_core.platform_profile import current_platform


def create_gui_root(*, require_dnd=False):
    """Keep one Tk interpreter even when the optional drag/drop extension fails."""
    from auto_reframe_core.platform_profile import initialize_gui_platform

    initialize_gui_platform()
    try:
        from tkinterdnd2 import TkinterDnD
    except ImportError:
        if require_dnd:
            raise
        return tk.Tk()

    class Root(tk.Tk, TkinterDnD.DnDWrapper):
        pass

    root = Root()
    try:
        root.TkdndVersion = TkinterDnD._require(root)
    except (tk.TclError, RuntimeError):
        if require_dnd:
            root.destroy()
            raise
    return root


def calculate_responsive_window_geometry(
    screen_width: int,
    screen_height: int,
) -> tuple[int, int, int, int]:
    """Return a usable initial and minimum size for desktop and small displays."""

    width = min(1100, max(640, screen_width - 96), screen_width)
    height = min(900, max(520, screen_height - 120), screen_height)
    min_width = min(820, width)
    min_height = min(620, height)
    return width, height, min_width, min_height


def copy_text_to_clipboard(root, text: str) -> None:
    """Copy text through Tk so the shortcut works on Windows and macOS."""
    root.clipboard_clear()
    root.clipboard_append(text)
    root.update_idletasks()


def choose_cjk_editor_font_family(available_families) -> str:
    """Choose a readable installed CJK family for the GUI text editors."""

    installed = {}
    for family in available_families:
        value = str(family)
        installed.setdefault(value.casefold(), value)
    for candidate in CJK_EDITOR_FONT_CANDIDATES:
        if candidate.casefold() in installed:
            return installed[candidate.casefold()]
    return "TkTextFont"


def build_cjk_editor_font(root):
    """Return an independent Tk text font with a CJK-capable family when available."""

    editor_font = tkfont.nametofont("TkTextFont", root).copy()
    family = choose_cjk_editor_font_family(tkfont.families(root))
    if family != "TkTextFont":
        editor_font.configure(family=family)
    return editor_font


class ScrollableTab(ttk.Frame):
    """A notebook tab whose content remains reachable in short windows."""

    def __init__(self, parent, *, padding=0):
        super().__init__(parent)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(
            self, orient="vertical", command=self.canvas.yview
        )
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.scrollbar.grid(row=0, column=1, sticky="ns")

        self.content = ttk.Frame(self.canvas, padding=padding)
        self._content_window = self.canvas.create_window(
            (0, 0), window=self.content, anchor="nw"
        )
        self.content.bind("<Configure>", self._update_scroll_region)
        self.canvas.bind("<Configure>", self._resize_content)
        top = self.winfo_toplevel()
        self._wheel_bindings = [
            (sequence, top.bind(sequence, self._wheel, add="+"))
            for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>")
        ]
        self.bind("<Destroy>", self._remove_wheel_bindings, add="+")

    def _update_scroll_region(self, _event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _resize_content(self, event):
        self.canvas.itemconfigure(self._content_window, width=event.width)

    def _wheel(self, event):
        widget = event.widget
        if widget.winfo_class() in {
            "Text",
            "Listbox",
            "Treeview",
            "TCombobox",
            "Scrollbar",
            "TScrollbar",
        }:
            return
        ancestor = widget
        while ancestor is not None and ancestor != self:
            ancestor = getattr(ancestor, "master", None)
        if ancestor != self:
            return
        units = current_platform().wheel_units(
            getattr(event, "delta", 0), getattr(event, "num", None)
        )
        if units:
            self.canvas.yview_scroll(units, "units")
            return "break"

    def _remove_wheel_bindings(self, event):
        if event.widget == self:
            top = self.winfo_toplevel()
            for sequence, binding in self._wheel_bindings:
                top.unbind(sequence, binding)


def configure_styles(root, style):
    """Let Tk/native DPI choose pixel sizes; share fonts and consistent spacing."""
    style.configure("TButton", padding=(8, 5))
    style.configure("TCheckbutton", padding=(3, 4))
    style.configure("TRadiobutton", padding=(3, 4))
    default_font = tkfont.nametofont("TkDefaultFont", root)
    style.configure(
        "Treeview", rowheight=max(24, default_font.metrics("linespace") + 10)
    )


def install_shortcuts(app):
    modifier = current_platform().shortcut_modifier
    for key, callback in (
        ("o", app._add_videos),
        ("s", app.save_settings),
        ("Return", app.start_job),
        ("p", app.open_preview),
    ):

        def invoke(_event, callback=callback):
            callback()
            return "break"

        app.root.bind(f"<{modifier}-{key}>", invoke, add="+")
    app.files_tree.bind("<Delete>", lambda _: app._remove_videos())
    app.root.bind("<Escape>", lambda _: app.stop_job(), add="+")
