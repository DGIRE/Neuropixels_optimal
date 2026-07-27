"""
np_console.py  --  Neuropixels Data Console (GUI)
=================================================
Run this file to open the console:

    python np_console.py
    python np_console.py  "C:\\path\\to\\np_aggregate.h5"     # custom aggregate

A window opens showing the probe layout for the selected experiment. Choose the
date, LFP frequency band, and time window (trial + within-trial seconds) at the
top, click one or more electrodes on the probe, then press "View" to open the
signal window (ethanol, sniff, filtered LFP + rasters). "Save PNG…" there writes
the figure plus a matching metadata .txt.

Uses only the Python standard library's Tkinter plus numpy / scipy / matplotlib /
h5py (the same packages the aggregate tools use) -- no extra install. Heavy work
(band-pass filtering) is cached per band on the loaded session, and the probe map
is drawn once per date, so interaction stays responsive.
"""
from __future__ import annotations

import os
import sys
import datetime

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
import matplotlib.pyplot as plt

import console_data as CD
import console_figures as CF

DEFAULT_H5 = r"C:\Projects\Repos\Neuropixels\DATA\Aggregate\np_aggregate.h5"
CLICK_TOL_X = 220.0      # µm: max lateral distance from an electrode to select it
CLICK_TOL_Y = 140.0      # µm: max depth distance


class ConsoleApp:
    def __init__(self, root: tk.Tk, h5_path: str):
        self.root = root
        self.h5_path = h5_path
        self.session: CD.Session | None = None
        self.selected: list[int] = []       # electrode indices, click order
        self.fig = self.ax = self.canvas = self.sel_artist = None

        root.title("Neuropixels Data Console")
        root.geometry("760x1000")
        self._build_controls()
        self._build_canvas_area()
        self._load_dates()

    # -- layout ------------------------------------------------------------
    def _build_controls(self):
        bar = ttk.Frame(self.root, padding=6)
        bar.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(bar, text="Date:").grid(row=0, column=0, sticky="w")
        self.date_var = tk.StringVar()
        self.date_cb = ttk.Combobox(bar, textvariable=self.date_var, width=14,
                                    state="readonly")
        self.date_cb.grid(row=0, column=1, padx=4)
        self.date_cb.bind("<<ComboboxSelected>>", lambda e: self.on_date_change())

        ttk.Label(bar, text="LFP band:").grid(row=0, column=2, sticky="w")
        self.band_var = tk.StringVar(value="Raw (broadband)")
        vals = list(CD.BAND_PRESETS.keys()) + ["Custom (Hz fields)"]
        self.band_cb = ttk.Combobox(bar, textvariable=self.band_var, width=18,
                                    values=vals, state="readonly")
        self.band_cb.grid(row=0, column=3, padx=4)

        ttk.Label(bar, text="custom Hz:").grid(row=0, column=4, sticky="w")
        self.flo_var = tk.StringVar(value="30")
        self.fhi_var = tk.StringVar(value="80")
        ttk.Entry(bar, textvariable=self.flo_var, width=5).grid(row=0, column=5)
        ttk.Label(bar, text="-").grid(row=0, column=6)
        ttk.Entry(bar, textvariable=self.fhi_var, width=5).grid(row=0, column=7)

        ttk.Label(bar, text="Trial:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.trial_var = tk.StringVar()
        self.trial_cb = ttk.Combobox(bar, textvariable=self.trial_var, width=14,
                                     state="readonly")
        self.trial_cb.grid(row=1, column=1, padx=4, pady=(6, 0))

        ttk.Label(bar, text="start (s):").grid(row=1, column=2, sticky="w", pady=(6, 0))
        self.t0_var = tk.StringVar(value="0")
        ttk.Entry(bar, textvariable=self.t0_var, width=6).grid(row=1, column=3,
                                                               sticky="w", pady=(6, 0))
        ttk.Label(bar, text="end (s):").grid(row=1, column=4, sticky="w", pady=(6, 0))
        self.t1_var = tk.StringVar(value="5")
        ttk.Entry(bar, textvariable=self.t1_var, width=6).grid(row=1, column=5,
                                                               sticky="w", pady=(6, 0))

        btns = ttk.Frame(self.root, padding=(6, 0))
        btns.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(btns, text="View", command=self.on_view).pack(side=tk.LEFT)
        ttk.Button(btns, text="Select all",
                   command=self.select_all).pack(side=tk.LEFT, padx=6)
        ttk.Button(btns, text="Clear selection",
                   command=self.clear_selection).pack(side=tk.LEFT, padx=6)
        self.status = ttk.Label(btns, text="0 electrodes selected")
        self.status.pack(side=tk.LEFT, padx=12)

    def _build_canvas_area(self):
        self.canvas_frame = ttk.Frame(self.root)
        self.canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # -- data --------------------------------------------------------------
    def _load_dates(self):
        try:
            dates = CD.list_dates(self.h5_path)
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Cannot open aggregate",
                                 "Could not open:\n%s\n\n%s" % (self.h5_path, e))
            dates = []
        self.date_cb["values"] = dates
        if dates:
            self.date_var.set(dates[0])
            self.on_date_change()

    def on_date_change(self):
        date = self.date_var.get()
        if not date:
            return
        try:
            self.session = CD.Session(self.h5_path, date)
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Load error", "Failed to load %s:\n%s" % (date, e))
            return
        self.selected = []
        self._populate_trials()
        self._draw_probe()
        self._update_status()

    def _populate_trials(self):
        trials = self.session.trials() if self.session else []
        if trials:
            self.trial_cb["values"] = [str(t) for t in trials]
            self.trial_var.set(str(trials[0]))
            self.trial_cb.configure(state="readonly")
        else:
            self.trial_cb["values"] = ["(whole recording)"]
            self.trial_var.set("(whole recording)")
            self.trial_cb.configure(state="disabled")

    def _draw_probe(self):
        for w in self.canvas_frame.winfo_children():
            w.destroy()
        if self.fig is not None:
            plt.close(self.fig)
        self.fig, self.ax = CF.build_probe_figure(self.session)
        self.sel_artist = self.ax.scatter([], [], s=175, facecolors="none",
                                          edgecolors="yellow", linewidths=2.0, zorder=8)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
        self.canvas.draw()
        NavigationToolbar2Tk(self.canvas, self.canvas_frame).update()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("button_press_event", self.on_click)

    # -- interaction -------------------------------------------------------
    def on_click(self, event):
        if event.inaxes is not self.ax or self.session is None:
            return
        if event.xdata is None or event.ydata is None:
            return
        # ignore clicks while a pan/zoom tool is active
        tb = self.canvas.toolbar
        if tb is not None and getattr(tb, "mode", ""):
            return
        e = self.session.nearest_electrode(event.xdata, event.ydata)
        if e < 0:
            return
        if (abs(self.session.chan_x[e] - event.xdata) > CLICK_TOL_X or
                abs(self.session.chan_y[e] - event.ydata) > CLICK_TOL_Y):
            return
        if e in self.selected:
            self.selected.remove(e)
        else:
            self.selected.append(e)
        xs, ys = CF.selection_positions(self.session, self.selected)
        self.sel_artist.set_offsets(np.c_[xs, ys] if len(xs) else np.empty((0, 2)))
        self.canvas.draw_idle()
        self._update_status()

    def select_all(self):
        """Select every electrode on the probe."""
        if self.session is None or not self.session.n_elec:
            return
        self.selected = list(range(self.session.n_elec))
        xs, ys = CF.selection_positions(self.session, self.selected)
        if self.sel_artist is not None:
            self.sel_artist.set_offsets(np.c_[xs, ys] if len(xs) else np.empty((0, 2)))
            self.canvas.draw_idle()
        self._update_status()

    def clear_selection(self):
        self.selected = []
        if self.sel_artist is not None:
            self.sel_artist.set_offsets(np.empty((0, 2)))
            self.canvas.draw_idle()
        self._update_status()

    def _update_status(self):
        n = len(self.selected)
        extra = ""
        if self.session and n:
            depths = ", ".join("%.0fµm" % self.session.chan_y[e]
                               for e in self.selected[:6])
            extra = " (%s%s)" % (depths, "…" if n > 6 else "")
        self.status.config(text="%d electrode(s) selected%s" % (n, extra))

    # -- selections -> parameters -----------------------------------------
    def get_band(self):
        name = self.band_var.get()
        if name.startswith("Custom"):
            try:
                lo, hi = float(self.flo_var.get()), float(self.fhi_var.get())
            except ValueError:
                raise ValueError("Custom band needs numeric low/high Hz.")
            if hi <= lo:
                raise ValueError("Custom high Hz must exceed low Hz.")
            return (lo, hi), "Custom (%g-%g Hz)" % (lo, hi)
        return CD.BAND_PRESETS[name], name

    def get_window(self):
        try:
            t0, t1 = float(self.t0_var.get()), float(self.t1_var.get())
        except ValueError:
            raise ValueError("Start/end seconds must be numeric.")
        if t1 <= t0:
            raise ValueError("End must be greater than start.")
        trials = self.session.trials()
        trial = None
        if trials and self.trial_var.get() not in ("", "(whole recording)"):
            trial = int(self.trial_var.get())
        a0, a1 = self.session.trial_window_to_abs(trial, t0, t1)
        a0 = max(0.0, a0)
        a1 = min(a1, self.session.lfp_duration_s) if self.session.lfp_duration_s else a1
        if a1 <= a0:
            raise ValueError("Window falls outside the recording.")
        return (a0, a1), trial, (t0, t1)

    # -- view --------------------------------------------------------------
    def on_view(self):
        if self.session is None:
            return
        if not self.selected:
            messagebox.showinfo("Select electrodes",
                                "Click one or more electrodes on the probe first.")
            return
        try:
            band, band_name = self.get_band()
            (a0, a1), trial, (wt0, wt1) = self.get_window()
        except ValueError as e:
            messagebox.showerror("Invalid input", str(e))
            return
        show_sniff = self.session.sniff_present(a0, a1)
        # trial-relative x-axis: subtract the trial's absolute start so a selected
        # trial reads from its own start (0-based). Whole-recording -> offset 0.
        t_offset = (self.session.trial_window_to_abs(trial, 0.0, 0.0)[0]
                    if trial is not None else 0.0)
        fig, meta = CF.build_view_figure(self.session, self.selected, band,
                                         band_name, a0, a1, show_sniff,
                                         t_offset=t_offset)
        meta.update(trial=trial, within_trial_start_s=wt0, within_trial_end_s=wt1,
                    h5_path=self.h5_path)
        # everything the per-unit detail window needs, captured at View time
        ctx = dict(session=self.session, band=band, band_name=band_name,
                   wt0=wt0, wt1=wt1, trial=trial, show_sniff=show_sniff,
                   h5_path=self.h5_path)
        self._open_view_window(fig, meta, ctx)

    def _open_view_window(self, fig, meta, ctx):
        win = tk.Toplevel(self.root)
        win.title("View — %s — %s" % (meta["date"], meta["band"]))
        win.geometry("1200x820")
        bar = ttk.Frame(win, padding=4)
        bar.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(bar, text="Save PNG…",
                   command=lambda: self.on_save_png(fig, meta)).pack(side=tk.LEFT)
        ttk.Label(bar, text="  %d electrode(s), %d unit(s)   ·   click a unit's "
                  "raster for its per-trial detail"
                  % (len(meta["electrodes"]), len(meta["unit_ids"]))).pack(side=tk.LEFT)
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        NavigationToolbar2Tk(canvas, win).update()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        canvas.mpl_connect("button_press_event",
                           lambda ev: self._on_view_click(ev, canvas, meta, ctx))
        win.protocol("WM_DELETE_WINDOW", lambda: (plt.close(fig), win.destroy()))

    def _on_view_click(self, event, canvas, meta, ctx):
        """A click inside a unit's raster lane opens that unit's detail window."""
        ax = event.inaxes
        if ax is None or event.xdata is None or event.ydata is None:
            return
        tb = getattr(canvas, "toolbar", None)
        if tb is not None and getattr(tb, "mode", ""):
            return  # pan/zoom active
        lanes = meta.get("raster_lanes") or []
        if not lanes:
            return
        d0 = meta["t_start_s"] - meta.get("t_offset_s", 0.0)
        d1 = meta["t_end_s"] - meta.get("t_offset_s", 0.0)
        if not (min(d0, d1) <= event.xdata <= max(d0, d1)):
            return
        best, best_dy = None, None
        for ln in lanes:
            dy = abs(event.ydata - ln["y_center"])
            if dy <= ln["half_height"] and (best_dy is None or dy < best_dy):
                best, best_dy = ln, dy
        if best is not None:
            self._open_unit_window(ctx, int(best["uid"]), int(best["elec"]))

    def _open_unit_window(self, ctx, uid, elec):
        session = ctx["session"]
        trials = session.trials()
        if not trials:
            messagebox.showinfo(
                "No trial structure",
                "This session has no trial (TR) structure, so a per-trial unit "
                "view can't be built. The single-window rasters are in the View "
                "figure.")
            return
        try:
            fig, meta = CF.build_unit_figure(
                session, uid, elec, ctx["band"], ctx["band_name"],
                ctx["wt0"], ctx["wt1"], trials,
                show_sniff=ctx["show_sniff"], h5_path=ctx["h5_path"])
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Unit view failed", str(e))
            return
        win = tk.Toplevel(self.root)
        win.title("Unit %d — %s" % (uid, meta["date"]))
        win.geometry("1000x900")
        bar = ttk.Frame(win, padding=4)
        bar.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(bar, text="Save PNG…",
                   command=lambda: self.on_save_unit_png(fig, meta)).pack(side=tk.LEFT)
        ttk.Label(bar, text="  unit %d · %d trials · Sh%d %.0fµm"
                  % (uid, meta["n_trials_used"], meta["peak_electrode_shank"],
                     meta["peak_electrode_y_um"])).pack(side=tk.LEFT)
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        NavigationToolbar2Tk(canvas, win).update()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        win.protocol("WM_DELETE_WINDOW", lambda: (plt.close(fig), win.destroy()))

    def on_save_png(self, fig, meta):
        folder = filedialog.askdirectory(title="Choose a folder to save the PNG")
        if not folder:
            return
        base = "%s_%s_%.2f-%.2fs" % (
            meta["date"], meta["band"].split(" (")[0].replace(" ", ""),
            meta["t_start_s"], meta["t_end_s"])
        base = "".join(c for c in base if c not in '<>:"/\\|?*')
        png = os.path.join(folder, base + ".png")
        txt = os.path.join(folder, base + ".txt")
        try:
            fig.savefig(png, dpi=150)
            self._write_metadata(txt, meta, png)
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Save failed", str(e))
            return
        messagebox.showinfo("Saved", "Saved:\n%s\n%s" % (png, txt))

    def _write_metadata(self, path, meta, png_path):
        lines = [
            "Neuropixels Data Console — figure metadata",
            "generated: %s" % datetime.datetime.now().isoformat(timespec="seconds"),
            "png: %s" % os.path.basename(png_path),
            "",
            "aggregate: %s" % meta.get("h5_path", ""),
            "date: %s" % meta["date"],
            "experiment_name: %s" % meta.get("experiment_name", ""),
            "LFP band: %s  (Hz=%s)" % (meta["band"], meta.get("band_hz")),
            "LFP fs (Hz): %s" % meta.get("lfp_fs"),
            "trial: %s" % meta.get("trial"),
            "within-trial window (s): %s - %s" % (meta.get("within_trial_start_s"),
                                                  meta.get("within_trial_end_s")),
            "absolute window (s): %.4f - %.4f" % (meta["t_start_s"], meta["t_end_s"]),
            "x-axis: %s (t_offset_s=%.4f)" % (meta.get("x_axis", "absolute seconds"),
                                              meta.get("t_offset_s", 0.0)),
            "ethanol shown: %s" % meta.get("ethanol_shown"),
            "sniff shown: %s" % meta.get("sniff_shown"),
            "",
            "electrodes (index, x_um, y_um, shank):",
        ]
        for e, (x, y, sh) in zip(meta["electrodes"], meta["electrode_positions"]):
            lines.append("  %d\t%.1f\t%.1f\t%d" % (e, x, y, sh))
        lines.append("")
        lines.append("unit IDs shown (%d): %s"
                     % (len(meta["unit_ids"]),
                        ", ".join(str(u) for u in meta["unit_ids"])))
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")

    # -- per-unit detail save ---------------------------------------------
    def on_save_unit_png(self, fig, meta):
        folder = filedialog.askdirectory(title="Choose a folder to save the PNG")
        if not folder:
            return
        base = "%s_unit%d_%s_%.2f-%.2fs" % (
            meta["date"], meta["unit_id"],
            meta["band"].split(" (")[0].replace(" ", ""),
            meta["within_trial_start_s"], meta["within_trial_end_s"])
        base = "".join(c for c in base if c not in '<>:"/\\|?*')
        png = os.path.join(folder, base + ".png")
        txt = os.path.join(folder, base + ".txt")
        try:
            fig.savefig(png, dpi=150)
            self._write_unit_metadata(txt, meta, png)
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Save failed", str(e))
            return
        messagebox.showinfo("Saved", "Saved:\n%s\n%s" % (png, txt))

    def _write_unit_metadata(self, path, meta, png_path):
        used = meta.get("trials_used", [])
        skipped = meta.get("trials_skipped_out_of_range", [])
        lines = [
            "Neuropixels Data Console — per-unit detail metadata",
            "generated: %s" % datetime.datetime.now().isoformat(timespec="seconds"),
            "png: %s" % os.path.basename(png_path),
            "",
            "aggregate: %s" % meta.get("h5_path", ""),
            "date: %s" % meta["date"],
            "experiment_name: %s" % meta.get("experiment_name", ""),
            "unit_id: %d" % meta["unit_id"],
            "peak electrode: index=%d  x_um=%.1f  y_um=%.1f  shank=%d" % (
                meta["peak_electrode"], meta["peak_electrode_x_um"],
                meta["peak_electrode_y_um"], meta["peak_electrode_shank"]),
            "nearest LFP site: row=%s  x_um=%s  y_um=%s" % (
                meta.get("lfp_site_row"), meta.get("lfp_site_x_um"),
                meta.get("lfp_site_y_um")),
            "LFP band: %s  (Hz=%s)" % (meta["band"], meta.get("band_hz")),
            "LFP fs (Hz): %s" % meta.get("lfp_fs"),
            "LabView fs (Hz): %s" % meta.get("LV_Fs"),
            "within-trial window (s): %.4f - %.4f" % (
                meta["within_trial_start_s"], meta["within_trial_end_s"]),
            "x-axis: within-trial seconds (each trial aligned to its own start)",
            "PSTH bin (s): %s" % meta.get("psth_bin_s"),
            "ethanol shown: %s" % meta.get("ethanol_shown"),
            "sniff shown: %s" % meta.get("sniff_shown"),
            "trials used (%d): %s" % (len(used),
                                      ", ".join(str(t) for t in used)),
            "trials skipped, window out of recording (%d): %s" % (
                len(skipped), ", ".join(str(t) for t in skipped)),
        ]
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")


def main():
    h5 = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_H5
    root = tk.Tk()
    try:
        ttk.Style().theme_use("clam")
    except Exception:  # noqa: BLE001
        pass
    ConsoleApp(root, h5)
    root.mainloop()


if __name__ == "__main__":
    main()
