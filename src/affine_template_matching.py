#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Template-based Affine Video Stabiliser (GUI + Thread-Safe Progress + In-place Edit)
================================================================================
*Python 3.11 / OpenCV ≥ 4.9*

Features
--------
1. **Interactive Tk GUI** for picking templates/search windows on arbitrary frames.
2. **Double-click in-place edit** of template rows (cx, cy, tpl, src).
3. **Threaded processing** of matching & affine warp with live progress pushed to a
   ttk.Progressbar (queue-driven; no UI freeze).
4. CLI fallback (`--nogui`) identical to previous versions.

Install: `pip install opencv-python-headless pandas numpy tqdm pillow`
Run GUI: `python affine_template_matching.py --video sample.mp4`
Run CLI: `python affine_template_matching.py --video sample.mp4 --nogui`
"""
from __future__ import annotations

import argparse
import csv
import gc
import queue
import shutil
import threading
from pathlib import Path
from typing import Any, Dict, List, Literal, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# =============================================================================
# Utils
# =============================================================================


def _quad(a, b, c):
    d = a - 2 * b + c
    return 0.0 if d == 0 else 0.5 * (a - c) / d


def refine(mat: np.ndarray, loc: Sequence[int]) -> Tuple[float, float]:
    x, y = loc[0], loc[1]
    if 0 < x < mat.shape[1] - 1 and 0 < y < mat.shape[0] - 1:
        return _quad(mat[y, x - 1], mat[y, x], mat[y, x + 1]), _quad(mat[y - 1, x], mat[y, x],
                                                                     mat[y + 1, x])
    return 0., 0.


def crop(src: np.ndarray, cx: int, cy: int, h: int):
    x0, y0, cx, cy = cx - h, cy - h, cx, cy
    x1, y1 = cx + h, cy + h
    H, W = src.shape[:2]
    pl, pt = max(0, -x0), max(0, -y0)
    pr, pb = max(0, x1 - W), max(0, y1 - H)
    roi = src[max(0, y0):min(H, y1), max(0, x0):min(W, x1)]
    if any((pl, pt, pr, pb)):
        roi = cv2.copyMakeBorder(roi, pt, pb, pl, pr, cv2.BORDER_CONSTANT)
    return roi


# Crop rectangular ROI with half-width hw and half-height hh; pad with zeros if crossing border.
def crop_rect(src: np.ndarray, cx: int, cy: int, hw: int, hh: int):
    """Crop rectangular ROI with half‑width hw and half‑height hh; pad with zeros if crossing border."""
    x0, y0 = cx - hw, cy - hh
    x1, y1 = cx + hw, cy + hh
    H, W = src.shape[:2]
    pl, pt = max(0, -x0), max(0, -y0)
    pr, pb = max(0, x1 - W), max(0, y1 - H)
    roi = src[max(0, y0):min(H, y1), max(0, x0):min(W, x1)]
    if any((pl, pt, pr, pb)):
        roi = cv2.copyMakeBorder(roi, pt, pb, pl, pr, cv2.BORDER_CONSTANT)
    return roi


# Crop a rectangle using individual pixel margins measured from the centre.
def crop_lrbt(src: np.ndarray, cx: int, cy: int, left: int, right: int, top: int,
              bottom: int) -> np.ndarray:
    """
    Crop a rectangle using individual pixel margins measured from the centre:
    left, right, top, bottom (all >=0). Pads with zeros where the window
    crosses image borders.
    """
    x0, y0 = cx - left, cy - top
    x1, y1 = cx + right, cy + bottom
    H, W = src.shape[:2]
    pl, pt = max(0, -x0), max(0, -y0)
    pr, pb = max(0, x1 - W), max(0, y1 - H)
    roi = src[max(0, y0):min(H, y1), max(0, x0):min(W, x1)]
    if any((pl, pt, pr, pb)):
        roi = cv2.copyMakeBorder(roi, pt, pb, pl, pr, cv2.BORDER_CONSTANT)
    return roi


# =============================================================================
# Matcher
# =============================================================================
class TemplateMatcher:

    def __init__(self, video: str, csv_path: str):
        self.cap = cv2.VideoCapture(video, apiPreference=cv2.CAP_FFMPEG)
        ok, first = self.cap.read()
        assert ok, 'cannot read video'
        self.first = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
        self.templates = self._load(csv_path)

    def _load(self, p: str):
        df = pd.read_csv(p)
        out = []
        for _, r in df.iterrows():
            half_tpl = int(r['テンプレートの大きさ（正方形）']) // 2
            top = int(r['探索上'])
            bottom = int(r['探索下'])
            left = int(r['探索左'])
            right = int(r['探索右'])
            out.append(
                dict(id=int(r['No.']),
                     cx=int(r['x座標']),
                     cy=int(r['y座標']),
                     th=half_tpl,
                     t=top,
                     b=bottom,
                     l=left,
                     r=right,
                     patch=crop(self.first, int(r['x座標']), int(r['y座標']), half_tpl)))
        return out

    def _match(self, g: np.ndarray, t: Dict[str, Any]):
        roi = crop_lrbt(g, t['cx'], t['cy'], t['l'], t['r'], t['t'], t['b'])
        res = cv2.matchTemplate(roi, t['patch'], cv2.TM_CCORR_NORMED)
        *_, loc = cv2.minMaxLoc(res)
        dx, dy = refine(res, tuple(loc))
        mx = t['cx'] - t['l'] + loc[0] + dx + t['th']
        my = t['cy'] - t['t'] + loc[1] + dy + t['th']
        return mx, my

    def run(self, out_csv='Matched.csv', q: queue.Queue | None = None):
        rec = []
        idx = 1
        total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
        if q:
            q.put(('setmax', total))
            prog = 0
        with tqdm(total=total, desc='match') as bar:
            while True:
                ok, f = self.cap.read()
                if not ok:
                    break
                idx += 1
                g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                for t in self.templates:
                    mx, my = self._match(g, t)
                    rec.append([idx, t['id'], t['cx'], t['cy'], mx, my])
                bar.update(1)
                prog += 1
                if q:
                    q.put(('prog', prog))
        self.cap.release()
        pd.DataFrame(rec, columns=['Frame_No', 'Template_No', 'x', 'y', 'mx',
                                   'my']).to_csv(out_csv, index=False)
        del self.first
        gc.collect()


# =============================================================================
# Affine
# =============================================================================
class AffineCorrector:

    def __init__(self, video: str, csv: str, outv='Affined.mp4', tmp='temp'):
        self.matches = pd.read_csv(csv)
        self.video = Path(video)
        self.out = Path(outv)
        self.tmp = Path(tmp)
        self.img = self.tmp / 'images'
        self.conv = self.tmp / 'converted'
        self.img.mkdir(parents=True, exist_ok=True)
        self.conv.mkdir(parents=True, exist_ok=True)

    def split(self):
        c = cv2.VideoCapture(str(self.video), apiPreference=cv2.CAP_FFMPEG)
        fps = c.get(cv2.CAP_PROP_FPS)
        w, h = int(c.get(3)), int(c.get(4))
        idx = 1
        while True:
            ok, f = c.read()
            if not ok:
                break
            cv2.imwrite(str(self.img / f'frame_{idx:06d}.png'), f)
            idx += 1
        c.release()
        return w, h, fps

    def _M(self, n: int):
        df = self.matches[self.matches.Frame_No == n]
        src = df[['mx', 'my']].to_numpy(np.float32)
        dst = df[['x', 'y']].to_numpy(np.float32)
        M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=2)
        if M is None:
            raise RuntimeError('affine fail')
        return M

    def correct(self, q: queue.Queue | None = None):
        w, h, fps = self.split()
        frames = sorted(self.img.glob('frame_*.png'))
        if q:
            q.put(('setmax', len(frames) - 1))
            done = 0
        fourcc_fn = cv2.VideoWriter_fourcc if hasattr(
            cv2, "VideoWriter_fourcc") else cv2.VideoWriter.fourcc
        out = cv2.VideoWriter(str(self.out), fourcc_fn(*'mp4v'), fps if fps > 0 else 30, (w, h))
        for fp in tqdm(frames[1:], desc='warp'):
            n = int(fp.stem.split('_')[1])
            img = cv2.imread(str(fp))
            aff = cv2.warpAffine(img,
                                 self._M(n), (w, h),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT)
            out.write(aff)
            cv2.imwrite(str(self.conv / f'conv_{n:06d}.png'), aff)
            done += 1
            q.put(('prog', done)) if q else None
        out.release()

    def cleanup(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


# =============================================================================
# GUI
# =============================================================================
class TemplateGUI:

    def __init__(self, video: str, csv_p='Input.csv'):
        import tkinter as tk
        from tkinter import messagebox, simpledialog, ttk

        from PIL import Image, ImageTk

        # For Pillow ≥10 compatibility: Resampling alias fallback
        try:
            ResamplingFilter = Image.Resampling.LANCZOS
        except AttributeError:
            ResamplingFilter = Image.LANCZOS
        self.tk = tk
        self.ttk = ttk
        self.messagebox = messagebox
        self.simpledialog = simpledialog
        self.video = video
        self.csv = Path(csv_p)
        self.cap = cv2.VideoCapture(video, apiPreference=cv2.CAP_FFMPEG)
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.tpls: List[Dict[str, Any]] = []
        self.hist: List[Dict[str, Any]] = []
        self.zoom = 1.0
        self.root = tk.Tk()
        self.root.title('Template Designer')
        self._build_widgets()
        self._load_csv()
        self._show(1)
        self.root.mainloop()

    # -------- widgets --------
    def _build_widgets(self):
        ttk = self.ttk
        ctl = ttk.Frame(self.root)
        ctl.pack(fill='x')
        ttk.Label(ctl, text='Template').pack(side='left')
        # Replace Spinbox with Entry for template size
        self.ent_tpl = ttk.Entry(ctl, width=6)
        self.ent_tpl.insert(0, "64")
        self.ent_tpl.pack(side='left')
        # Replace Spinboxes for search margins with Entry widgets
        ttk.Label(ctl, text='SearchLeft').pack(side='left')
        self.ent_left = ttk.Entry(ctl, width=6)
        self.ent_left.insert(0, "128")
        self.ent_left.pack(side='left')
        ttk.Label(ctl, text='SearchRight').pack(side='left')
        self.ent_right = ttk.Entry(ctl, width=6)
        self.ent_right.insert(0, "128")
        self.ent_right.pack(side='left')
        ttk.Label(ctl, text='SearchTop').pack(side='left')
        self.ent_top = ttk.Entry(ctl, width=6)
        self.ent_top.insert(0, "128")
        self.ent_top.pack(side='left')
        ttk.Label(ctl, text='SearchBottom').pack(side='left')
        self.ent_bottom = ttk.Entry(ctl, width=6)
        self.ent_bottom.insert(0, "128")
        self.ent_bottom.pack(side='left')
        # Buttons
        self.btn_delete = ttk.Button(ctl, text='Delete', command=self._del)
        self.btn_delete.pack(side='right')
        self.btn_undo = ttk.Button(ctl, text='Undo', command=self._undo)
        self.btn_undo.pack(side='right')
        self.btn_execute = ttk.Button(ctl, text='Execute', command=self._exec)
        self.btn_execute.pack(side='right')
        # Zoom in/out buttons
        self.btn_zoom_in = ttk.Button(ctl, text='+', command=self._zoom_in)
        self.btn_zoom_in.pack(side='right')
        self.btn_zoom_out = ttk.Button(ctl, text='-', command=self._zoom_out)
        self.btn_zoom_out.pack(side='right')
        # tree
        self.tree = ttk.Treeview(self.root,
                                 columns=('no', 'cx', 'cy', 'tpl', 'l', 'r', 't', 'b'),
                                 show='headings',
                                 height=4)
        for col, text in [
            ('no', 'No.'),
            ('cx', 'x座標'),
            ('cy', 'y座標'),
            ('tpl', 'テンプレートサイズ'),
            ('l', '探索左'),
            ('r', '探索右'),
            ('t', '探索上'),
            ('b', '探索下'),
        ]:
            self.tree.heading(col, text=text)
        self.tree.bind('<Double-1>', self._edit)
        self.tree.pack(fill='x')
        # progress bar
        self.pb = ttk.Progressbar(self.root, orient='horizontal', length=300, mode='determinate')
        self.pb.pack(fill='x', padx=4, pady=4)
        # slider & canvas
        self.scale = ttk.Scale(self.root,
                               from_=1,
                               to=self.total,
                               orient='horizontal',
                               command=self._seek)
        self.scale.pack(fill='x')
        self.canvas = self.tk.Canvas(self.root)
        self.canvas.pack()
        # Status label below canvas
        self.status = self.ttk.Label(self.root, text="x: -, y: -")
        self.status.pack(fill='x', pady=2)
        self.canvas.bind('<Button-1>', self._click)
        self.canvas.bind('<Motion>', self._on_motion)
        self._photo = None

    # -------- data io --------
    def _load_csv(self):
        if not self.csv.exists():
            return
        df = pd.read_csv(self.csv)
        self.tpls.clear()
        self.tree.delete(*self.tree.get_children())
        for _, r in df.iterrows():
            self._add_tpl(int(r['x座標']),
                          int(r['y座標']),
                          int(r['テンプレートの大きさ（正方形）']),
                          int(r['探索左']),
                          int(r['探索右']),
                          int(r['探索上']),
                          int(r['探索下']),
                          existing=True,
                          id=int(r['No.']))

    def _add_tpl(self,
                 cx: int,
                 cy: int,
                 tpl: int,
                 left: int,
                 right: int,
                 top: int,
                 bottom: int,
                 existing=False,
                 id: int | None = None):
        if not existing:
            id = len(self.tpls) + 1
        d = dict(id=id, cx=cx, cy=cy, tpl=tpl, l=left, r=right, t=top, b=bottom)
        self.tpls.append(d)
        self.tree.insert('',
                         'end',
                         iid=d['id'],
                         values=(d['id'], cx, cy, tpl, left, right, top, bottom))

    # -------- UI actions --------
    def _show(self, idx: int):
        from PIL import Image, ImageTk

        # For Pillow ≥10 compatibility: Resampling alias fallback
        try:
            ResamplingFilter = Image.Resampling.LANCZOS
        except AttributeError:
            ResamplingFilter = Image.LANCZOS
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx - 1)
        ok, frame = self.cap.read()
        if not ok:
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        for t in self.tpls:
            cx, cy = t['cx'], t['cy']
            ht = t['tpl'] // 2
            hl = t['l']
            hr = t['r']
            ht_margin = t['t']
            hb = t['b']
            # Draw template rectangle (red)
            cv2.rectangle(rgb, (cx - ht, cy - ht), (cx + ht, cy + ht), (0, 0, 255), 2)
            # Draw index label above the template rectangle
            cv2.putText(rgb, str(t['id']), (cx - ht, cy - ht - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 255), 2, cv2.LINE_AA)
            # Draw ROI rectangle (blue)
            cv2.rectangle(rgb, (cx - hl, cy - ht_margin), (cx + hr, cy + hb), (255, 0, 0), 1)
        img = Image.fromarray(rgb)
        # Apply zoom if not 1.0
        if self.zoom != 1.0:
            w, h = img.size
            img = img.resize((int(w * self.zoom), int(h * self.zoom)), ResamplingFilter)
        self._photo = ImageTk.PhotoImage(img)
        self.canvas.config(width=img.width, height=img.height)
        self.canvas.create_image(0, 0, anchor='nw', image=self._photo)

    def _seek(self, val):
        self._show(int(float(val)))

    def _click(self, e):
        tpl = int(self.ent_tpl.get()) + int(self.ent_tpl.get()) % 2
        left = int(self.ent_left.get())
        right = int(self.ent_right.get())
        top = int(self.ent_top.get())
        bottom = int(self.ent_bottom.get())
        # Map canvas (zoomed) coordinates back to original image coordinates
        orig_x = int(e.x / self.zoom)
        orig_y = int(e.y / self.zoom)
        self._add_tpl(orig_x, orig_y, tpl, left, right, top, bottom)
        self._show(int(self.scale.get()))

    def _del(self):
        iid = self.tree.focus()
        if not iid:
            return
        self.hist.append(next(t for t in self.tpls if t['id'] == int(iid)))
        self.tpls = [t for t in self.tpls if t['id'] != int(iid)]
        self.tree.delete(iid)
        self._show(int(self.scale.get()))

    def _undo(self):
        if not self.hist:
            return
        t = self.hist.pop()
        self.tpls.append(t)
        self.tree.insert('',
                         'end',
                         iid=t['id'],
                         values=(t['cx'], t['cy'], t['tpl'], t['l'], t['r'], t['t'], t['b']))
        self._show(int(self.scale.get()))

    def _edit(self, evt):
        iid = self.tree.focus()
        if not iid:
            return
        row = next(t for t in self.tpls if t['id'] == int(iid))
        for key in ('cx', 'cy', 'tpl', 'l', 'r', 't', 'b'):
            val = self.simpledialog.askinteger('Edit',
                                               f'{key} value',
                                               initialvalue=row[key],
                                               parent=self.root,
                                               minvalue=0)
            if val is None:
                return
            row[key] = val
        self.tree.item(iid,
                       values=(row['cx'], row['cy'], row['tpl'], row['l'], row['r'], row['t'],
                               row['b']))
        self._show(int(self.scale.get()))

    def _on_motion(self, event):
        # Map canvas (zoomed) coords to original
        x = int(event.x / self.zoom)
        y = int(event.y / self.zoom)
        self.status.config(text=f"x: {x}, y: {y}")

    def _zoom_in(self):
        self.zoom *= 1.2
        self._show(int(float(self.scale.get())))

    def _zoom_out(self):
        self.zoom /= 1.2
        self._show(int(float(self.scale.get())))

    # -------- execute --------
    def _exec(self):
        if len(self.tpls) < 3:
            self.messagebox.showwarning('warn', 'テンプレートは3点以上必要')
            return
        with open(self.csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['No.', 'x座標', 'y座標', 'テンプレートの大きさ（正方形）', '探索左', '探索右', '探索上', '探索下'])
            [
                w.writerow([t['id'], t['cx'], t['cy'], t['tpl'], t['l'], t['r'], t['t'], t['b']])
                for t in self.tpls
            ]
        self.btn_delete.config(state='disabled')
        self.btn_undo.config(state='disabled')
        self.btn_execute.config(state='disabled')
        q: queue.Queue[Tuple[str, Any]] = queue.Queue()

        def task():
            try:
                TemplateMatcher(self.video, str(self.csv)).run('Matched.csv', q)
                AffineCorrector(self.video, 'Matched.csv').correct(q)
                AffineCorrector(self.video, 'Matched.csv').cleanup()
                q.put(('done', None))
            except Exception as e:
                q.put(('err', e))

        threading.Thread(target=task, daemon=True).start()
        self._poll(q)

    def _poll(self, q: queue.Queue[Tuple[str, Any]]):
        try:
            typ, *rest = q.get_nowait()
            if typ == 'setmax':
                self.pb['maximum'] = rest[0]
                self.pb['value'] = 0
            elif typ == 'prog':
                self.pb['value'] = rest[0]
            elif typ == 'done':
                self.btn_delete.config(state='normal')
                self.btn_undo.config(state='normal')
                self.btn_execute.config(state='normal')
                self.messagebox.showinfo('完了', '補正完了')
                self.root.destroy()
                return
            elif typ == 'err':
                self.btn_delete.config(state='normal')
                self.btn_undo.config(state='normal')
                self.btn_execute.config(state='normal')
                self.messagebox.showerror('Error', str(rest[0]))
                self.root.destroy()
                return
        except queue.Empty:
            pass
        self.root.after(100, self._poll, q)


# =============================================================================
# main
# =============================================================================


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--video', required=True)
    p.add_argument('--csv', default='Input.csv')
    p.add_argument('--nogui', action='store_true')
    a = p.parse_args()
    if a.nogui:
        TemplateMatcher(a.video, a.csv).run('Matched.csv')
        AffineCorrector(a.video, 'Matched.csv').correct()
        AffineCorrector(a.video, 'Matched.csv').cleanup()
    else:
        TemplateGUI(a.video, a.csv)


if __name__ == '__main__':
    main()
