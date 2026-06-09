#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_wake.py  —  wake_integrator.py の出力(中間 parquet/CSV)から作図する CLI

計算とは独立に走るので、重い畳み込みを再計算せずに何度でも描き直せる。

使い方
------
    python plot_wake.py --dir "0_integrated wake/version2.2/2021c_physics"

主な出力（旧ノートブックのプロットに対応）
* fig_sz0.5/Wz_total_s.png, Wx_total_s.png, Wy_total_s.png : 総和 wake
* fig_sz0.5/kz_each.png 等                                  : 成分別 loss/kick ランキング
* fig/conv_*.png                                            : バンチ長依存サマリ
各 sigma_z の畳み込み wake は intermediate/conv_sz*/ に保存されたものを使用。
"""

import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')                 # ヘッドレス環境用（画面表示しない）
import matplotlib.pyplot as plt
from scipy.constants import c

plt.rcParams.update({
    # seaborn 風スタイルは廃止。素の matplotlib をベースに最小限だけ調整。
    'figure.facecolor': 'white',
    'axes.facecolor':   'white',
    'axes.grid':        True,
    'grid.linestyle':   '--',
    'grid.linewidth':   0.5,
    'grid.alpha':       0.4,
    'axes.edgecolor':   'black',
    'font.family':      'sans-serif',
    'font.size':        12.0,
    'lines.linewidth':  1.5,
})

SAVE = dict(bbox_inches='tight', pad_inches=0.05, dpi=300,
            pil_kwargs={'compress_level': 1})


def _load_labels(interm):
    p = os.path.join(interm, 'labels.json')
    if os.path.exists(p):
        with open(p, encoding='utf-8') as f:
            return json.load(f)
    return {}


def plot_totals(dir_name, sz_tag):
    """総和 wake (sum 列) を時間軸でプロット。"""
    interm = os.path.join(dir_name, 'intermediate')
    figdir = os.path.join(dir_name, f'fig_sz{sz_tag}')
    os.makedirs(figdir, exist_ok=True)
    specs = [('wakez', r'$W_\mathrm{z}$ [V/pC]', 'Wz_total_s.png', 'Longitudinal'),
             ('wakeDx', r'$W_\mathrm{x,D}$ [V/pC]', 'Wx_total_s.png', 'Horizontal Dipolar'),
             ('wakeDy', r'$W_\mathrm{y,D}$ [V/pC]', 'Wy_total_s.png', 'Vertical Dipolar')]
    for attr, ylabel, fname, title in specs:
        path = os.path.join(interm, f'{attr}.parquet')
        if not os.path.exists(path):
            continue
        df = pd.read_parquet(path)
        if 'sum' not in df.columns:
            continue
        # ±1e5 の境界点を除いた物理領域だけ描画
        phys = df[(df.index > -1e4) & (df.index < 1e4)]
        plt.close('all')
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(phys.index, phys['sum'], '-')
        ax.set_xlabel('$t$ [ns]'); ax.set_ylabel(ylabel)
        ax.set_title(f'Total {title} Wake')
        fig.savefig(os.path.join(figdir, fname), **SAVE)
        plt.close(fig)


def plot_rankings(dir_name, sz_tag):
    """成分別 loss/kick factor のランキング棒グラフ。"""
    out = os.path.join(dir_name, f'out_sz{sz_tag}')
    figdir = os.path.join(dir_name, f'fig_sz{sz_tag}')
    os.makedirs(figdir, exist_ok=True)
    specs = [('loss_factor.csv', 'Loss Factor [V/pC]', 'kz_each.png'),
             ('kick_factorDx.csv', r'$\beta_x k_{x,D}$ [V/pC]', 'kxd_each.png'),
             ('kick_factorDy.csv', r'$\beta_y k_{y,D}$ [V/pC]', 'kyd_each.png'),
             ('kick_factorQx.csv', r'$\beta_x k_{x,Q}$ [V/pC]', 'kxq_each.png'),
             ('kick_factorQy.csv', r'$\beta_y k_{y,Q}$ [V/pC]', 'kyq_each.png')]
    for fname, ylabel, outname in specs:
        path = os.path.join(out, fname)
        if not os.path.exists(path):
            continue
        s = pd.read_csv(path, index_col=0).iloc[0].dropna().sort_values(ascending=False)
        plt.close('all')
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.bar(range(len(s)), s.values)
        ax.set_xticks(range(len(s)))
        ax.set_xticklabels(s.index, rotation=90, fontsize=7)
        ax.set_ylabel(ylabel)
        ax.set_title(f'{ylabel}  (Σ = {s.sum():.2e})')
        fig.savefig(os.path.join(figdir, outname), **SAVE)
        plt.close(fig)


def plot_conv_summary(dir_name):
    """バンチ長依存（conv_summary.csv）の散布図。"""
    path = os.path.join(dir_name, 'out', 'conv_summary.csv')
    if not os.path.exists(path):
        return
    df = pd.read_csv(path, index_col=0)
    figdir = os.path.join(dir_name, 'fig')
    os.makedirs(figdir, exist_ok=True)
    for col, ylabel, fname in [('loss', 'Total Loss Factor [V/pC]', 'conv_loss.png'),
                               ('kick_x_d', r'Total $\beta_x k_{x,D}$ [V/pC]', 'conv_kxd.png'),
                               ('kick_y_d', r'Total $\beta_y k_{y,D}$ [V/pC]', 'conv_kyd.png')]:
        if col not in df.columns:
            continue
        plt.close('all')
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(df.index, df[col], 'o')
        ax.set_xlabel(r'$\sigma_z$ [mm]'); ax.set_ylabel(ylabel)
        ax.set_title(ylabel + r' vs $\sigma_z$')
        fig.savefig(os.path.join(figdir, fname), **SAVE)
        plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description='wake_integrator の出力から作図')
    ap.add_argument('--dir', required=True, help='対象の出力ディレクトリ')
    args = ap.parse_args(argv)
    dir_name = args.dir

    # sz_tag（生データの sigma_z）を out_sz*/ から推定（conv 用 sz は別途）
    sz_dirs = sorted(glob.glob(os.path.join(dir_name, 'out_sz*')))
    sz_tags = [re.search(r'out_sz([0-9.]+)$', d).group(1) for d in sz_dirs
               if re.search(r'out_sz([0-9.]+)$', d)]
    for tag in sz_tags:
        plot_rankings(dir_name, tag)
    # 総和 wake は生データ sz の中間ファイルから
    plot_totals(dir_name, sz_tags[0] if sz_tags else '0.5')
    plot_conv_summary(dir_name)
    print(f'[完了] 作図: {dir_name}')


if __name__ == '__main__':
    main()
