#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
wake_integrator.py  —  LER/HER 統合 wake インテグレータ（計算専用 CLI）

旧 LER_wake_integrator.ipynb / HER_wake_integrator.ipynb を 1 本に統合し、
コマンドラインから実行できるようにしたもの。

主な変更点
----------
* リング(LER/HER)はパラメータファイルの成分名／親ディレクトリ名から自動判定。
* コリメータは comp_list の directory でグループ化し、aperture の要素数で
  両ジョー(generate_clm_wake) / 片ジョー(generate_kclm_wake) を自動振り分け。
  → vcL10_list / kvc_list 等の手動リストは不要。
* プロットは含まない。計算結果(CSV)に加え、各成分の wake を parquet に保存し、
  plot_wake.py で後から作図できるようにした。
* グローバル変数による状態共有をやめ、State オブジェクトに集約。

使い方
------
    python wake_integrator.py --param 2021c_physics
    python wake_integrator.py --param 2021c_physics --conv-sigmaz 6e-3 9e-3 4
    python wake_integrator.py --param 2021c_physics --plot      # 計算後に作図も実行

旧ノートブックと同じ作業ディレクトリ（1_parameters/, 0_integrated wake/ がある場所）
で実行することを想定している。
"""

import argparse
import os
import re
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.constants import c
from decimal import Decimal, ROUND_HALF_UP

# =====================================================================
# 物理関数（バンチ分布と Green 関数フィット用）
# =====================================================================
def gauss(x, sigma):
    return np.exp(-x**2 / (2 * sigma * sigma)) / (sigma * np.sqrt(2 * np.pi))

def cgauss(x, sigma, mu):
    return np.exp(-(x - mu)**2 / (2 * sigma * sigma)) / (sigma * np.sqrt(2 * np.pi))

def p_gauss(x, sigma):
    return -x * np.exp(-x**2 / (2 * sigma * sigma)) / (np.sqrt(2 * np.pi) * sigma**3)

def cp_gauss(x, sigma, mu):
    return -(x - mu) * np.exp(-(x - mu)**2 / (2 * sigma * sigma)) / (np.sqrt(2 * np.pi) * sigma**3)

def convert_to_length(time):
    """ns → m"""
    return time * c * 1e-9

# sigmaz を引数化した Green 関数（curve_fit 用に sigma を閉じ込める）
def make_green_fitting(sigma, centered=False):
    if centered:
        def f(x, *p):
            R, L, mu = p
            return (-c * R * cgauss(x, sigma, mu) - (c**2) * L * cp_gauss(x, sigma, mu)) * 1e-12
        return f
    def f(x, *p):
        R, L = p
        return (-c * R * gauss(x, sigma) - (c**2) * L * p_gauss(x, sigma)) * 1e-12
    return f

def make_green_fittingR(sigma):
    def f(x, R):
        return (-c * R * gauss(x, sigma)) * 1e-12
    return f

# =====================================================================
# 計算状態（旧コードのグローバル DataFrame 群を 1 オブジェクトに集約）
# =====================================================================
@dataclass
class State:
    comp_list: list
    comp_name: list
    comp_name2: list           # CSR/CWR を除いた成分（横方向用）
    sigmaz: float              # 元データのバンチ長 [m]（通常 0.5e-3）
    ds: float                  # 時間刻み [ns]
    slen: np.ndarray           # 出力 s グリッド [ns]
    centered_fit: bool

    # 縦方向
    wakez: pd.DataFrame = field(default_factory=pd.DataFrame)
    loss_factor: pd.DataFrame = None
    resistance: pd.DataFrame = None
    inductance: pd.DataFrame = None
    resistance_R: pd.DataFrame = None
    # 横方向 quad/dipole
    wakeQx: pd.DataFrame = field(default_factory=pd.DataFrame)
    wakeQy: pd.DataFrame = field(default_factory=pd.DataFrame)
    wakeDx: pd.DataFrame = field(default_factory=pd.DataFrame)
    wakeDy: pd.DataFrame = field(default_factory=pd.DataFrame)
    kick_factorQx: pd.DataFrame = None
    kick_factorQy: pd.DataFrame = None
    kick_factorDx: pd.DataFrame = None
    kick_factorDy: pd.DataFrame = None
    # ラベル（プロット用。parquet とは別に json で保存）
    labelz_dict: dict = field(default_factory=dict)
    labelQx_dict: dict = field(default_factory=dict)
    labelQy_dict: dict = field(default_factory=dict)
    labelDx_dict: dict = field(default_factory=dict)
    labelDy_dict: dict = field(default_factory=dict)

    @classmethod
    def create(cls, comp_list, sigmaz, ds, slen, centered_fit):
        comp_name = [d.get('name') for d in comp_list]
        comp_name2 = [n for n in comp_name if n not in ('CSR', 'CWR')]
        s = cls(comp_list=comp_list, comp_name=comp_name, comp_name2=comp_name2,
                sigmaz=sigmaz, ds=ds, slen=slen, centered_fit=centered_fit)
        s.loss_factor   = pd.DataFrame(index=['V/pC'], columns=comp_name)
        s.resistance    = pd.DataFrame(index=['ohm'],  columns=comp_name)
        s.inductance    = pd.DataFrame(index=['nH'],   columns=comp_name)
        s.resistance_R  = pd.DataFrame(index=['ohm'],  columns=comp_name)
        s.kick_factorQx = pd.DataFrame(index=['V/pC'], columns=comp_name2)
        s.kick_factorQy = pd.DataFrame(index=['V/pC'], columns=comp_name2)
        s.kick_factorDx = pd.DataFrame(index=['V/pC'], columns=comp_name2)
        s.kick_factorDy = pd.DataFrame(index=['V/pC'], columns=comp_name2)
        return s


# =====================================================================
# パラメータファイル読み込み & リング自動判定
# =====================================================================
def load_parameters(param_path):
    """{param}.py を実行し comp_list（と ave_betax 等）を取得する。"""
    ns = {}
    with open(param_path, encoding='utf-8') as f:
        exec(f.read(), ns)
    if 'comp_list' not in ns:
        raise KeyError(f'{param_path} に comp_list が定義されていません。')
    return ns['comp_list']

def resolve_param_path(param, param_root, mv):
    """--param を解決して (パラメータファイルのパス, パラメータ名) を返す。

    - パスっぽい指定（`.py` で終わる / パス区切りを含む / 実在ファイル）は
      そのまま読み込む。例: `./2021c_physics.py`, `parameters/2021c_physics.py`
    - そうでなければ従来構造 `<param_root>/version<mv>/<param>.py` を組み立てる。
      例: `2021c_physics` → `parameters/version2.2/2021c_physics.py`

    パラメータ名（出力ディレクトリや wakeLT のファイル名に使う）は、解決した
    ファイルの拡張子なし基底名とする。見つからなければ分かりやすいエラーを出す。
    """
    looks_like_path = (param.endswith('.py') or ('/' in param)
                       or (os.sep in param) or os.path.exists(param))
    if looks_like_path:
        candidate = param
    else:
        candidate = os.path.join(param_root, f'version{mv}', f'{param}.py')

    if not os.path.exists(candidate):
        legacy = os.path.join(param_root, f'version{mv}',
                              param if param.endswith('.py') else param + '.py')
        raise FileNotFoundError(
            f'パラメータファイルが見つかりません: {candidate}\n'
            f'  --param には次のいずれかを指定してください:\n'
            f'    ・ファイルパス（例: ./2021c_physics.py, parameters/2021c_physics.py）\n'
            f'    ・<param-root>/version<mv>/ 直下に置いた名前（例: 2021c_physics '
            f'→ {legacy}）')
    param_name = os.path.splitext(os.path.basename(candidate))[0]
    return candidate, param_name

def detect_ring(comp_list, cwd=None):
    """成分名(QC1RP→LER / QC1RE→HER)と親ディレクトリ名の両方から判定し、
    食い違えば警告を出す。判定できなければ None を返す。"""
    names = [d.get('name', '') for d in comp_list]
    by_comp = None
    if any(n.endswith(('RP', 'LP')) for n in names):   # P = positron
        by_comp = 'LER'
    elif any(n.endswith(('RE', 'LE')) for n in names):  # E = electron
        by_comp = 'HER'

    cwd = (cwd or os.getcwd()).lower()
    by_dir = 'LER' if 'ler' in cwd else 'HER' if 'her' in cwd else None

    if by_comp and by_dir and by_comp != by_dir:
        print(f'[警告] リング判定が不一致: 成分名→{by_comp}, ディレクトリ→{by_dir}。'
              f' 成分名({by_comp})を採用します。', file=sys.stderr)
    ring = by_comp or by_dir
    if ring is None:
        print('[警告] リングを自動判定できませんでした。', file=sys.stderr)
    return ring


# =====================================================================
# コリメータ wake 生成
# =====================================================================
WAKE_SPECS = {
    'z':  ('longitudinal', '_z.txt'),
    'Qx': ('quadrupole_x', '_Qx.txt'),
    'Qy': ('quadrupole_y', '_Qy.txt'),
    'Dx': ('dipole_x',     '_Dx.txt'),
    'Dy': ('dipole_y',     '_Dy.txt'),
}

def _read_aperture_frames(d_list, clm_dir):
    """各 wake 種別について、アパーチャ値を列名とした DataFrame を構築。"""
    # アパーチャ間で s 格子は本来同一だが、GdfidL 出力には ~1e-6 ns の数値的
    # ドリフトが乗ることがある。これを実在の s 差と誤認すると格子が分裂して
    # アーティファクトになるため、ノートブック同様「行番号(=同一行は同一 s)」で
    # 整列し、s グリッドは先頭アパーチャのものを採用する。
    # （各アパーチャファイルは同じ行数・同じ行順であることが前提。）
    dfs = {}
    for key, (subdir, suffix) in WAKE_SPECS.items():
        first = pd.read_table(
            f'{clm_dir}/out/{subdir}/{d_list[0]}{suffix}', header=None
        ).rename(columns={0: 's', 1: float(d_list[0][1:])})
        cols = {'s': first['s'].values, float(d_list[0][1:]): first[float(d_list[0][1:])].values}
        n = len(first)
        for d in d_list[1:]:
            ap = float(d[1:])
            tmp = pd.read_table(f'{clm_dir}/out/{subdir}/{d}{suffix}', header=None)
            if len(tmp) != n:
                raise ValueError(
                    f'{clm_dir}/{subdir}: アパーチャ {d} の行数({len(tmp)})が '
                    f'{d_list[0]}({n})と異なります。行番号整列ができません。')
            cols[ap] = tmp[1].values            # 行番号で整列（先頭の s に合わせる）
        dfs[key] = pd.DataFrame(cols).set_index('s')
    return dfs['z'], dfs['Qx'], dfs['Qy'], dfs['Dx'], dfs['Dy']

def _dz_of(df):
    idx = df.index.values.tolist()
    return float(Decimal(idx[1] - idx[0]).quantize(Decimal("1e-15"), rounding=ROUND_HALF_UP))

def generate_clm_wake(st, d_list, clm_dir, col_list):
    """両ジョー(上下2値アパーチャ)コリメータ。st のグローバル wake に追記する。"""
    z, Qx, Qy, Dx, Dy = _read_aperture_frames(d_list, clm_dir)
    info_by_name = {d['name']: d for d in st.comp_list if d.get('name') in col_list}

    def interp(df, ap):
        df[ap] = np.nan
        df = df.sort_index(axis=1)
        return df.interpolate(method='values', axis=1)

    for name in col_list:
        info = info_by_name.get(name)
        if info is None:
            print(f'[警告] {name} が comp_list に見つかりません。スキップ。', file=sys.stderr)
            continue
        beta_x, beta_y = info['beta_x'], info['beta_y']
        top, bottom = info['aperture']                # 2要素集合
        half = round(abs((top - bottom) / 2), 2)

        z, Qx, Qy, Dx, Dy = (interp(z, half), interp(Qx, half), interp(Qy, half),
                             interp(Dx, half), interp(Dy, half))

        st.wakez[name] = z[half]
        b = gauss(st.wakez.index.values, st.sigmaz / c / 1e-9)
        st.loss_factor[name] = (-b * st.wakez[name] * _dz_of(st.wakez)).sum()
        st.labelz_dict[name] = f'{name}, d={half} mm, $k_z=${st.loss_factor[name].iloc[0]:.2e} V/pC'

        for W, beta, Wcol, kf, tag in [
            (Qx, beta_x, st.wakeQx, st.kick_factorQx, 'x,Q'),
            (Qy, beta_y, st.wakeQy, st.kick_factorQy, 'y,Q'),
            (Dx, beta_x, st.wakeDx, st.kick_factorDx, 'x,D'),
            (Dy, beta_y, st.wakeDy, st.kick_factorDy, 'y,D'),
        ]:
            Wcol[name] = W[half] * beta
            b = gauss(Wcol.index.values, st.sigmaz / c / 1e-9)
            kf[name] = (-b * Wcol[name] / beta * _dz_of(Wcol)).sum() * beta
    return z, Qx, Qy, Dx, Dy

def generate_kclm_wake(st, d_list, clm_dir, col_list):
    """片ジョー(knife-edge, 1値アパーチャ)コリメータ。データの s グリッドが
    他と異なるため、行方向に concat→補間→重複除去でグローバル wake を作り直す。
    横方向 wake は片側ジョーのため aperture 符号で向きを反転する。
    （旧 HER_wake_integrator の generate_kclm_wake を忠実移植）"""
    z, Qx, Qy, Dx, Dy = _read_aperture_frames(d_list, clm_dir)
    info_by_name = {d['name']: d for d in st.comp_list if d.get('name') in col_list}

    # --- グローバル wake に行方向(axis=0)で結合し補間 ---
    def merge_axis0(glob, local):
        glob = pd.concat([glob, local]).sort_index(axis=0)
        return glob.interpolate(method='values', axis=0)

    st.wakez  = merge_axis0(st.wakez,  z)
    st.wakeQx = merge_axis0(st.wakeQx, Qx)
    st.wakeQy = merge_axis0(st.wakeQy, Qy)
    st.wakeDx = merge_axis0(st.wakeDx, Dx)
    st.wakeDy = merge_axis0(st.wakeDy, Dy)
    # アパーチャ値の作業列を除去（成分名列だけ残す）
    for d in d_list:
        col = float(d[1:])
        for attr in ('wakez', 'wakeQx', 'wakeQy', 'wakeDx', 'wakeDy'):
            W = getattr(st, attr)
            if col in W.columns:
                setattr(st, attr, W.drop(col, axis=1))

    # ローカル frame を辞書で保持（コリメータごとに half 列を追加していく）
    loc = {'z': z, 'Qx': Qx, 'Qy': Qy, 'Dx': Dx, 'Dy': Dy}

    def interp_col(key, ap):
        df = loc[key]
        df[ap] = np.nan
        df = df.sort_index(axis=1)
        df = df.interpolate(method='values', axis=1)
        loc[key] = df
        return df

    for name in col_list:
        info = info_by_name.get(name)
        if info is None:
            print(f'[警告] {name} が comp_list に見つかりません。スキップ。', file=sys.stderr)
            continue
        beta_x, beta_y = info['beta_x'], info['beta_y']
        ap0 = list(info['aperture'])[0]                # 1要素集合
        half = abs(ap0)
        sgn = -np.sign(ap0)                            # 片ジョー符号反転

        # 縦方向
        zc = interp_col('z', half)
        st.wakez[name] = zc[half]
        st.wakez = st.wakez.interpolate(method='values', axis=0)
        b = gauss(st.wakez.index.values, st.sigmaz / c / 1e-9)
        st.loss_factor[name] = (-b * st.wakez[name] * _dz_of(st.wakez)).sum()
        st.labelz_dict[name] = f'{name}, d={half} mm, $k_z=${st.loss_factor[name].iloc[0]:.2e} V/pC'

        # 横方向（符号反転 + beta 重み）
        for key, beta, attr, kf, tag, ld in [
            ('Qx', beta_x, 'wakeQx', st.kick_factorQx, 'x,Q', st.labelQx_dict),
            ('Qy', beta_y, 'wakeQy', st.kick_factorQy, 'y,Q', st.labelQy_dict),
            ('Dx', beta_x, 'wakeDx', st.kick_factorDx, 'x,D', st.labelDx_dict),
            ('Dy', beta_y, 'wakeDy', st.kick_factorDy, 'y,D', st.labelDy_dict),
        ]:
            Wsrc = interp_col(key, half)
            Wcol = getattr(st, attr)
            Wcol[name] = sgn * Wsrc[half] * beta
            Wcol = Wcol.interpolate(method='values', axis=0)
            setattr(st, attr, Wcol)
            b = gauss(Wcol.index.values, st.sigmaz / c / 1e-9)
            kf[name] = (-b * Wcol[name] / beta * _dz_of(Wcol)).sum() * beta
            betalbl = r'$\beta_x$' if 'x' in tag else r'$\beta_y$'
            ld[name] = f'{name}, d={half} mm, {betalbl}={beta} m, $k_{{{tag}}}=${kf[name].iloc[0]:.2e} V/pC'

    # 重複インデックスを最後の値で一意化
    for attr in ('wakez', 'wakeQx', 'wakeQy', 'wakeDx', 'wakeDy'):
        setattr(st, attr, getattr(st, attr).groupby(level=0).last())
    return loc['z'], loc['Qx'], loc['Qy'], loc['Dx'], loc['Dy']

def get_collimator_groups(comp_list):
    """D**[VH]* 形式のコリメータを directory 単位でグループ化（comp_list 順を保持）。"""
    pat = re.compile(r'D\d\d[VH]\d$')
    groups = {}
    for d in comp_list:
        name = d.get('name', '')
        if pat.match(name):
            groups.setdefault(d['directory'], []).append(name)
    return groups

def process_collimators(st, vc_d_list, hc_d_list):
    """全コリメータグループを自動振り分けして処理。プロット用に vc/hc 名リストを返す。"""
    groups = get_collimator_groups(st.comp_list)
    vc_list, hc_list = [], []

    def cardinality(name):
        info = next(d for d in st.comp_list if d.get('name') == name)
        return len(info.get('aperture'))

    for clm_dir, names in groups.items():
        if 'v_collimator' in clm_dir:
            d_list, bucket = vc_d_list, vc_list
        elif 'h_collimator' in clm_dir:
            d_list, bucket = hc_d_list, hc_list
        else:
            print(f'[警告] "{clm_dir}" を v/h と判定できません。スキップ。', file=sys.stderr)
            continue
        bucket.extend(names)

        cards = {cardinality(n) for n in names}
        if cards == {2}:
            print(f'[コリメータ/両ジョー] {clm_dir} → {names}')
            generate_clm_wake(st, d_list, clm_dir, names)
        elif cards == {1}:
            print(f'[コリメータ/片ジョー] {clm_dir} → {names}')
            generate_kclm_wake(st, d_list, clm_dir, names)
        else:
            raise ValueError(f'{clm_dir} 内で aperture 要素数が混在: {names} '
                             '(同一 directory のコリメータは同じジョー構成である必要があります)')
    return vc_list, hc_list


# =====================================================================
# コリメータ以外の成分（CSR/CWR と汎用成分）
# =====================================================================
def process_other_components(st):
    for d in st.comp_list:
        name = d.get('name')
        if re.match(r'D\d\d[VH]\d', name):
            continue                                   # コリメータは処理済み
        comp_dir = d.get('directory')

        if re.match(r'(CSR|CWR)', name):
            z = pd.read_table(comp_dir + '/out/z.txt', header=None)
            z = z.rename(columns={0: 's', 1: name}).set_index('s')
            st.wakez = pd.concat([st.wakez, z]).sort_index()
            st.loss_factor[name] = np.loadtxt(comp_dir + '/loss/loss_factor.txt', dtype='float')
            st.labelz_dict[name] = f'{name}, $k_z=${st.loss_factor[name].iloc[0]:.2e} V/pC'
            continue

        q = d.get('quantity')
        bx, by = d.get('beta_x'), d.get('beta_y')
        for sub, col, Wcol, kf, beta, label_dict, tag in [
            ('z',  name, 'wakez',  st.loss_factor,  None, st.labelz_dict, None),
            ('Qx', name, 'wakeQx', st.kick_factorQx, bx,  st.labelQx_dict, 'x,Q'),
            ('Qy', name, 'wakeQy', st.kick_factorQy, by,  st.labelQy_dict, 'y,Q'),
            ('Dx', name, 'wakeDx', st.kick_factorDx, bx,  st.labelDx_dict, 'x,D'),
            ('Dy', name, 'wakeDy', st.kick_factorDy, by,  st.labelDy_dict, 'y,D'),
        ]:
            fname = {'z': 'z.txt', 'Qx': 'Qx.txt', 'Qy': 'Qy.txt',
                     'Dx': 'Dx.txt', 'Dy': 'Dy.txt'}[sub]
            df = pd.read_table(comp_dir + '/out/' + fname, header=None)
            df = df.rename(columns={0: 's', 1: name})
            factor = q if beta is None else beta * q
            df[name] = df[name] * factor
            df = df.set_index('s')
            cur = getattr(st, Wcol)
            setattr(st, Wcol, pd.concat([cur, df]).sort_index())

            lossfile = {'z': 'loss_factor.txt', 'Qx': 'kick_factorQx.txt',
                        'Qy': 'kick_factorQy.txt', 'Dx': 'kick_factorDx.txt',
                        'Dy': 'kick_factorDy.txt'}[sub]
            val = np.loadtxt(comp_dir + '/loss/' + lossfile, dtype='float') * factor
            kf[name] = val
            if sub == 'z':
                st.labelz_dict[name] = f'{name}, qty: {q}, $k_z=${kf[name].iloc[0]:.2e} V/pC'
            else:
                label_dict[name] = (f'{name}, qty: {q}, '
                                    + (r'$\beta_x$' if 'x' in tag else r'$\beta_y$')
                                    + f'={beta} m, $k_{{{tag}}}=${kf[name].iloc[0]:.2e} V/pC')


# =====================================================================
# wake 統合（s グリッド整列・補間・sum 列付与）
# =====================================================================
def assemble_wakes(st):
    def finalize(df):
        df = df.sort_index()
        df.index = df.index.to_series().apply(lambda x: np.round(x, 5))
        df = df.groupby(level=0).last()
        return df

    st.wakez  = finalize(st.wakez)
    st.wakeQx = finalize(st.wakeQx)
    st.wakeQy = finalize(st.wakeQy)
    st.wakeDx = finalize(st.wakeDx)
    st.wakeDy = finalize(st.wakeDy)

    slen_idx = pd.Index(st.slen)
    for attr in ('wakez', 'wakeQx', 'wakeQy', 'wakeDx', 'wakeDy'):
        df = getattr(st, attr)
        df = df.reindex(df.index.union(slen_idx))
        df.loc[-1e5] = 0.0
        df.loc[1e5] = 0.0
        df = df.sort_index().interpolate(method='values', axis=0)
        df['sum'] = df.sum(axis=1)
        setattr(st, attr, df)

    slen_append = np.concatenate(([-1e5], st.slen, [1e5]))
    for attr in ('wakez', 'wakeQx', 'wakeQy', 'wakeDx', 'wakeDy'):
        setattr(st, attr, getattr(st, attr).loc[slen_append, :])


def fit_RL(st):
    """各成分の縦方向 wake を Green 関数で R/L フィット。"""
    slength = list(map(convert_to_length, st.wakez.iloc[1:-1, :].index.values.tolist()))
    green = make_green_fitting(st.sigmaz, st.centered_fit)
    greenR = make_green_fittingR(st.sigmaz)
    for d in st.comp_list:
        name = d.get('name')
        y = st.wakez.iloc[1:-1, :][name]
        if st.centered_fit:
            param, _ = curve_fit(green, slength, y, (100, 0.01, 0.0))
        else:
            param, _ = curve_fit(green, slength, y, (100, 0.01))
            paramR, _ = curve_fit(greenR, slength, y, 100)
            st.resistance_R[name] = paramR[0]
        st.resistance[name] = param[0]
        st.inductance[name] = param[1] * 1e9


# =====================================================================
# 出力（sigma_z = データ生値）
# =====================================================================
def write_base_outputs(st, dir_name):
    sz = st.sigmaz * 1e3
    out = f'{dir_name}/out_sz{sz}'
    st.loss_factor.to_csv(f'{out}/loss_factor.csv', sep=',')
    st.kick_factorDx.to_csv(f'{out}/kick_factorDx.csv', sep=',')
    st.kick_factorDy.to_csv(f'{out}/kick_factorDy.csv', sep=',')
    st.kick_factorQx.to_csv(f'{out}/kick_factorQx.csv', sep=',')
    st.kick_factorQy.to_csv(f'{out}/kick_factorQy.csv', sep=',')
    st.resistance.to_csv(f'{out}/resistance.csv', sep=',')
    st.inductance.to_csv(f'{out}/inductance.csv', sep=',')

def write_wakeLT(st, dir_name, parameter_name):
    path = f'{dir_name}/out/wakeLT_{parameter_name}.txt'
    pd.concat([st.wakez['sum'], st.wakeDx['sum'], st.wakeQx['sum'],
               st.wakeDy['sum'], st.wakeQy['sum']], axis=1).to_csv(
        path, sep='\t', header=False, index=True, float_format='%.8e')
    KO = pd.read_table(path, header=None, sep=r'\s+')
    KO[0] = KO[0] * c * 1e-9
    KO = KO.iloc[1:-1]
    KO.to_csv(f'{dir_name}/out/wakeLT_{parameter_name}_KO.dat',
              sep='\t', header=False, index=False, float_format='%.8e')

def save_intermediate(st, dir_name):
    """プロット再現用に各成分 wake と factor を parquet/json で保存。"""
    import json
    interm = f'{dir_name}/intermediate'
    os.makedirs(interm, exist_ok=True)
    for attr in ('wakez', 'wakeQx', 'wakeQy', 'wakeDx', 'wakeDy'):
        getattr(st, attr).to_parquet(f'{interm}/{attr}.parquet')
    for attr in ('loss_factor', 'resistance', 'inductance',
                 'kick_factorQx', 'kick_factorQy', 'kick_factorDx', 'kick_factorDy'):
        getattr(st, attr).to_parquet(f'{interm}/{attr}.parquet')
    labels = {k: getattr(st, k) for k in
              ('labelz_dict', 'labelQx_dict', 'labelQy_dict', 'labelDx_dict', 'labelDy_dict')}
    with open(f'{interm}/labels.json', 'w', encoding='utf-8') as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)


# =====================================================================
# 畳み込み（標準バンチ長での impedance budgeting）
# =====================================================================
def run_convolution(st, dir_name, conv_sigmaz_list):
    ds = st.ds
    comp_name, comp_name2 = st.comp_name, st.comp_name2
    conv_summary = pd.DataFrame(
        index=np.arange(len(conv_sigmaz_list)),
        columns=['sigma_z', 'loss', 'kick_x_d', 'kick_x_q', 'kick_y_d', 'kick_y_q'])

    for ite, conv_sigmaz in enumerate(conv_sigmaz_list):
        conv_sigmaz_name = round(conv_sigmaz * 1e3, 2)
        print(f'count: {ite+1}/{len(conv_sigmaz_list)}  (sigma_z = {conv_sigmaz_name} mm)')
        os.makedirs(f'{dir_name}/out_sz{conv_sigmaz_name}', exist_ok=True)
        os.makedirs(f'{dir_name}/intermediate/conv_sz{conv_sigmaz_name}', exist_ok=True)

        conv_blen = np.arange(-5 * conv_sigmaz / c / 1e-9,
                              5 * conv_sigmaz / c / 1e-9 + ds, ds)
        conv_b_shape = gauss(conv_blen, conv_sigmaz / c / 1e-9)

        conv_loss_factor   = pd.DataFrame(index=['V/pC'], columns=comp_name)
        conv_resistance    = pd.DataFrame(index=['ohm'],  columns=comp_name)
        conv_inductance    = pd.DataFrame(index=['nH'],   columns=comp_name)
        conv_kick_factorQx = pd.DataFrame(index=['V/pC'], columns=comp_name2)
        conv_kick_factorQy = pd.DataFrame(index=['V/pC'], columns=comp_name2)
        conv_kick_factorDx = pd.DataFrame(index=['V/pC'], columns=comp_name2)
        conv_kick_factorDy = pd.DataFrame(index=['V/pC'], columns=comp_name2)

        conv = {}   # 'z','Dx',... → conv_wake DataFrame
        srcmap = {'z': st.wakez, 'Dx': st.wakeDx, 'Dy': st.wakeDy,
                  'Qx': st.wakeQx, 'Qy': st.wakeQy}

        for key, src in srcmap.items():
            cols = comp_name if key == 'z' else comp_name2
            built = None
            for name in cols:
                convw = ds * np.convolve(src.iloc[1:-1, :][name], conv_b_shape, mode='full')
                if built is None:
                    base = src.index.tolist()
                    cs = np.arange(base[1] + conv_blen.min(),
                                   base[-2] + conv_blen.max() + ds, ds)
                    if len(convw) - len(cs) != 0:
                        cs = np.arange(base[1] + conv_blen.min(),
                                       base[-2] + conv_blen.max()
                                       + (len(convw) - len(cs) + 1) * ds, ds)
                    built = pd.DataFrame({'s': cs, name: convw})
                else:
                    built = pd.concat([built, pd.DataFrame({name: convw})], axis=1)
            built['sum'] = built.drop(columns='s').sum(axis=1)
            built['b_shape'] = gauss(built['s'].values, conv_sigmaz / c / 1e-9)
            conv[key] = built.set_index('s')

        # loss / R / L
        conv_slength = list(map(convert_to_length, conv['z'].index.values.tolist()))
        green = make_green_fitting(conv_sigmaz, st.centered_fit)
        greenR = make_green_fittingR(conv_sigmaz)
        for name in comp_name:
            conv_loss_factor[name] = (-conv['z']['b_shape'] * conv['z'][name] * ds).sum()
            if st.centered_fit:
                p, _ = curve_fit(green, conv_slength, conv['z'][name], (100, 0.1, 0))
            else:
                p, _ = curve_fit(green, conv_slength, conv['z'][name], (100, 0.1))
                curve_fit(greenR, conv_slength, conv['z'][name], 100)
            conv_resistance[name] = p[0]
            conv_inductance[name] = p[1] * 1e9

        for name in comp_name2:
            conv_kick_factorDx[name] = (-conv['Dx']['b_shape'] * conv['Dx'][name] * ds).sum()
            conv_kick_factorDy[name] = (-conv['Dy']['b_shape'] * conv['Dy'][name] * ds).sum()
            conv_kick_factorQx[name] = (-conv['Qx']['b_shape'] * conv['Qx'][name] * ds).sum()
            conv_kick_factorQy[name] = (-conv['Qy']['b_shape'] * conv['Qy'][name] * ds).sum()

        # 出力
        out = f'{dir_name}/out_sz{conv_sigmaz_name}'
        conv_loss_factor.to_csv(f'{out}/loss_factor.csv', sep=',')
        conv_kick_factorDx.to_csv(f'{out}/kick_factorDx.csv', sep=',')
        conv_kick_factorDy.to_csv(f'{out}/kick_factorDy.csv', sep=',')
        conv_kick_factorQx.to_csv(f'{out}/kick_factorQx.csv', sep=',')
        conv_kick_factorQy.to_csv(f'{out}/kick_factorQy.csv', sep=',')
        conv_resistance.to_csv(f'{out}/resistance.csv', sep=',')
        conv_inductance.to_csv(f'{out}/inductance.csv', sep=',')
        # プロット再現用 conv wake
        for key, df in conv.items():
            df.to_parquet(f'{dir_name}/intermediate/conv_sz{conv_sigmaz_name}/conv_wake{key}.parquet')

        conv_summary.loc[ite, 'sigma_z'] = conv_sigmaz_name
        conv_summary.loc[ite, 'loss']    = conv_loss_factor.sum(axis=1).iloc[0]
        conv_summary.loc[ite, 'kick_x_d'] = conv_kick_factorDx.sum(axis=1).iloc[0]
        conv_summary.loc[ite, 'kick_x_q'] = conv_kick_factorQx.sum(axis=1).iloc[0]
        conv_summary.loc[ite, 'kick_y_d'] = conv_kick_factorDy.sum(axis=1).iloc[0]
        conv_summary.loc[ite, 'kick_y_q'] = conv_kick_factorQy.sum(axis=1).iloc[0]

    conv_summary = conv_summary.set_index('sigma_z')
    conv_summary.to_csv(f'{dir_name}/out/conv_summary.csv', sep=',')
    _fit_summary(conv_summary, dir_name)
    return conv_summary

def _fit_summary(conv_summary, dir_name):
    """conv_summary を a*x^b+c でフィットし fit_param を出力。
    データ点が 3 未満ならフィットをスキップ(NaN)。"""
    def func_exp(x, a, b, c):
        return a * x**b + c
    targets = {
        'kz_fit_param.txt':  ('loss',     [100, -2, 100]),
        'bkx_fit_param.txt': ('kick_x_d', [1e5, -2, 2e4]),
        'bky_fit_param.txt': ('kick_y_d', [1e5, -2, 2e4]),
    }
    x = conv_summary.index.astype(float)
    for fname, (col, p0) in targets.items():
        if len(conv_summary) >= len(p0):
            param, _ = curve_fit(func_exp, x, conv_summary[col].astype(float), p0=p0)
        else:
            param = [np.nan] * 3
            print(f'[警告] conv_summary の点数({len(conv_summary)}) < パラメータ数({len(p0)})。'
                  f' {fname} のフィットをスキップ。', file=sys.stderr)
        with open(f'{dir_name}/out/{fname}', 'w') as f:
            f.write(f'{param[0]},{param[1]},{param[2]}\n')


# =====================================================================
# メイン
# =====================================================================
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description='LER/HER 統合 wake インテグレータ（計算）')
    ap.add_argument('--param', required=True,
                    help="パラメータ名 or パラメータファイルのパス。"
                         "名前のみ（例: 2021c_physics）なら "
                         "<param-root>/version<mv>/<param>.py を読み込む。"
                         "パス（例: ./2021c_physics.py, parameters/2021c_physics.py）なら直接読み込む")
    ap.add_argument('--model-version', default='2.2', help='モデルバージョン（既定 2.2）')
    ap.add_argument('--param-root', default='parameters',
                    help='パラメータファイルの親ディレクトリ')
    ap.add_argument('--out-root', default='integrated wake',
                    help='出力の親ディレクトリ')
    ap.add_argument('--conv-sigmaz', nargs='+', type=float, metavar='V',
                    default=[6e-3, 9e-3, 1],
                    help='畳み込みバンチ長 [m]。値を3個渡すと linspace(START, STOP, NUM)、'
                         'それ以外の個数（1個含む）は渡した値をそのまま使用。'
                         '例: "--conv-sigmaz 6e-3"（6mmのみ） / '
                         '"--conv-sigmaz 5e-3 6e-3 2"（linspace=5,6mm）')
    ap.add_argument('--sigmaz', type=float, default=0.5e-3,
                    help='元 wake データのバンチ長 [m]（既定 0.5e-3）')
    ap.add_argument('--ds', type=float, default=1e-4, help='時間刻み [ns]')
    ap.add_argument('--centered-fit', action='store_true', help='中心オフセット込みでフィット')
    ap.add_argument('--plot', action='store_true', help='計算後に plot_wake.py を呼んで作図')
    return ap.parse_args(argv)

def main(argv=None):
    args = parse_args(argv)
    mv = args.model_version
    param = args.param

    param_path, param_name = resolve_param_path(param, args.param_root, mv)
    dir_name = f'{args.out_root}/version{mv}/{param_name}'

    comp_list = load_parameters(param_path)
    ring = detect_ring(comp_list)
    print(f'[情報] パラメータ: {param_name}  ({param_path})  /  判定リング: {ring}')

    # 出力ディレクトリ
    sz = args.sigmaz * 1e3
    for sub in ['/fig', '/out', f'/out_sz{sz}', f'/fig_sz{sz}',
                f'/fig_sz{sz}/check', f'/fig_sz{sz}/check/R']:
        os.makedirs(dir_name + sub, exist_ok=True)

    slen = np.round(np.arange(-1e-2, 3.335e-1 + args.ds, args.ds), 5)
    st = State.create(comp_list, args.sigmaz, args.ds, slen, args.centered_fit)

    # コリメータのアパーチャ刻み（縦は sub-mm、横は mm）
    vc_d_list = ['d0.5', 'd1.0', 'd1.5', 'd2.0', 'd3.0', 'd4.0', 'd5.0', 'd7.0', 'd9.0']
    hc_d_list = ['d5', 'd7', 'd9', 'd11', 'd13', 'd15', 'd17', 'd19', 'd21', 'd23']

    print('--- コリメータ処理 ---')
    process_collimators(st, vc_d_list, hc_d_list)
    print('--- その他成分処理 ---')
    process_other_components(st)
    print('--- wake 統合 ---')
    assemble_wakes(st)
    print('--- R/L フィット ---')
    fit_RL(st)

    write_base_outputs(st, dir_name)
    write_wakeLT(st, dir_name, param_name)
    save_intermediate(st, dir_name)

    cs = args.conv_sigmaz
    if len(cs) == 3:                       # START STOP NUM → linspace
        conv_sigmaz_list = np.linspace(cs[0], cs[1], int(cs[2]))
    else:                                  # 1個 or その他 → そのまま使用
        conv_sigmaz_list = np.array(cs, dtype=float)
    print(f'--- 畳み込み（sigma_z = {conv_sigmaz_list*1e3} mm）---')
    run_convolution(st, dir_name, conv_sigmaz_list)

    print(f'[完了] 出力先: {dir_name}')

    if args.plot:
        try:
            import plot_wake
            plot_wake.main(['--dir', dir_name])
        except Exception as e:
            print(f'[警告] 作図でエラー: {e}', file=sys.stderr)

    return dir_name


if __name__ == '__main__':
    main()
