# wake_integrator — LER/HER wake インテグレータ

各コンポーネントの wake potential（バンチ長 σz=0.5 mm）を統合し、インピーダンス
バジェット用の量（loss factor / kick factor / 抵抗 R・インダクタンス L）を計算する
コマンドラインツール。標準バンチ長への畳み込みにも対応。計算（`wake_integrator.py`）
と作図（`plot_wake.py`）を分離している。

## 構成
- `wake_integrator.py` … 計算本体。CSV と中間 parquet を出力（図は作らない）。
- `plot_wake.py` … 中間ファイルから作図（計算をやり直さず何度でも描き直せる）。

## 入力レイアウト
作業ディレクトリ直下に以下を置いて実行する。

```
作業ディレクトリ/
├ parameters/version2.2/<param>.py          # パラメータ（comp_list を定義）
├ <component directory>/out/...              # 各コンポーネントの wake データ
└ integrated wake/                          # 出力先（自動生成）
```

`<param>.py` は `comp_list`（各コンポーネントの名前・beta・aperture・directory・
quantity を収めた辞書のリスト）を定義する。各コンポーネントの wake データは
`comp_list` の `directory` で指定したフォルダ以下にある：
- コリメータ: `out/{longitudinal,quadrupole_x,quadrupole_y,dipole_x,dipole_y}/<d>_*.txt`
  （`<d>` はアパーチャ、例 `d0.5`, `d5`）
- その他成分: `out/{z,Qx,Qy,Dx,Dy}.txt` と `loss/{loss_factor,kick_factor*}.txt`

## 使い方
```bash
# 基本（畳み込みバンチ長は既定 6 mm のみ）
python wake_integrator.py --param 2021c_physics

# 畳み込みバンチ長: 値3個 = linspace(START, STOP, NUM)
python wake_integrator.py --param 2021c_physics --conv-sigmaz 4e-3 9e-3 4
# 畳み込みバンチ長: 1個だけ指定（6 mm のみ）
python wake_integrator.py --param 2021c_physics --conv-sigmaz 6e-3

# テスト実行
python wake_integrator.py --param ./test.py

# 計算後そのまま作図も実行
python wake_integrator.py --param 2021c_physics --plot

# 作図だけ後から（計算結果のディレクトリを指定）
python plot_wake.py --dir "integrated wake/version2.2/2021c_physics"
```
主なオプション: `--model-version`(既定 2.2) / `--sigmaz`(元データのバンチ長, 既定
0.5e-3) / `--ds`(時間刻み, 既定 1e-4) / `--centered-fit` / `--param-root` / `--out-root`。

## 出力
`integrated wake/version<mv>/<param>/` 以下に生成される。
- `out_sz<σz>/` … loss_factor.csv, kick_factor{Dx,Dy,Qx,Qy}.csv, resistance.csv,
  inductance.csv（σz=0.5 mm と各畳み込みバンチ長ごと）
- `out/wakeLT_<param>.txt`, `..._KO.dat` … 統合 wake（PyHEADTAIL 等の入力用）
- `out/conv_summary.csv`, `out/*_fit_param.txt` … バンチ長依存のまとめとフィット係数
- `intermediate/` … 作図再現用の中間データ（parquet / labels.json）

## リング・コリメータの自動判定
- **リング**: 成分名（`QC1RP`→LER / `QC1RE`→HER）と親ディレクトリ名（`ler`/`her`）の
  両方で判定し、食い違えば警告（成分名を優先採用）。
- **コリメータ**: `D##V#`/`D##H#` 形式の名前を `directory` 単位でグループ化し、
  `aperture` の要素数で自動振り分け。2 値=両ジョー、1 値=片ジョー。
  新しいコリメータはパラメータファイルに 1 行追加するだけで対応する。

## 仕様メモ
- **アパーチャ補間の整列**: コリメータの各アパーチャ wake ファイルは同一の s 格子を
  持つ前提で、行番号（同一行＝同一 s）で結合し、s 格子は先頭アパーチャのものを採用する。
  GdfidL 出力に乗りうる ~1e-6 ns 程度の数値ドリフトを実在の s 差と誤認しないための措置。
  各アパーチャの行数が異なる場合は明示的にエラーを出す。
- **片ジョー処理の dz**: 片ジョーコリメータは両ジョーと別の s 格子を持ちうるため、
  マージ後の隣接インデックス間隔を dz に使う。両格子が偶然完全一致すると dz=0 に
  なり得るので、格子構成を変える場合は注意。
- 中間 parquet の読み書きには pyarrow が必要（`pip install pyarrow`）。

## Wakeデータ
- Wakeデータは SuperKEKB International Task Force TMCI Subgroup の indicoページ（Impedance data repository: https://kds.kek.jp/event/40318/）から入手可能。
