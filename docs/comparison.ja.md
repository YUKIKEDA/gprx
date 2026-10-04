## 他ライブラリとの比較

GPR の論文で使われる回帰で、gprx を scikit-learn、GPyTorch、GPy、libgp、friedrich と比べる。見る順は、計算が合っているか、予測が良いか、大きいデータで回るか。これは報告であり、テストの合否ではない。他のライブラリの方が良い行も、表から消さない。

| 段 | データ | 見るもの | gprx のモデル |
| --- | --- | --- | --- |
| T0 | Snelson 1 次元（200 点）、Mauna Loa CO₂ | 当てはまりを目で見る | `Gpr` |
| T1 | UCI。Hernández-Lobato & Adams の split（各 20。Boston は除く）: yacht、energy、concrete、wine (red)、power plant、kin8nm、naval | 精度と、較正された不確実性 | `Gpr` |
| T2 | Kin40k、Protein（5 split） | 中規模のスケール | `K` がメモリに入る範囲は `Gpr`、それ以外と比較用に `Sgpr` / `Svgp` |
| T3 | 3DRoad、Song、Buzz、HouseElectric（`treforevans/uci_datasets`。90 / 10 の 10 split） | 大規模のスケール | `Sgpr` / `Svgp` |

どのモデルも、入力の次元ごとに長さを持つ RBF カーネルと、ガウス分布のノイズを使う。入力と目的変数は学習データの平均と分散で揃え、初期値も同じ（長さ 1、信号の分散 1、ノイズの分散 0.1）。

指標は 3 つ。予測のずれ（RMSE）、予測分布の対数損失（NLPD）、95% の予測区間がテスト点を覆った割合。単位は `y` のもとの単位で、データの分割ごとの平均と標準誤差を書く。

比較の前に、学習せず、ハイパーパラメータを 1 組に固定して全ライブラリを評価する（`just perf-real-check`）。負の対数周辺尤度、RMSE、NLPD が 1e-6 で一致する。学習したあとの差は、目的関数の違いではなく、最適化の違いである。誘導点を使うモデルでも、周辺尤度の下界は gprx、GPyTorch、GPy で一致する（`just perf-real-check yacht 0 sgpr`）。GPyTorch だけは、テスト点の分散を自分の低ランクの式で出すので、同じパラメータと誘導点でも RMSE と NLPD が他の 2 つと 1% 未満ずれる。

### 最適化器が効く

学習にかかる時間は、行列の計算と同じくらい、最適化のやり方で変わる。各行には、どのやり方で学習したかと、尤度と勾配をまとめて計算した回数を書いてある。

やり方は 2 つ。

- ライブラリの既定。そのライブラリが学習するときの設定のまま。
- 条件を揃える。目的関数と勾配は各ライブラリ自身のものを使い、反復は最大 100 回、勾配の止まり具合は √ε、履歴は 10 で共通にする。プログラムまで同じではない。gprx は argmin の L-BFGS と More–Thuente の線探索で、この値が最初から既定になっている。scikit-learn、GPyTorch、GPy は scipy の L-BFGS-B に同じ上限を渡す。libgp は Rprop しかなく、勾配の止まり具合を指定できない。SVGP には、数十万点でも通る Adam の既定が無いので、学習率 0.01、バッチ 1024、データを 3 周、で固定した。

評価の回数が違う行どうしで、かかった時間の差を速さとはみない。メモリのピークは、プロセス全体の常駐量の最大。時間に沿ったメモリは、結果の図にある。

### インターフェース

同じ回帰を各ライブラリで書く。ARD RBF、ハイパーパラメータの学習、平均と観測分散の予測、NLPD の計算。実行できるファイル: [`compare/perf/real/snippets/`](https://github.com/YUKIKEDA/gprx/blob/main/compare/perf/real/snippets/)。

<!-- snippets:begin -->
<details><summary>gprx</summary>

```rust
let kernel = KernelSpec::from(ConstantKernel::new(1.0)?)
    * KernelSpec::from(RbfArdKernel::new(&vec![1.0; d])?);
let fitted = Gpr::new(kernel, GaussianLikelihood::new(0.1)?)
    .fit(&x, n, d, &y) // L-BFGS on the negative log marginal likelihood
    .map_err(|(_, e)| e)?;
let pred = fitted.predict(&xs, m, d)?; // variance includes the noise
let nlpd = (0..m)
    .map(|i| {
        let (v, e) = (pred.variance[i], ys[i] - pred.mean[i]);
        0.5 * (2.0 * std::f64::consts::PI * v).ln() + 0.5 * e * e / v
    })
    .sum::<f64>()
    / m as f64;
```

</details>

<details><summary>scikit-learn</summary>

```python
kernel = ConstantKernel(1.0) * RBF([1.0] * d) + WhiteKernel(0.1)
gp = GaussianProcessRegressor(kernel).fit(x, y)
mean, std = gp.predict(xs, return_std=True)  # std includes the noise
nlpd = np.mean(0.5 * np.log(2 * np.pi * std**2) + 0.5 * (ys - mean) ** 2 / std**2)
```

</details>

<details><summary>GPyTorch</summary>

```python
class ExactGP(gpytorch.models.ExactGP):
    def __init__(self, x, y, likelihood):
        super().__init__(x, y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=d))

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))


likelihood = gpytorch.likelihoods.GaussianLikelihood().double()
model = ExactGP(x, y, likelihood).double()
model.train(); likelihood.train()
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # there is no default optimizer
for _ in range(50):
    optimizer.zero_grad()
    loss = -mll(model(x), y)
    loss.backward()
    optimizer.step()
model.eval(); likelihood.eval()
with torch.no_grad():
    pred = likelihood(model(xs))  # the observation distribution
    mean, var = pred.mean, pred.variance
nlpd = torch.mean(0.5 * torch.log(2 * torch.pi * var) + 0.5 * (ys - mean) ** 2 / var)
```

</details>

<details><summary>GPy</summary>

```python
kernel = GPy.kern.RBF(d, variance=1.0, lengthscale=[1.0] * d, ARD=True)
model = GPy.models.GPRegression(x, y[:, None], kernel)
model.Gaussian_noise.variance = 0.1
model.optimize()
mean, var = model.predict(xs)  # includes the noise
nlpd = np.mean(0.5 * np.log(2 * np.pi * var) + 0.5 * (ys[:, None] - mean) ** 2 / var)
```

</details>

<details><summary>libgp</summary>

```cpp
libgp::GaussianProcess gp(d, "CovSum ( CovSEard, CovNoise)");
Eigen::VectorXd loghyper(d + 2);  // log ell (d of them), log sf, log sn
loghyper << 0.0, 0.0, 0.0, 0.0, std::log(std::sqrt(0.1));
gp.covf().set_loghyper(loghyper);
for (int i = 0; i < n; ++i) {  // add_patterns(x, y) would read strided rows
    std::vector<double> row(d);
    for (int j = 0; j < d; ++j) row[j] = x(i, j);
    gp.add_pattern(row.data(), y(i));
}
libgp::RProp rprop;  // libgp's optimizer: resilient backpropagation
rprop.init();
rprop.maximize(&gp, 100, false);
const Eigen::MatrixXd pred = gp.predict(xs, true);  // mean, latent variance
const double noise = std::exp(2.0 * gp.covf().get_loghyper()(d + 1));
double nlpd = 0.0;
for (int i = 0; i < m; ++i) {
    const double v = pred(i, 1) + noise, e = ys(i) - pred(i, 0);
    nlpd += 0.5 * std::log(2.0 * M_PI * v) + 0.5 * e * e / v;
}
nlpd /= m;
```

</details>
<!-- snippets:end -->

libgp には、比較のプログラムが避けている仕様が 2 つある。`predict` の分散はノイズを含まない。一括の `add_patterns(x, y)` は、列優先の行列を行として読むので、次元が 1 のときだけ正しい。

### 機能

`✓` は、その版のソースか文書に見つかったもの。`—` は、そこに見つからなかったもの（拡張については何も言わない）。版: scikit-learn 1.6.1、GPyTorch 1.15.2、GPy 1.14.2、libgp `f4a2fb7`、friedrich 0.6.0。

| | gprx | scikit-learn | GPyTorch | GPy | libgp | friedrich |
| --- | --- | --- | --- | --- | --- | --- |
| 言語 | Rust | Python | Python (PyTorch) | Python | C++ (Eigen) | Rust |
| ARD RBF カーネル | ✓ | ✓ | ✓ | ✓ | ✓ | — |
| カーネルの和・積 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 誘導点つきの疎 GP | ✓ `Sgpr` | — | ✓ `InducingPointKernel` | ✓ `SparseGPRegression` | — | — |
| ミニバッチの SVGP | ✓ `Svgp`（Adam） | — | ✓ | クラスのみ。ミニバッチのループは無い | — | — |
| 全部を学習し直さずに学習点を足す | ✓ `OnlineGpr` | — | ✓ `get_fantasy_model` | — | ✓ `add_pattern` | ✓ `add_samples` |
| 学習点を消す | ✓ `OnlineGpr` | — | — | — | — | — |
| 事後サンプル | ✓ `sample` | ✓ `sample_y` | ✓ | ✓ `posterior_samples` | — | ✓ `sample_at` |
| f32 / 混合精度 | ✓ | — | ✓ | — | — | — |
| 学習済みモデルの保存と読み込み | ✓ | ✓ pickle / joblib | ✓ `state_dict` | ✓ `save_model` | ✓ `write` | ✓ serde |
| 最適化器の差し替え | ✓ `Optimizer` | ✓ callable | ✓ 任意の torch 最適化器 | ✓ `optimizer=` | ✓ `RProp` / `CG` | — |

### 結果

この節の数値は、1 台の PC で、ライブラリを同じ学習条件にして比べた結果である。

全学習点を使うモデルと、誘導点を 512 個に固定したモデルは、L-BFGS で最大 100 回まで反復する。ミニバッチのモデルは Adam で、学習率 0.01、一度に 1024 点、データを 3 周する。どのライブラリも、その製品が最初から使う最適化の設定では比べていない。

全学習点を使うのは Snelson、Mauna Loa、yacht、energy。誘導点 512 個は、wine、power plant、naval が 20 分割、kin40k が 5 分割、3droad と song が 1 分割。ミニバッチは kin40k、3droad、song、HouseElectric が 1 分割ずつ。

表の各列は次の内容を示している。RMSE は予測のずれ、NLPD は予測分布の対数損失で、どちらも値が小さいほど良い。95% 区間はテスト点が予測区間に入った割合を表す。学習の秒は実測した学習時間、評価回数は尤度と勾配をまとめて計算した回数であり、評価回数が揃っている行に限って秒数を速さとして比較できる。反復の列は最適化器が数えた回数（gprx は空欄）、1 回あたりのミリ秒は学習の秒を評価回数で割った中央値である。負の対数周辺尤度は学習完了時の目的関数の値、メモリはプロセス全体の最大常駐量を表している。

song の誘導点モデルでは、gprx の NLPD が GPyTorch と GPy と違う。power plant では GPy の予測が一部の分割で壊れていて、その RMSE と NLPD の平均は当てはまりの良さにならない。

図は表のあとにある。各図の直前に、その図が何を描いているかを書いた。

<!-- bench:begin -->
測定した機械は Intel64 Family 6 Model 191 Stepping 2, GenuineIntel（論理 CPU 16、メモリ 47.8 GiB、Windows-11-10.0.26200-SP0）。scikit-learn 1.6.1, gpytorch 1.15.2, GPy 1.14.2, torch 2.14.0, scipy 1.18.1, argmin 0.11.0。libgp f4a2fb7d。

各マスは、その版のソースから書き写した設定。

| ライブラリ | 既定の最適化 | 条件を揃えた最適化 | 探索する量 | 範囲 |
| --- | --- | --- | --- | --- |
| gprx | argmin 0.11 LBFGS + MoreThuente line search; history 10, max 100 iterations, gradient-norm tolerance sqrt(eps) | same call: the gprx default already equals the shared setting | logit of log θ inside each interval, so the search is unconstrained | (1e-5, 1e5) on ℓ, signal variance and noise variance |
| sklearn | scipy minimize L-BFGS-B via optimizer='fmin_l_bfgs_b' with scipy defaults (maxiter 15000, ftol 2.2e-9, gtol 1e-5, maxcor 10, maxls 20) | scipy L-BFGS-B, maxiter 100, gtol sqrt(eps), ftol 0, maxcor 10; the library's own objective and gradient | log θ | (1e-5, 1e5) on ℓ, constant value and noise level (kernel defaults) |
| gpytorch | torch.optim.Adam, lr 0.1, 50 steps (the exact-GP tutorial setting; GPyTorch has no default optimizer) | scipy L-BFGS-B, maxiter 100, gtol sqrt(eps), ftol 0, maxcor 10; the library's own objective and gradient; gradient by autograd on -mll × n | raw parameters behind softplus | noise ≥ 1e-5 (set here; GPyTorch's default is 1e-4); ℓ and outputscale positive only |
| gpy | model.optimize(): paramz opt_lbfgsb = scipy fmin_l_bfgs_b with maxfun = maxiter = 1000, factr 1e7, pgtol 1e-5 | scipy L-BFGS-B, maxiter 100, gtol sqrt(eps), ftol 0, maxcor 10; the library's own objective and gradient | softplus (Logexp) of each parameter | positive only, no upper bound |
| libgp | RProp (resilient backpropagation), 100 iterations, eps_stop 0, Delta0 0.1, Deltamin 1e-6, Deltamax 50, eta- 0.5, eta+ 1.2; keeps the best likelihood seen | N/A: libgp offers RProp and CG only, and RProp has no gradient tolerance | log ℓ, log sf, log sn (amplitude and std, not variances) | none |
| friedrich | N/A: no ARD kernel | N/A: no ARD kernel | - | - |

#### 全学習点を使うモデル

| データセット | ライブラリ | 測れた分割 | RMSE | NLPD | 95%区間 | 学習 [秒] | 評価回数 | 反復 | 1回あたり [ms] | 負の対数周辺尤度 | メモリ [MiB] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| energy | gprx | 20/20 | 0.4796 ± 0.014 | 0.6993 ± 0.033 | 0.923 ± 0.0067 | 1.283 ± 0.013 | 116 ± 0.58 | N/A | 10.94 | -1013 ± 2.5 | 47.3 |
| energy | sklearn | 20/20 | 0.4773 ± 0.013 | 0.692 ± 0.029 | 0.924 ± 0.0077 | 10.26 ± 0.79 | 94.3 ± 7.1 | 53 ± 5.6 | 108.1 | -973.3 ± 6.1 | 225.1 |
| energy | gpytorch | 20/20 | 0.4773 ± 0.012 | 0.6894 ± 0.031 | 0.921 ± 0.0082 | 1.614 ± 0.036 | 111 ± 1.8 | 96.8 ± 1.6 | 14.2 | -968.5 ± 7.4 | 316.2 |
| energy | gpy | 20/20 | 0.4756 ± 0.013 | 0.6859 ± 0.031 | 0.923 ± 0.0075 | 10.24 ± 0.23 | 119 ± 2.7 | 99.2 ± 0.8 | 85.45 | -972 ± 7.6 | 245.9 |
| maunaloa | gprx | 1/1 | 8.749 | 4.838 | 0.358 | 1.484 | 130 | N/A | 11.41 | -1479 | 70.6 |
| maunaloa | sklearn | 1/1 | 8.495 | 4.856 | 0.369 | 16.54 | 120 | 100 | 137.8 | -1479 | 176.5 |
| maunaloa | gpytorch | 1/1 | 8.649 | 4.98 | 0.339 | 2.733 | 118 | 100 | 23.16 | -1479 | 522.9 |
| maunaloa | gpy | 1/1 | 8.717 | 5.072 | 0.332 | 17.98 | 116 | 100 | 155 | -1479 | 408.0 |
| snelson | gprx | 1/1 | N/A | N/A | N/A | 0.02802 | 23 | N/A | 1.218 | 89.81 | 11.5 |
| snelson | sklearn | 1/1 | N/A | N/A | N/A | 0.3259 | 126 | 18 | 2.587 | 89.81 | 109.8 |
| snelson | gpytorch | 1/1 | N/A | N/A | N/A | 0.1444 | 55 | 20 | 2.626 | 89.81 | 282.3 |
| snelson | gpy | 1/1 | N/A | N/A | N/A | 0.3382 | 50 | 19 | 6.765 | 89.81 | 142.4 |
| yacht | gprx | 20/20 | 0.7522 ± 0.081 | 1.01 ± 0.14 | 0.927 ± 0.009 | 0.8274 ± 0.11 | 327 ± 51 | N/A | 2.655 | -338.2 ± 20 | 13.9 |
| yacht | sklearn | 20/20 | 0.936 ± 0.074 | 1.411 ± 0.059 | 0.945 ± 0.0094 | 1.093 ± 0.079 | 89 ± 5.3 | 44.5 ± 1.6 | 11.58 | -270.2 ± 1.7 | 124.4 |
| yacht | gpytorch | 20/20 | 0.4157 ± 0.056 | 0.2 ± 0.099 | 0.916 ± 0.013 | 0.4499 ± 0.03 | 108 ± 6.5 | 68 ± 5.9 | 3.835 | -451.2 ± 27 | 287.3 |
| yacht | gpy | 20/20 | 0.3931 ± 0.055 | 0.1971 ± 0.1 | 0.91 ± 0.014 | 2.575 ± 0.14 | 120 ± 6 | 78.1 ± 4.7 | 21.12 | -460.6 ± 27 | 151.4 |

#### 誘導点 512 個のモデル

| データセット | ライブラリ | 測れた分割 | RMSE | NLPD | 95%区間 | 学習 [秒] | 評価回数 | 反復 | 1回あたり [ms] | 負の対数周辺尤度 | メモリ [MiB] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3droad | gprx | 1/1 | 8.823 | 3.597 | N/A | 5230 | 788 | N/A | 6637 | N/A | N/A |
| 3droad | gpytorch | 1/1 | 8.823 | 3.597 | N/A | 1694 | 129 | N/A | 1.314e+04 | N/A | N/A |
| 3droad | gpy | 1/1 | 8.823 | 3.597 | N/A | 3970 | 135 | N/A | 2.941e+04 | N/A | N/A |
| kin40k | gprx | 5/5 | 0.2863 ± 0.0029 | 0.1634 ± 0.0083 | 0.964 ± 0.0028 | 273.4 ± 96 | 394 ± 1.3e+02 | N/A | 681.8 | 1.039e+04 ± 59 | 890.7 |
| kin40k | gpytorch | 5/5 | 0.2865 ± 0.0028 | 0.1634 ± 0.0079 | 0.964 ± 0.0028 | 73.99 ± 4.9 | 86.4 ± 5.6 | 43 ± 2.3 | 854.9 | 1.039e+04 ± 59 | 1746.8 |
| kin40k | gpy | 5/5 | 0.2863 ± 0.0029 | 0.1634 ± 0.0083 | 0.964 ± 0.0028 | 231.8 ± 42 | 88 ± 16 | 42.4 ± 2.8 | 2643 | 1.039e+04 ± 59 | 1820.1 |
| naval | gprx | 20/20 | 1.582e-05 ± 1.1e-07 | -8.994 ± 0.00076 | 1 | 56.39 ± 8.8 | 252 ± 38 | N/A | 220.7 | -5.084e+04 ± 1.4 | 288.1 |
| naval | gpytorch | 20/20 | 1.707e-05 ± 4.4e-07 | 3784 ± 5.9e+02 | 0.419 ± 0.052 | 22.54 ± 1.3 | 82.8 ± 5 | 37.9 ± 3.8 | 272.7 | -5.075e+04 ± 20 | 745.1 |
| naval | gpy | 20/20 | 1.571e-05 ± 6e-07 | -9.681 ± 0.0095 | 0.94 ± 0.0036 | 72.17 ± 5.3 | 71.5 ± 5.3 | 7.65 ± 0.99 | 1021 | -5.624e+04 ± 63 | 675.9 |
| power_plant | gprx | 20/20 | 3.775 ± 0.039 | 2.752 ± 0.01 | 0.957 ± 0.0016 | 55.88 ± 8.6 | 309 ± 46 | N/A | 178 | -377 ± 11 | 232.3 |
| power_plant | gpytorch | 20/20 | 3.777 ± 0.04 | 2.753 ± 0.011 | 0.957 ± 0.0016 | 18.94 ± 1.4 | 82.8 ± 5 | 33.5 ± 0.87 | 222.3 | -377 ± 11 | 652.0 |
| power_plant | gpy | 20/20 | 3.097e+13 ± 3.1e+13 | 3.312e+40 ± 3.3e+40 | 0.958 ± 0.0026 | 54.58 ± 2.5 | 79.8 ± 3.5 | 33.6 ± 1.3 | 666.6 | -7.044e+45 ± 7e+45 | 570.0 |
| song | gprx | 1/1 | 0.5741 | 6.122 | N/A | 31.96 | 26 | N/A | 1229 | N/A | N/A |
| song | gpytorch | 1/1 | 0.5741 | 0.8639 | N/A | 579.9 | 47 | N/A | 1.234e+04 | N/A | N/A |
| song | gpy | 1/1 | 0.5741 | 0.8639 | N/A | 6355 | 47 | N/A | 1.352e+05 | N/A | N/A |
| wine_red | gprx | 20/20 | 0.6279 ± 0.0082 | 0.951 ± 0.014 | 0.939 ± 0.0048 | 10.78 ± 2.9 | 226 ± 56 | N/A | 45.09 | 1687 ± 2 | 63.5 |
| wine_red | gpytorch | 20/20 | 0.6276 ± 0.0082 | 0.9506 ± 0.014 | 0.939 ± 0.0047 | 6.03 ± 0.13 | 114 ± 0.82 | 100 | 51.29 | 1688 ± 2 | 376.5 |
| wine_red | gpy | 20/20 | 0.6275 ± 0.0082 | 0.9504 ± 0.014 | 0.94 ± 0.0048 | 22.8 ± 1 | 112 ± 3.1 | 97.3 ± 2.7 | 196.3 | 1688 ± 2 | 299.8 |

#### ミニバッチのモデル

| データセット | ライブラリ | 測れた分割 | RMSE | NLPD | 95%区間 | 学習 [秒] | 評価回数 | 反復 | 1回あたり [ms] | 負の対数周辺尤度 | メモリ [MiB] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3droad | gprx | 1/1 | 11.18 | 3.834 | 0.948 | 34.49 | 1.15e+03 | N/A | 30.02 | N/A | 6236.7 |
| 3droad | gpytorch | 1/1 | 11.16 | 3.832 | 0.944 | 42.93 | 1.15e+03 | 1.15e+03 | 37.36 | N/A | 1284.3 |
| houseelectric | gprx | 1/1 | 0.05622 | -1.457 | 0.946 | 176.5 | 5.41e+03 | N/A | 32.65 | N/A | 29590.8 |
| houseelectric | gpytorch | 1/1 | 0.05872 | -1.419 | 0.945 | 203 | 5.41e+03 | 5.41e+03 | 37.55 | N/A | 5508.9 |
| kin40k | gprx | 1/1 | 0.6087 | 0.9093 | 0.921 | 3.181 | 108 | N/A | 29.45 | N/A | 638.8 |
| kin40k | gpytorch | 1/1 | 0.5975 | 0.8712 | 0.939 | 4.912 | 108 | 108 | 45.48 | N/A | 457.7 |
| song | gprx | 1/1 | 0.4669 | 0.6575 | 0.947 | 57.79 | 1.36e+03 | N/A | 42.53 | N/A | 8650.4 |
| song | gpytorch | 1/1 | 0.4709 | 0.666 | 0.954 | 55.96 | 1.36e+03 | 1.36e+03 | 41.17 | N/A | 3769.8 |

点の色と形はどの図でも同じ。青丸が gprx、橙の四角が scikit-learn、緑の三角が GPyTorch、黄の菱形が GPy。

**予測の誤差（全学習点）**

データセットごとに、上段に RMSE、下段に NLPD を並べている（どちらも小さいほど良い）。プロットされた点はライブラリ、縦棒は分割ごとのばらつきを表す。Snelson はテスト用データが無いため列が空になっている。

![予測の誤差（全学習点）](bench/accuracy_matched.svg)

**学習の時間（全学習点）**

上段に対数軸で学習の所要秒数を、下段に尤度と勾配をまとめて計算した回数を示した。秒数を比較する際は、下段の計算回数が揃っているかに着目する。

![学習の時間（全学習点）](bench/fit_time_matched.svg)

**予測の誤差（誘導点 512 個）**

図の見方は全学習点の予測誤差と同じで、上段に RMSE、下段に NLPD を配置している。

![予測の誤差（誘導点 512 個）](bench/accuracy_sgpr_matched.svg)

**学習の時間（誘導点 512 個）**

全学習点と同様に対数軸を使い、所要秒数（上段）と尤度・勾配の計算回数（下段）をプロットしている。

![学習の時間（誘導点 512 個）](bench/fit_time_sgpr_matched.svg)

**予測の誤差（ミニバッチ）**

gprx と GPyTorch を同一条件（Adam、学習率 0.01、バッチ 1024、データ 3 周）で比較した結果。GPy にはこの学習形式が無い。上段に RMSE、下段に NLPD を示している。

![予測の誤差（ミニバッチ）](bench/accuracy_svgp_matched.svg)

**学習の時間（ミニバッチ）**

上段が所要秒数、下段が Adam の更新回数。ここでは更新回数が揃っているため、秒数の違いがそのまま速さの差を表す。

![学習の時間（ミニバッチ）](bench/fit_time_svgp_matched.svg)

**メモリの推移（energy、全学習点）**

線はプロセス全体の常駐メモリ。横軸はプロセスが始まってからの秒。点線は、その色のライブラリが学習または予測を始めた時刻。分割は 0 番。

![メモリの推移（energy、全学習点）](bench/rss_timeline_energy_exact_s0_matched.svg)

**メモリの推移（kin40k、誘導点 512 個）**

読み方は energy のメモリの図と同じ。分割は 0 番。

![メモリの推移（kin40k、誘導点 512 個）](bench/rss_timeline_kin40k_sgpr_s0_matched.svg)

**Mauna Loa の予測**

1 枚が 1 ライブラリ。線が予測の平均、帯が 95% 区間。塗った点は学習データ、抜き点はテストデータ。

![Mauna Loa の予測](bench/curve_maunaloa_matched.svg)

**Snelson の予測**

1 枚が 1 ライブラリ。線が予測の平均、帯が 95% 区間。点は学習データ。テスト用の点は無い。

![Snelson の予測](bench/curve_snelson_matched.svg)
<!-- bench:end -->

### 再現

```text
just perf-real-full                                            # 上の比較を取り、この節を作り直す
```

`--timeline` をつけると、プロセス全体の常駐メモリを 10 ms ごとに記録する。生の出力は `compare/perf/out/real/` に残り、コミットしない。`docs/bench/summary.json` には、表の数値、測定した機械、ライブラリの版、最適化の設定が入る。詳細: [`compare/perf/README.md`](https://github.com/YUKIKEDA/gprx/blob/main/compare/perf/README.md)。
