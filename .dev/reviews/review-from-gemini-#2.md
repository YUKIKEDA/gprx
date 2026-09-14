全体として前回のレビュー指摘（ノイズとJitterの分離、混合精度の適用範囲限定、動的Plan生成によるバッファ決定など）が的確に反映されており、**非常に高い解像度で実用性と高速性を両立させようとしている秀逸な設計書**です。

その上で、実際にRustで実装を進めた際に**確実に壁に突き当たる「数理的計算量とバッファの不整合」「faerの実APIとのギャップ」「型・前処理の欠落」**に関して、優先度別（P0〜P2）にレビューと改善提案をまとめました。

---

## 【P0: 致命的】実装着手前に設計修正が必要な事項

### 1. MLL勾配計算におけるトレース項 $\operatorname{Tr}(K^{-1} \frac{\partial K}{\partial \theta})$ と Workspace の欠落

ドキュメントでは `value_and_gradient_into` をアロケーションフリーにする方針が示されていますが、**ハイパーパラメータ勾配の計算アルゴリズムと必要なメモリ**が設計から抜け落ちています。

- **数理的背景**:
  周辺対数尤度（MLL）のカーネルパラメータ $\theta_i$ に関する勾配は以下です。
  $$\frac{\partial \log p(y|\theta)}{\partial \theta_i} = \frac{1}{2} \left[ \alpha^T \frac{\partial K}{\partial \theta_i} \alpha - \operatorname{Tr}\left( K^{-1} \frac{\partial K}{\partial \theta_i} \right) \right] \quad (\alpha = K^{-1}y)$$
  第1項はベクトル積のため $O(n^2)$ で計算できますが、第2項のトレース項 $\operatorname{Tr}(K^{-1} \frac{\partial K}{\partial \theta_i})$ は、直接計算すると $K^{-1}$ 全体（陽な逆行列）を求める必要があり、**$O(n^3)$ の計算コストと追加の $n \times n$ 行列領域**を消費します。
- **現状の設計の不整合**:
  §7.1 の `Workspace` には `k_matrix`（$L$ で上書きされる）, `dist_cache`, `exp_buf` しか存在せず、**$K^{-1}$ または $W = \alpha \alpha^T - K^{-1}$ を保持するバッファがありません**。これがないと、勾配計算ループで一時行列を確保してアロケーション方針に違反するか、パラメータごとに線形ソルブを繰り返して $O(p \cdot n^3)$（$p$ はパラメータ数）に爆発します。
- **対策**:
  1. `Workspace` に $K^{-1}$（または $W = \alpha \alpha^T - K^{-1}$）を保持する $n \times n$ バッファ `inv_k_matrix: Mat<P::Storage>` を明示的に追加する。
  2. $L$ から $K^{-1}$ をインプレース逆変換（Cholesky factor inversion: $L^{-T} L^{-1}$）で計算し、$W$ を一度だけ構築してから、各パラメータ $\theta_i$ に対して $\sum_{j,k} W_{jk} \left(\frac{\partial K}{\partial \theta_i}\right)_{jk}$（要素ごとの内積、$O(n^2)$）で回す標準アルゴリズムを明記する。

---

### 2. faer の Cholesky 増分更新 API の実態と「削除」の実現性

§3 および §11 で `llt::update::{insert,delete}_rows_and_cols_clobber` の使用が前提とされています。

- **問題点**:
  現在の faer（0.19〜0.24系）の公開APIにおいて、Cholesky の更新モジュール（`faer::linalg::cholesky::llt::update` 等）に提供されているのは主に **`rank_one_update`（および `rank_r_update`）** であり、**任意の行・列の動的挿入・削除を行う専用高水準APIは標準提供されていない可能性が極めて高い**です。
- **実装への影響**:
  - **1点追加 (Insert)**: §11 に記載の通り $L v = k$ の前進消去と $d = \sqrt{k_{new} - v^T v}$ の計算（$O(n^2)$）で自前実装が容易です。
  - **1点削除 (Delete)**: 行列の途中から行・列を削除した場合、下三角行列 $L$ の三角形状が崩れます。これを $O(n^2)$ で回復するには、**Givens 回転（または Householder 変換）を用いてヘッセンベルグ形式から三角行列へ逐次復元するアルゴリズム**を自前で実装する必要があります。
- **対策**:
  - §14(4) の「API検証」を待たずに、「**faer に直接の行・列削除 API がない場合は、自前で Givens 回転ベースの Cholesky Downdate を実装する**（または Phase 3 までは末尾削除のみサポートするか、フル再計算にフォールバックする）」という現実的な設計方針に改めるべきです。

---

## 【P1: 重要】設計・数理の整合性改善

### 3. ターゲット変数 $y$ の前処理（逆変換）の欠落

§5.5 で入力 $X$ に対する `Transform` パイプラインが定義されていますが、$y$（目的変数）の前処理が考慮されていません。

- **問題点**:
  GPRでは、平均関数（Mean Function）を明示的に持たない場合、**$y$ を平均 0・分散 1 に標準化（Standardize）することが数値安定性と推論精度の生命線**になります。
  また、$y$ を前処理した場合、予測時（`predict`）に得られた平均値と分散を元のスケールに逆変換（`untransform_mean`, `untransform_variance`）しなければユーザーに正しい予測値を返せません。
- **対策**:
  `Transform` トレイトを $X$ 用と $y$ 用で明確に分けるか、$y$ 用のターゲット変換器をモデルに組み込む設計を追加してください。
  ```rust
  trait TargetTransform<T: Scalar>: Send + Sync {
      fn fit(&mut self, y: &[T]);
      fn transform(&self, y: &mut [T]);
      fn inverse_transform_mean(&self, mean: &mut [T]);
      fn inverse_transform_variance(&self, var: &mut [T]);
  }
  ```

---

### 4. 混合精度反復改良における悪条件行列と Jitter のジレンマ

§4.0 および §4.2 で「$A$ は jitter を含まない真のモデル行列」「反復改良のリトライ時は jitter のみを増やす」とされています。

- **数理的ジレンマ**:
  反復改良（Iterative Refinement）は、方程式 $A x = y$ を解くために前処理行列（ここでは $L L^T \approx A$）を使って残差 $r = y - A x$ を補正していく手法です。
  もし真のモデル行列 $A$ の条件数 $\kappa(A)$ が悪すぎて反復改良が収束しない場合、**「分解側（$L$）の jitter だけを増やす」と、前処理行列 $L L^T$ と解きたい行列 $A$ の乖離が拡大し、反復改良の縮小率 $\approx \|I - (L L^T)^{-1} A\|$ が 1 を超えて発散します**。
- **対策**:
  Jitter を増やす場合は、反復改良のターゲット方程式そのものも $A_{jitter} = A + \epsilon I$ にシフトさせる（＝真のモデル自体に微小正則化を許容する）か、あるいは設計書にある `RefinementFallback::FallbackToDoublePrecision`（f64への完全フォールバック）を第一選択とするのが数値解析的に安全です。

---

### 5. カーネル対称性の活用と `KernelTerm::apply` の引数

§7.2 に「対称性を利用し上三角/下三角のみ計算」と記載されていますが、§5.1 の `KernelTerm` インターフェースにそれが反映されていません。

- **問題点**:
  ```rust
  fn apply(&self, dist: MatRef<T>, out: MatMut<T>);
  ```
  このシグネチャでは、実装者が「全要素を埋めるべきか、下三角（Lower）のみを埋めればよいのか」が曖昧です。faer の `cholesky_in_place` は下三角（または指定した三角側）のみを参照するため、全要素を愚直に計算すると**ホットパスであるカーネル評価の計算時間が厳密に2倍無駄になります**。
- **対策**:
  評価対象の三角成分を指定可能にするか、下三角のみ書き込む契約を型で強制すべきです。
  ```rust
  enum Triangle { Lower, Upper, Full }
  fn apply(&self, dist: MatRef<T>, out: MatMut<T>, uplo: Triangle);
  ```

---

## 【P2: 改善】Rust 型安全性・拡張性

### 6. Rayon 並列化と `Objective` の `&mut self` 借用衝突

- **課題**:
  ```rust
  trait Objective<T: Scalar> {
      fn value_and_gradient_into(&mut self, params: &[T], out: &mut [T]) -> Result<T, GpError>;
  }
  ```
  `ExactGP` の実装内部で、カーネル行列構築や距離計算を Rayon で並列化する際、`&mut self`（特に `Workspace`）をクロージャに渡そうとすると Rust の借用チェッカー（`cannot borrow self as mutable more than once`）に引っかかります。
- **対策**:
  §7.1 の `thread_scratch: Vec<Mat<P::Storage>>` をそのまま使う場合、並列領域に入る直前に `let scratches = &mut self.workspace.thread_scratch[..];` のようにローカルにスライスとして取り出し、Rayon の `par_chunks_mut` や zip で各ワーカースレッドに分配するイディオムを明記しておくと実装時の手戻りを防げます。

### 7. Sparse GP における誘導点 $Z$ の最適化境界

§6.1 で誘導点勾配 `grad_wrt_coords` に触れられていますが、`Objective` との接続が未定義です。
- 誘導点 $Z$（サイズ $m \times d$）をハイパーパラメータ $\theta$ と同時に最適化する場合、`params: &[T]` の末尾にフラット化して結合するのか、それともカーネルハイパラと $Z$ を交互に最適化（Alternate Optimization）するのかを決定しておく必要があります。
- 同時最適化する場合、パラメータ数が跳ね上がるため、L-BFGS のメモリ使用量や収束性に大きく影響します。初期実装（Phase 4）では「**誘導点 $Z$ は k-means 等で固定し、最適化対象はカーネルハイパラのみとする**」と明記してスコープを絞ることを推奨します。

---

## レビューまとめ

| 項目 | 重要度 | 影響範囲 | 提言内容 |
| :--- | :---: | :--- | :--- |
| **MLL勾配バッファ** | **P0** | メモリ・性能 | `Workspace` に $K^{-1}$ または $W$ 行列用バッファを追加し、勾配計算アルゴリズムを明記 |
| **Cholesky削除API** | **P0** | オンライン学習 | faer 単体での任意行削除は期待せず、Givens回転の実装計画を明記 |
| **ターゲット $y$ の前処理** | **P1** | 推論精度 | $y$ の標準化および予測時の `untransform` パイプラインを追加 |
| **反復改良とJitter** | **P1** | 数値安定性 | Jitter増加時は方程式側も変更するか、f64フォールバックを優先する |
| **カーネルの三角評価** | **P1** | 性能 (2倍速) | `KernelTerm::apply` に下三角のみ計算する `uplo` 制御を導入 |
| **誘導点 $Z$ の最適化** | **P2** | Sparse GP | Phase 4 では $Z$ 固定とし、ハイパラ同時最適化は分離 |

これらが設計書に反映されれば、Phase 1（Exact GP）から手戻りなく、理論的にも実装的にも極めて堅牢なライブラリが構築できると考えます。