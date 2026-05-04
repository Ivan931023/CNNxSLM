# 📝 完整對話紀錄：Zernike CNN 優化之旅
> 專案目錄：`/Users/ivanchen/Downloads/cnn_SLM_project/Simulated_flattop/`
> 時間：2026-04-25 ~ 2026-05-01

---

## 第一階段：專案建立與初始訓練 (2026-04-25)

### 🔧 專案建構
- 建立了完整的 CNN 訓練管線（Pipeline），包含：
  - `dataset_generator.py`：利用傅立葉光學模擬產生 50,000 張 Zernike 干擾平頂光圖片
  - `train_flattop_cnn.py`：VGG-style 4 層 CNN 訓練腳本
  - `app.py`：Streamlit 網頁推論介面
  - `run_pipeline.sh`：一鍵自動化腳本
- 使用 `tensors.mat`（MATLAB 產生的光學基底）作為物理引擎

### 📊 初始訓練結果
- 架構：4 層 VGG-style CNN，128×128 輸入
- 訓練 10 Epochs，50,000 張圖片
- **問題**：Validation Loss 劇烈震盪（0.87 → 2.75 → 0.46 → 2.60）
- 最終 Best Val Loss：`0.2555`（但不穩定）

---

## 第二階段：修復震盪問題 (2026-04-28 凌晨)

### 🛠️ 用戶提問
**Q：最後的結果怎麼會 validation 表現得不好 該怎麼辦？**

### 💡 診斷與修復
- **原因**：學習率過高 + 缺乏正則化
- **修復方案**：
  1. 加入 **L2 正則化** (`weight_decay=1e-4`)
  2. 加入 **CosineAnnealingLR** 學習率退火排程器
  3. 訓練 Epochs 從 10 增加到 20
- 重新訓練後，Val Loss 穩定收斂，Best Val Loss 降至 `0.1180`（進步 54%）

---

## 第三階段：技術問答 (2026-04-28 凌晨)

### 🎓 用戶技術問題與解答

**Q：ReLU & BatchNorm 的作用？**
- ReLU：非線性轉換器，f(x)=max(0,x)，過濾負值信號
- BatchNorm：訓練穩定器，將每批數據正規化為均值 0、標準差 1

**Q：哪個先？**
- 標準順序：Conv2d → BatchNorm2d → ReLU
- 先 BatchNorm 讓數據以 0 為中心呈常態分佈，再用 ReLU 切掉負值

**Q：沒有範圍限制是什麼意思？**
- 最後一層用 `nn.Linear`（純線性輸出），不加 Sigmoid/Softmax
- 因為 Zernike 係數可以是任意實數（正負皆可），不受 0~1 範圍限制
- Sigmoid 適合分類（機率），Linear 適合回歸（預測連續數值）

---

## 第四階段：App 顯示修復 (2026-04-28 凌晨)

### 🎨 用戶反映
**Q：app 的顯示圖為何最大跟最小都是藍色啊，能不能跟 MATLAB 出來的圖長一樣？**

### 🔍 原因
- 上傳的圖已經是 MATLAB `turbo` 偽色彩的彩色圖
- App 把彩色圖強制轉灰階後又重新套一次 turbo，導致顏色錯亂

### ✅ 修復
- 分離「顯示用圖片」(RGB 原色) 與「模型輸入」(灰階張量)
- 顯示原始色彩，灰階僅用於模型推論

---

## 第五階段：理想光斑測試問題 (2026-04-28 凌晨)

### 🔬 用戶測試
**Q：我放一個理想的圖（應該全部為零）但預測結果卻不對？**

### 🔍 原因
- 用戶上傳的是 MATLAB 「圖表(Plot)」截圖（含標題、座標軸、Colorbar）
- CNN 訓練時從未見過這些額外元素，造成 Domain Gap

### ✅ 解決方案
- 撰寫 `generate_ideal.py`：產生純淨的 500×500 理想光斑圖片（無邊框、無文字）
- 使用此純淨圖測試，結果顯著改善但仍有微小偏差

---

## 第六階段：終極優化計畫 - ResNet-18 升級 (2026-04-28 下午)

### 💡 用戶需求
**Q：結果還是有點偏差，你有什麼改進的想法嗎？能不能都使用使達到最佳？**

### 🚀 終極優化計畫（第一版）
1. **大腦升級**：VGG-4 → **ResNet-18** (18 層殘差網路)
2. **課程學習 (Curriculum Learning)**：
   - Phase 1：50,000 張 [-2.0, 2.0] 大範圍粗調 (SmoothL1Loss)
   - Phase 2：生成 5,100 張微調資料（含 100 張完美 0 錨點）
   - Phase 3：純 **L1 Loss** 極限微調 15 Epochs（稀疏促進，強制歸零）
3. 自動化腳本 `run_ultimate_training.sh` 一鍵三階段訓練

### 📊 ResNet-18 訓練結果
- Phase 1 完成：Val Loss 穩定收斂
- Phase 3 完成：**最終 Val MAE = `0.0093`** ⚡
- 相比之前 0.1~0.4 的誤差，提升了 **10~40 倍精度**

---

## 第七階段：自訂 Zernike 生成工具 (2026-04-29 凌晨)

### 🔧 用戶需求
**Q：幫我寫類似 generate_ideal.py 的檔案但是我可以輸入調整我想要的 Zernike mode**

### ✅ 產出
- 撰寫 `generate_custom_zernike.py`：CLI 工具，可自由調整 Z1~Z15
- 使用方式：
  ```bash
  python generate_custom_zernike.py --Z4 1.5 --Z5 -0.5 --output my_test.png
  ```

---

## 第八階段：Hyper-Precision 終極升級 (2026-04-29 凌晨~2026-05-01)

### 💡 用戶需求
**Q：還能訓練得更加精細精準嗎？**

### 🌟 Hyper-Precision 計畫
- **解除解析度封印**：128×128 → **500×500**（像素量增加 15 倍）
- **升級骨幹網路**：ResNet-18 → **ResNet-34**（更深、感受野更大）
- **擴增微調資料**：5,100 → **20,100 張**
- **Batch Size**：64 → **16**（避免記憶體溢出）

### 📊 Hyper-Precision 訓練進度 (截至 2026-05-01)
- Phase 1（粗調 20 Epochs）：✅ 完成，Val MAE 降至 0.0820
- Phase 2（生成 20,100 張微調資料）：✅ 完成（花費約 19 小時）
- Phase 3（L1 微調 15 Epochs）：🔄 進行中（Epoch 5/15，Val MAE = 0.0160）

### 🔀 App 分離
- `app.py`（舊版）：ResNet-18 / 128×128，載入 `best_resnet_finetuned_old.pth`
- `app_v2.py`（新版）：ResNet-34 / 500×500，載入 `best_resnet_finetuned.pth`

---

## 第九階段：模型改進分析 (2026-05-01)

### 💡 用戶提問
**Q：你覺得這個模型還有改進的空間嗎？**

### 🔮 未來改進方向（按優先順序）
1. ⭐ **Data Augmentation**：隨機旋轉、翻轉、微小位移
2. ⭐ **Attention 機制**：CBAM / SE-Net，讓模型學會「看哪裡最重要」
3. ⭐ **Per-Mode 加權損失**：高階 Zernike 給更高的 loss 權重
4. ⚡ **Test-Time Augmentation (TTA)**：推論時多次增強取平均
5. ⚡ **Ensemble（模型集成）**：多模型平均
6. 🧪 **Physics-Informed Input**：加入徑向/角度座標作為額外輸入通道
7. 🧪 **Vision Transformer (ViT)**：用 Transformer 取代 CNN

---

## 📁 重要檔案清單

| 檔案 | 用途 |
|---|---|
| `train_flattop_cnn.py` | 主訓練腳本（ResNet-34, 500×500） |
| `finetune_flattop_cnn.py` | L1 Loss 微調腳本 |
| `dataset_generator.py` | 50,000 張訓練資料生成器 |
| `dataset_generator_finetune.py` | 20,100 張微調資料生成器 |
| `generate_ideal.py` | 理想平頂光圖片生成器 |
| `generate_custom_zernike.py` | 自訂 Zernike 係數圖片生成器 |
| `app.py` | Streamlit App（舊版 ResNet-18） |
| `app_v2.py` | Streamlit App（新版 ResNet-34） |
| `run_ultimate_training.sh` | 三階段自動化訓練腳本 |
| `run_pipeline.sh` | 初始版一鍵訓練腳本 |

## 📈 模型演進史

| 版本 | 架構 | 解析度 | 微調資料 | Best Val MAE |
|---|---|---|---|---|
| v1 | VGG-4 | 128×128 | 無 | ~0.5876（震盪嚴重） |
| v2 | VGG-4 + 退火 | 128×128 | 無 | ~0.1180 |
| v3 | ResNet-18 | 128×128 | 5,100 | **0.0093** |
| v4 | ResNet-34 | 500×500 | 20,100 | 訓練中（目前 ~0.016） |
