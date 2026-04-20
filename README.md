# ML-KEM-512 Black-Box AI Cryptanalysis 

## 🔍 Research Question 
Can a deep learning model break **ML-KEM-512 (FIPS 203)** by learning a mapping from:
(encapsulation key ek, ciphertext ct) → shared secret ss
without access to the private key? 

## 🧠 Motivation 
Recent work has shown that neural networks can: 
* Break reduced-round ciphers (Gohr, CRYPTO 2019 — SPECK) 
* Attack weak LWE variants (SALSA, NeurIPS 2022) 
👉 This project tests: **Does this extend to real-world post-quantum cryptography?** 

--- 

## 📊 Dataset 
* **12,000 NIST KAT vectors (ML-KEM-512)** 
* 10,000 training / 2,000 testing 
* Each sample: 
	* Input: ek (800 bytes) + ct (768 bytes) → 1568 bytes 
	* Target: ss (32 bytes) 
📁 Included in repo:
dataset_mlkem512_12000.rsp (~76MB)

## ⚙️ Dataset Generation

This project uses **NIST ML-KEM-512 Known Answer Test (KAT) vectors** generated from the official Kyber reference implementation.

### 🔹 Option 1 — Use Included Dataset 
The dataset is already provided:

dataset_mlkem512_12000.rsp (~76MB)

### 🔹 Option 2 — Reproduce Dataset (From Source)

Clone the official Kyber repository:

```bash
git clone https://github.com/pq-crystals/kyber
cd kyber/ref/nistkat
```
Modify the generator:
```c
#define KATNUM 12000   // default = 100
```
Compile and run:
```bash
gcc -o PQCgenKAT_kem512 PQCgenKAT_kem.c -lcrypto
./PQCgenKAT_kem512
```
Output:
```
PQCkemKAT_1632.rsp
```
---

## 🧠 Model 
**ResNet CNN (Benamira et al., EUROCRYPT 2021)** 
* Input: 1568-byte sequence 
* Architecture: 
	* Conv1D stem 
	* 3 Residual blocks 
	* Global average pooling 
	* Fully connected layers 
* Parameters: ~100K 
* Loss: MSE 
* Optimizer: Adam 
* Epochs: 25 

## 🧪 Experiment Setup 
* Train on 10,000 samples 
* Evaluate on 2,000 unseen samples 
* Additional tests: 
	* Single unseen KAT vector 
	* Completely random input 
	
---

## 📈 Results 
| Metric | Value | 
| ------------------ | -------------------------------------- | 
| Best Test Accuracy | **0.004295** | 
| Random Baseline | **0.003906 (1/256)** | 
| Improvement | **~1.09× (noise level)** | 
| Unseen Input | **1 / 32 bytes correct (coincidence)** | 

## 🔍 Key Observation 
The model converges to predicting:
127 / 128 for every byte
👉 This is the **mean of byte distribution (255/2 ≈ 127.5)** 

## ❗ Critical Insight 
* Model outputs **same prediction for real ciphertext and random input** 
* No dependency on input observed 
* Model ignores (ek, ct) entirely 
👉 This proves: 
> The network learned **no mapping whatsoever** 

--- 

## 🔐 Conclusion 
**ML-KEM-512 is NOT vulnerable to black-box AI cryptanalysis** 
Reason: 
1. **MLWE hardness** → recovering key is computationally infeasible 
2. **SHAKE-256 hashing** → output is pseudorandom 
3. **IND-CCA2 security** → outputs indistinguishable from random 

## 📉 Comparison with Literature 
| Work | Target | Result | 
| -------------------- | ---------------- | ------------------ | 
| Gohr (2019) | SPECK (7 rounds) | 0.618 (broken) | 
| SALSA (2022) | LWE (sparse) | success | 
| **This work (2026)** | ML-KEM-512 | **0.004 (random)** | 

--- 
## ▶️ How to Run
```bash
python3 mlkem_resnet_attack.py
```

## 📁 Project Files 
* mlkem_resnet_attack.py — full experiment 
* dataset_mlkem512_12000.rsp — dataset 
* mlkem512_research_report.docx — detailed report 
* mlkem512_presentation.pptx — presentation 
* attack_report.txt — logs/output 

## 🧠 Key Takeaway 
> Deep learning can exploit **structure** — but ML-KEM has none. 

--- 
## 🚀 Future Work 
* Larger datasets (100k+ samples) 
* Transformer-based models (SALSA-style) 
* Polynomial / NTT-domain representations 
* Side-channel + ML hybrid attacks 

## 📚 References 
* NIST FIPS 203 (ML-KEM) 
* Gohr (CRYPTO 2019) 
* Benamira et al. (EUROCRYPT 2021) 
* SALSA (NeurIPS 2022) 

## 👨‍💻 Author
 Shreehari Menon B.Tech CSE (Design) Research Project — 2026


































