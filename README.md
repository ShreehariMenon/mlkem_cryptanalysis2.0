# ML-KEM-512 Black-Box AI Cryptanalysis

## Research Question
Can an AI model trained on NIST KAT vectors predict the shared secret
of ML-KEM-512 from the public key and ciphertext?

## Dataset
- 12,000 NIST KAT vectors for ML-KEM-512 (FIPS 203)
- Generated from official pq-crystals/kyber reference implementation
- 10,000 training / 2,000 test
- File: dataset_mlkem512_12000.rsp

## Model
ResNet CNN (Benamira et al., EUROCRYPT 2021)
- 100,384 parameters
- 25 epochs, Adam optimizer, MSE loss

## Result
Best test byte accuracy: 0.004295
Random baseline (1/256): 0.003906
Bytes correct on unseen input: 0/32

**ML-KEM-512 is NOT VULNERABLE to black-box AI attack.**

## Files
- mlkem_resnet_attack.py — full experiment script
- dataset_mlkem512_12000.rsp — NIST KAT vectors
