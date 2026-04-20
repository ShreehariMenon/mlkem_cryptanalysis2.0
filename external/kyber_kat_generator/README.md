# Kyber KAT Generator (Modified)

This is a modified version of the Kyber reference KAT generator.

## Change made

```c
#define KATNUM 12000
```
(Default was 100)

## Compile
```bash
gcc -o PQCgenKAT_kem512 PQCgenKAT_kem.c -lcrypto
```

## Run
```bash
./PQCgenKAT_kem512
```

## Output
```
PQCkemKAT_1632.rsp
```
This file is used as the dataset for the ML-KEM AI cryptanalysis experiment.
