# Punctured polar codes in the AIR information-reconciliation loop: a negative result

## Scope

This directory contains scripts and brief analysis about incompatability
between puncturing polar code and current information reconciliation protocol.
We use synthetic data over short block lengths ($N = 512$ and $N = 1024$).
The polar-code reliability tables are regenerated on the fly with
`polar-fbgen-rust`, and every quantity reported below is produced by the
scripts in this directory.

## Background

* Alice's sifted key is padded to $N = 2^n$ with the public constant 0, encoded,
  and only the values of the *frozen* codeword positions are revealed to Bob.
  The decoder is fed Bob's (noisy) padded key as the channel input.
* In this wiring the polar tree is used in a transposed way: the LLR input
  (the tree root, where `_load` writes `Y_N`) carries the **U side** (the
  padded key), while the hard decisions live on the **leaves**, where
  `frozen_bits` / `F_N` act and which carry the **X side** (the codeword
  $X = U G$).
* Both BGL trimming branches pad Alice's key at the trimmed positions, and
  they differ in what the *codeword* does there:
  * **shortening** (bit-reversal suffix $\mathcal{S}$): the construction forces
    $X_{\mathcal{S}} = 0$, so those codeword bits are known to both parties and
    the decoder can hard-freeze the corresponding leaves;
  * **puncturing** (bit-reversal prefix $\mathcal{P}$): $X_{\mathcal{P}}$ is *not*
    zero and is never revealed; the only knowledge is that the padded **input**
    is zero, and that fact lives on the observation (root) side.

## Why the puncturing branch cannot be repaired inside this decoder

1. The decoder's only hard-constraint mechanism acts on the **leaves**.
   The padding is a **root-side** fact, so it can only be conveyed as a channel
   LLR — there is no way to force $U_i = 0$ for $i \in \mathcal{P}$.
2. The SC/SCL recursion combines observations with the min approximation
   ($f(a,b) = \mathrm{sign}(a)\,\mathrm{sign}(b)\,\min(|a|,|b|)$), so a large
   LLR at one root position is diluted by its unreliable neighbours. It is
   *evidence*, not a *constraint*.
3. Numerically, making that evidence stronger does not help: it hurts
   (Table 2), and it saturates.
4. Consequently the punctured leaves stay effectively unconstrained: the codec
   carries $|\mathcal{P}|$ extra free dimensions. Increasing the SCL list does
   not remove them (Table 1).

## Numerical results (synthetic)

### Table 1 — trimming branch comparison

`key FER` is the probability that the reconstructed key differs from Alice's,
i.e. $(VG)[\text{non-sp}] \neq U[\text{non-sp}]$; `leaf FER` additionally
requires the whole codeword $V = X$. `union bound` is the BGL estimate over the
information positions.

| n | N | set | trim | BER | ub target | SCL | LLR at trimmed positions | K | union bound | key FER | leaf (codeword) FER |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 9 | 512 | punct | 205 | 0.12 | 1e-02 | 8 | amplify | 273 | 0.00918 | **0.5467** | 0.6287 |
| 9 | 512 | punct | 205 | 0.12 | 1e-02 | 8 | erase | 273 | 0.00918 | **0.0080** | 1.0000 |
| 9 | 512 | punct | 205 | 0.12 | 1e-02 | 32 | erase | 273 | 0.00918 | **0.0080** | 1.0000 |
| 9 | 512 | short | 205 | 0.12 | 1e-02 | 8 | amplify | 69 | 0.00879 | **0.0000** | 0.0000 |
| 9 | 512 | short | 205 | 0.12 | 1e-02 | 32 | amplify | 69 | 0.00879 | **0.0000** | 0.0000 |
| 10 | 1024 | punct | 358 | 0.12 | 1e-02 | 8 | amplify | 519 | 0.00956 | **0.8030** | 0.8807 |
| 10 | 1024 | punct | 358 | 0.12 | 1e-02 | 8 | erase | 519 | 0.00956 | **0.0087** | 1.0000 |
| 10 | 1024 | short | 358 | 0.12 | 1e-02 | 8 | amplify | 166 | 0.00978 | **0.0017** | 0.0017 |
| 10 | 1024 | punct | 358 | 0.12 | 1e-03 | 8 | erase | 494 | 0.000942 | **0.0003** | 1.0000 |
| 10 | 1024 | punct | 358 | 0.12 | 1e-03 | 32 | erase | 494 | 0.000942 | **0.0003** | 1.0000 |
| 10 | 1024 | short | 358 | 0.12 | 1e-03 | 8 | amplify | 141 | 0.000984 | **0.0003** | 0.0003 |
| 10 | 1024 | short | 358 | 0.12 | 1e-03 | 32 | amplify | 141 | 0.000984 | **0.0003** | 0.0003 |

### Table 2 — LLR magnitude at the punctured positions

Same case as the $N = 1024$, $|\mathcal{P}| = 358$ row of Table 1, varying only
the LLR applied at the punctured positions.

| LLR at punctured positions | key FER |
|---|---|
| 0 (erasure) | **0.0090** |
| 1x (raw) | **0.2290** |
| 17x | **0.8030** |
| 1e3x | **0.8560** |
| 1e6x | **0.8560** |

## Reproduction

```bash
# Table 1
python synthetic_study.py --frames 3000 --out results.md
# Table 2
python llr_sweep.py > llr_sweep.md
```

Requires the py_aff3ct bindings and `polar-fbgen-rust`; both paths can be
overridden with the `PY_AFF3CT_LIB` and `POLAR_FBGEN` environment variables.

## Conclusion

We also explored the use of punctured polar codes for information
reconciliation. Both the theoretical analysis and the numerical results show
that puncturing is **incompatible with the current IR protocol**: the zero
padding that puncturing relies on is only expressible on the *observation* side
of this decoder, where it cannot be enforced as a constraint, so the punctured
leaves remain unconstrained: under the LLR treatment the decoder actually
applies at those positions (raw or amplified, Table 2) reconciliation fails
even when the union bound over the information positions alone is met, and no
finite LLR amplification can repair it, because amplification is evidence
rather than a constraint.

The shortening branch does not have this problem: the shortened codeword bits
are zero by construction, so the same padding is an exact hard constraint on
the decision side. This is why both codecs of the released configuration
(`CHSH` and `ZZ keygen`) use pure shortening.