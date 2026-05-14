# Reading Log — DL for Quantum Imaging Survey

## Week 1 (May 1–7, 2026): Scope Definition & Literature Discovery

### Scope
- Defined three primary domains: (1) quantum ghost imaging, (2) compressed sensing with entangled photons, (3) sub-shot-noise imaging
- Identified hybrid classical-quantum reconstruction as cross-cutting theme
- Databases: IEEE Xplore, arXiv (quant-ph, eess.IV, cs.CV), Google Scholar, Optica Publishing

### Papers Identified
| # | Paper | Year | Domain | Priority |
|---|-------|------|--------|----------|
| 1 | Pittman et al., PRA 52 | 1995 | Ghost imaging (foundational) | High |
| 2 | Strekalov et al., PRL 74 | 1995 | Ghost interference | Medium |
| 3 | Bennink et al., PRL 89 | 2002 | Classical ghost imaging | High |
| 4 | Ferri et al., PRL 94 | 2005 | Thermal ghost imaging | Medium |
| 5 | Shapiro, PRA 78 | 2008 | Computational GI | High |
| 6 | Katz et al., APL 95 | 2009 | Compressive GI | High |
| 7 | Erkmen & Shapiro, AOP 2 | 2010 | GI review | High |
| 8 | Brida et al., Nat Photon 4 | 2010 | Sub-shot-noise | High |
| 9 | Morris et al., Nat Commun 6 | 2015 | Few-photon imaging | High |
| 10 | Lyu et al., Sci Rep 7 | 2017 | DL ghost imaging | High |
| 11 | Shimobaba et al., Opt Commun 413 | 2018 | DL ghost imaging | High |
| 12 | He et al., Sci Rep 8 | 2018 | DL ghost imaging | High |
| 13 | Wang et al., Opt Express 27 | 2019 | End-to-end DL GI | High |
| 14 | Barbastathis et al., Optica 6 | 2019 | DL computational imaging | High |
| 15 | Samantaray et al., Light Sci Appl 6 | 2017 | Sub-shot-noise microscope | High |

### Taxonomy Framework
Developed classification along three axes:
1. **Imaging modality**: Ghost imaging / CS quantum / Sub-shot-noise / Hybrid
2. **DL architecture**: CNN / GAN / U-Net / Autoencoder / Quantum-inspired
3. **Reconstruction task**: Denoising / Super-resolution / Compressed reconstruction / Phase retrieval

---

## Week 2 (May 8–14, 2026): Ghost Imaging Deep Dive

### Key Readings & Notes

**Lyu et al. (2017)** — First demonstration of DL-based ghost imaging
- Used a simple fully-connected network to reconstruct 28×28 images
- Achieved high-quality reconstruction at 12.5% sampling ratio (vs ~100% for traditional)
- Limitation: small image sizes, simple architecture

**He et al. (2018)** — Improved DL ghost imaging
- CNN architecture with multiple convolutional layers
- Better generalization than Lyu et al. to unseen object categories
- Demonstrated on both simulation and experimental data

**Shimobaba et al. (2018)** — Computational GI with deep learning
- Applied deep learning to computational ghost imaging specifically
- Showed improvement over traditional correlation-based reconstruction
- Limited analysis of noise robustness

**Wang et al. (2019)** — End-to-end learned GI
- Key paper: jointly optimizes illumination patterns and reconstruction network
- Significant improvement: learns optimal measurement strategy
- End-to-end approach achieves PSNR improvements of 3-5 dB over separate optimization

### Emerging Themes
1. DL enables 10-100x reduction in required measurements
2. Shift from hand-crafted to learned illumination patterns
3. GAN-based methods show better perceptual quality but may introduce artifacts
4. Limited standardized benchmarks across papers

---

## Week 3 (May 15–20, 2026): Compressed Sensing & Hybrid Methods

### Key Readings & Notes

**Katz et al. (2009)** — Compressive ghost imaging
- First application of CS theory to ghost imaging
- Sparsity assumption enables sub-Nyquist sampling
- Foundation for later DL-enhanced CS methods

**Donoho (2006) / Candès & Tao (2006)** — CS fundamentals
- Reviewed RIP conditions and their relevance to quantum measurements
- Quantum correlations provide natural incoherence properties useful for CS

**Morris et al. (2015)** — Few-photon imaging
- Demonstrated imaging with extremely low photon counts using entanglement
- Relevant as motivation for DL denoising in quantum imaging

### In Progress
- Reading Beer et al. (2020) on quantum neural networks
- Reviewing Killoran et al. (2019) on continuous-variable quantum NNs
- Cross-referencing hybrid classical-quantum frameworks
