# A Survey of Deep Learning-Based Reconstruction Methods for Quantum Imaging Systems

**PARIMANA Summer Internship Program 2026** | Qmet Tech Foundation

## Author
**Oishik Kar**  
B.Tech, Computer Science and Engineering  
Indian Institute of Information Technology Dharwad

## Project Duration
1 May 2026 – 30 June 2026

## Abstract
This project conducts a systematic survey of deep learning architectures applied to quantum imaging reconstruction, covering quantum ghost imaging, compressed sensing with entangled photons, and sub-shot-noise imaging. The study examines hybrid classical-quantum frameworks for reconstructing images from correlated photon measurements, identifies methodological trends and open problems, and proposes directions for integrating quantum-inspired neural architectures into imaging pipelines.

## Repository Structure

```
parimana/
├── README.md
├── requirements.txt          # Python dependencies
├── src/                      # Source code for experiments
│   ├── ghost_imaging/        # Core physics simulation
│   │   ├── simulator.py      # CGI forward model & traditional recon
│   │   └── dataset.py        # PyTorch dataset generator
│   ├── models/               # DL Architectures
│   │   ├── fcn.py            # Fully connected (Lyu et al. 2017)
│   │   ├── cnn.py            # Convolutional (He et al. 2018)
│   │   └── unet.py           # U-Net architecture
│   ├── train.py              # Training pipeline
│   ├── eval.py               # Evaluation & metrics
│   └── utils.py              # Visualization & helpers
├── progress-report/          # Periodic progress reports
│   ├── progress_report.tex   # Week 3 progress report (current)
│   └── references.bib
├── survey/                   # Main survey document
│   ├── main.tex              # Master document
│   ├── references.bib        # Bibliography
│   └── sections/             # Individual survey sections
├── notes/                    # Research notes and reading log
│   └── reading_log.md
└── figures/                  # Figures and diagrams
```

## Running the Code

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train a model:**
   ```bash
   # Train the CNN model with 98 measurements (12.5% sampling ratio)
   python -m src.train --model cnn --epochs 20 --measurements 98 --precompute
   ```

3. **Evaluate and generate visualizations:**
   ```bash
   # Evaluate the trained CNN
   python -m src.eval --model cnn --weights checkpoints/cnn_best.pth --measurements 98
   ```
   *Evaluation results and sample images will be saved in the `results/` directory.*

## Compilation (LaTeX)

### Progress Report
```bash
cd progress-report
pdflatex progress_report
bibtex progress_report
pdflatex progress_report
pdflatex progress_report
```

### Survey Document
```bash
cd survey
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

## Timeline

| Week | Period | Focus Area | Status |
|------|--------|------------|--------|
| 1 | May 1–7 | Scope definition, literature search | ✅ Complete |
| 2 | May 8–14 | DL for ghost imaging review | ✅ Complete |
| 3 | May 15–21 | Compressed sensing review | 🔄 In Progress |
| 4 | May 22–28 | Sub-shot-noise & hybrid methods | 📋 Planned |
| 5 | May 29–Jun 4 | Hybrid architectures deep dive | 📋 Planned |
| 6 | Jun 5–11 | Synthesis & gap analysis | 📋 Planned |
| 7 | Jun 12–18 | Writing & compilation | 📋 Planned |
| 8 | Jun 19–25 | Review & final submission | 📋 Planned |

## License
This work is submitted in partial fulfillment of the PARIMANA Summer Internship Program 2026.
