# Thesis Source

This directory contains a clean copy of the Overleaf source used for the thesis:

> **AI-Powered Crop Disease Detection System for Sustainable Agriculture in Bangladesh**

The compiled final document is available at [`../docs/AI_Powered_Crop_Disease_Detection_Thesis.pdf`](../docs/AI_Powered_Crop_Disease_Detection_Thesis.pdf).

## Compile

Run the following commands from this directory in a LaTeX environment containing the packages referenced by `main.tex`:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The source was originally maintained in Overleaf. Generated build files are intentionally excluded from the repository.

