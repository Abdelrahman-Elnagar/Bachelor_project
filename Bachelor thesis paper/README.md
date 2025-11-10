# Bachelor Thesis - Modular LaTeX Structure

This directory contains the modular LaTeX files for the bachelor thesis.

## File Structure

- `main.tex` - Main document that includes all other files
- `preamble.tex` - All LaTeX packages and setup
- `titlepage.tex` - Title page
- `abstract.tex` - Abstract
- `frontmatter.tex` - Table of contents and lists
- `chapter01_introduction.tex` through `chapter07_conclusion.tex` - Individual chapters
- `appendices.tex` - Appendices
- `bibliography.tex` - References

## Compilation

To compile the complete thesis:
```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Individual Chapter Compilation

Each chapter file can be compiled individually by creating a minimal wrapper:

```latex
\documentclass[11pt,a4paper]{report}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[english]{babel}
\usepackage{amsmath}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{natbib}

\begin{document}
\input{chapter01_introduction}
\end{document}
```

## Notes

- All files are designed to work together when included via `main.tex`
- No extra text has been added - all content is exactly as in the original `bachelor_thesis.tex`
- The modular structure makes it easier to work on individual chapters

