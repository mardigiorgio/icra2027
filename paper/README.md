# CENIC-on-GPU paper package (ICRA / IEEE conference template)

Overleaf: New Project -> Upload Project -> this zip. Compiler pdfLaTeX. Main file: `main.tex`.

- `conference_101719.tex` -- the IEEE conference LaTeX template, verbatim (the file behind https://www.ieee.org/conferences/publishing/templates; obtained from Overleaf's official mirror of it). Its sections ARE the formatting guidance. Compiles on its own (set it as main file to see it).
- `main.tex` -- the same template with ONLY the example sections ("Ease of Use" ... "Figures and Tables") replaced by the outline headings from CLAUDE.md and figure slots for the committed Part-1 figures. Title/author/abstract/keywords blocks, Acknowledgment, References guidance and the example bibliography are the template's, untouched.
- `IEEEtran.cls` -- the class (V1.8b, the version the IEEE zip bundles; Overleaf also has it built in).
- `fig1.png` -- PLACEHOLDER for the template's example figure (the IEEE zip ships the real one; only referenced by conference_101719.tex).
- `figures/` -- the committed Part-1 figures (newton-adaptive/scripts/bench/results/figures, 2026-08-29).
- `references.bib` -- references the outline will need (check each). To use it instead of the template's thebibliography: \bibliographystyle{IEEEtran} \bibliography{references}.
- `CENIC_working_notebook.docx` -- the working notebook (prompts, evidence pointers, fact bank, decisions).
- `template_extras/` -- IEEEtran package extras: IEEEtran_HOWTO.pdf (class documentation), IEEEtran.bst, IEEEabrv.bib, bare_conf.tex.
