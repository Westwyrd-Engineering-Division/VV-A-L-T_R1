# Makefile for V.V.A.L.T Whitepaper

# Main document name (without extension)
MAIN = whitepaper

# LaTeX compiler
LATEX = pdflatex
BIBTEX = bibtex

# Compiler flags
LATEX_FLAGS = -interaction=nonstopmode -halt-on-error

# Output PDF
PDF = $(MAIN).pdf

# Auxiliary files
AUX_FILES = *.aux *.log *.bbl *.blg *.out *.toc *.lof *.lot *.fls *.fdb_latexmk *.synctex.gz

.PHONY: all clean distclean view

# Default target
all: $(PDF)

# Build the PDF (full compilation with bibliography)
$(PDF): $(MAIN).tex references.bib
	@echo "=== First LaTeX pass ==="
	$(LATEX) $(LATEX_FLAGS) $(MAIN).tex
	@echo "=== Running BibTeX ==="
	$(BIBTEX) $(MAIN)
	@echo "=== Second LaTeX pass ==="
	$(LATEX) $(LATEX_FLAGS) $(MAIN).tex
	@echo "=== Third LaTeX pass ==="
	$(LATEX) $(LATEX_FLAGS) $(MAIN).tex
	@echo "=== PDF generation complete: $(PDF) ==="

# Quick compilation (no bibliography update)
quick: $(MAIN).tex
	@echo "=== Quick compilation ==="
	$(LATEX) $(LATEX_FLAGS) $(MAIN).tex

# Clean auxiliary files
clean:
	@echo "=== Cleaning auxiliary files ==="
	rm -f $(AUX_FILES)

# Clean everything including PDF
distclean: clean
	@echo "=== Removing PDF ==="
	rm -f $(PDF)

# View the PDF (Linux)
view: $(PDF)
	@if command -v xdg-open > /dev/null; then \
		xdg-open $(PDF); \
	elif command -v evince > /dev/null; then \
		evince $(PDF); \
	elif command -v okular > /dev/null; then \
		okular $(PDF); \
	else \
		echo "No PDF viewer found. Please open $(PDF) manually."; \
	fi

# Help message
help:
	@echo "V.V.A.L.T Whitepaper Makefile"
	@echo ""
	@echo "Usage:"
	@echo "  make          - Compile the whitepaper (full build with bibliography)"
	@echo "  make quick    - Quick compilation (no bibliography update)"
	@echo "  make clean    - Remove auxiliary files"
	@echo "  make distclean - Remove all generated files including PDF"
	@echo "  make view     - Open the PDF in a viewer (Linux)"
	@echo "  make help     - Show this help message"
