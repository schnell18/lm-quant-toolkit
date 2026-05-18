#!/bin/bash
#
TOOLKIT_DIR="../../.."

if [[ ! -d pdfs ]]; then
    mkdir pdfs
fi

$TOOLKIT_DIR/data-vis/gen-table-boost-mxq-llm.R \
  -d data/combined.csv \
  --comparison "sensi-vs-kurt" \
  --experiment "SensiBoost vs KurtBoost"

$TOOLKIT_DIR/data-vis/gen-table-boost-mxq-llm.R \
  -d data/combined.csv \
  --comparison "sensi-vs-ablation" \
  --experiment "SensiBoost vs Ablation"

$TOOLKIT_DIR/data-vis/gen-table-boost-mxq-llm.R \
  -d data/combined.csv \
  --comparison "kurt-vs-ablation" \
  --experiment "KurtBoost vs Ablation"

cat<<EOF > pdfs/Makefile
all:    sensi-vs-kurt sensi-ablation kurt-ablation

sensi-vs-kurt: sensi-vs-kurt-3bit.tex sensi-vs-kurt-4bit.tex
	pdflatex -sync-tex=1 -shell-escape sensi-vs-kurt-3bit.tex
	pdflatex -sync-tex=1 -shell-escape sensi-vs-kurt-4bit.tex

sensi-ablation: sensi-vs-ablation-3bit.tex sensi-vs-ablation-4bit.tex
	pdflatex -sync-tex=1 -shell-escape sensi-vs-ablation-3bit.tex
	pdflatex -sync-tex=1 -shell-escape sensi-vs-ablation-4bit.tex

kurt-ablation: kurt-vs-ablation-3bit.tex kurt-vs-ablation-4bit.tex
	pdflatex -sync-tex=1 -shell-escape kurt-vs-ablation-3bit.tex
	pdflatex -sync-tex=1 -shell-escape kurt-vs-ablation-4bit.tex

clean:
	rm -f *.aux *.log *.xml *.out *.pdf *.bcf *.toc *.blg *.bbl *.dvi *.fls
	rm -fr _minted-report/ *.lof *.lot *.idx *.nlo
	find . -name "*.aux" | xargs rm
EOF

cd pdfs
make
