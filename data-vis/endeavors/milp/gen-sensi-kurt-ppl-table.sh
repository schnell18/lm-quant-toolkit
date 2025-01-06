#!/bin/bash
#
TOOLKIT_DIR="../../.."

if [[ ! -d pdfs ]]; then
    mkdir pdfs
fi

$TOOLKIT_DIR/data-vis/gen-table-milp-mxq-llm.R \
  -d data/combined.csv \
  --comparison "sensi-vs-kurt" \
  --experiment "SensiMiLP vs KurtMiLP"

cat<<EOF > pdfs/Makefile
all:    sensi-vs-kurt

sensi-vs-kurt: sensi-vs-kurt.tex
	pdflatex -sync-tex=1 -shell-escape sensi-vs-kurt.tex
	pdflatex -sync-tex=1 -shell-escape sensi-vs-kurt.tex

clean:
	rm -f *.aux *.log *.xml *.out *.pdf *.bcf *.toc *.blg *.bbl *.dvi *.fls
	rm -fr _minted-report/ *.lof *.lot *.idx *.nlo
	find . -name "*.aux" | xargs rm
EOF

cd pdfs
make
