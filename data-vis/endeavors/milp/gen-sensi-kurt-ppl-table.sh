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

$TOOLKIT_DIR/data-vis/gen-table-milp-mxq-llm.R \
  -d data/combined.csv \
  --comparison "sensi-vs-ablation" \
  --experiment "SensiMiLP vs Ablation"

$TOOLKIT_DIR/data-vis/gen-table-milp-mxq-llm.R \
  -d data/combined.csv \
  --comparison "kurt-vs-ablation" \
  --experiment "KurtMiLP vs Ablation"

cat<<EOF > pdfs/Makefile
all:    sensi-vs-kurt-4bit sensi-vs-kurt-3bit sensi-vs-kurt-others sensi-abl kurt-abl

sensi-vs-kurt-4bit: sensi-vs-kurt-4bit.tex
	pdflatex -sync-tex=1 -shell-escape sensi-vs-kurt-4bit.tex
	pdflatex -sync-tex=1 -shell-escape sensi-vs-kurt-4bit.tex

sensi-vs-kurt-3bit: sensi-vs-kurt-3bit.tex
	pdflatex -sync-tex=1 -shell-escape sensi-vs-kurt-3bit.tex
	pdflatex -sync-tex=1 -shell-escape sensi-vs-kurt-3bit.tex

sensi-vs-kurt-others: sensi-vs-kurt-others.tex
	pdflatex -sync-tex=1 -shell-escape sensi-vs-kurt-others.tex
	pdflatex -sync-tex=1 -shell-escape sensi-vs-kurt-others.tex

sensi-abl: sensi-vs-ablation.tex
	pdflatex -sync-tex=1 -shell-escape sensi-vs-ablation.tex
	pdflatex -sync-tex=1 -shell-escape sensi-vs-ablation.tex

kurt-abl: kurt-vs-ablation.tex
	pdflatex -sync-tex=1 -shell-escape kurt-vs-ablation.tex
	pdflatex -sync-tex=1 -shell-escape kurt-vs-ablation.tex

clean:
	rm -f *.aux *.log *.xml *.out *.pdf *.bcf *.toc *.blg *.bbl *.dvi *.fls
	rm -fr _minted-report/ *.lof *.lot *.idx *.nlo
	find . -name "*.aux" | xargs rm
EOF

cd pdfs
make
