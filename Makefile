.PHONY: book smoke check clean
book:
	latexmk -xelatex hsf_hd.tex
smoke:
	latexmk -xelatex smoke.tex
check:
	python3 scripts/check_manuscript.py
clean:
	latexmk -C hsf_hd.tex smoke.tex
