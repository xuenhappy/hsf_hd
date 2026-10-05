.PHONY: book smoke rewrite-review check clean
book:
	latexmk -xelatex hsf_hd.tex
smoke:
	latexmk -xelatex smoke.tex
rewrite-review:
	latexmk -xelatex rewrite-review.tex
	python3 scripts/check_rewrite_layout.py
check:
	python3 examples/check_mst_reference.py
	python3 scripts/check_manuscript.py
	python3 scripts/check_rewrite.py
	python3 scripts/check_content_conservation.py
clean:
	latexmk -C hsf_hd.tex smoke.tex
