$pdf_mode = 5;
$out_dir = 'build';
$xelatex = 'xelatex -no-shell-escape -interaction=nonstopmode -halt-on-error -file-line-error %O %S';
$max_repeat = 5;
$xdvipdfmx = 'xdvipdfmx -E -V 7 %O -o %D %S';
