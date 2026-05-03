$pdf_mode = 5;  # xelatex
$pdflatex = 'xelatex -interaction=nonstopmode -synctex=1 %O %S';
$clean_ext = 'synctex.gz synctex.gz(busy) fdb_latexmk fls xdv bbl blg';

# Build artifacts (.aux/.log/.bbl/.pdf) live alongside the .tex source in
# docs/ so the repo root stays clean. Running `latexmk` from the repo root
# cd's into docs/ via @default_files and the per-directory output.
$out_dir   = 'docs';
$aux_dir   = 'docs';
@default_files = ('docs/technical_report.tex');
