#!/bin/bash
# HPC3: TeX Live 2025 (the frozen tlnet-final) with cm-super,
# and poppler (pdftoppm, pdftotext, pdffonts, pdfinfo) for the paper figure steps. Installs into
#   /pub/jinboz1/hnl/env/texlive          (TeX Live; bin/x86_64-linux is put first on the PATH by the figure steps)
#   /pub/jinboz1/hnl/env/mamba/envs/texfig (conda-forge poppler; appended to the PATH after the venv)
# Nothing under $HOME: TEXMFVAR/CONFIG/HOME live inside the TeX Live tree, micromamba runs through hnl_mm.
#   sbatch slurm/hpc3/install_tex.sbatch
#SBATCH -J hpc3_install_tex
#SBATCH -p free
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 03:00:00
#SBATCH -o /pub/jinboz1/hnl/logs_hpc3/%x_%j.out
export HNL_ROOT=${HNL_ROOT:-/pub/jinboz1/hnl} HNL_DATA=${HNL_DATA:-/pub/jinboz1/hnl_lrt}
set -euo pipefail
source $HNL_ROOT/hnl-displaced/env/activate.sh || exit 1
echo "host $(hostname) job ${SLURM_JOB_ID:-none} $(date -u +%FT%TZ)"

TL=$HNL_ROOT/env/texlive
REPO=${TL_REPO:-https://ftp.math.utah.edu/pub/tex/historic/systems/texlive/2025/tlnet-final}
W=$HNL_ROOT/tmp/install_tex_${SLURM_JOB_ID:-$$}
mkdir -p "$W"
export HOME=$W/home TEXMFHOME=$TL/texmf-home TEXMFVAR=$TL/texmf-var TEXMFCONFIG=$TL/texmf-config
mkdir -p "$HOME"

if [ ! -x "$TL/bin/x86_64-linux/kpsewhich" ]; then
    cd "$W"
    curl -fsSL --retry 5 -o install-tl-unx.tar.gz "$REPO/install-tl-unx.tar.gz"
    tar xzf install-tl-unx.tar.gz
    cat > tl.profile <<EOF
selected_scheme scheme-basic
TEXDIR $TL
TEXMFLOCAL $TL/texmf-local
TEXMFSYSVAR $TL/texmf-var
TEXMFSYSCONFIG $TL/texmf-config
TEXMFVAR $TL/texmf-var
TEXMFCONFIG $TL/texmf-config
TEXMFHOME $TL/texmf-home
binary_x86_64-linux 1
instopt_adjustpath 0
instopt_adjustrepo 0
instopt_letter 1
instopt_portable 1
instopt_write18_restricted 1
tlpdbopt_autobackup 0
tlpdbopt_install_docfiles 0
tlpdbopt_install_srcfiles 0
EOF
    ./install-tl-*/install-tl -repository "$REPO" -profile tl.profile -no-interaction
fi
export PATH=$TL/bin/x86_64-linux:$PATH
# matplotlib usetex: type1cm, type1ec (cm-super), geometry, underscore; amsmath/amssymb from the figure preamble
tlmgr option repository "$REPO"
tlmgr install cm-super type1cm ec geometry underscore dvipng amsfonts
mktexlsr > /dev/null
updmap-sys > /dev/null 2>&1 || true

if [ ! -x "$MAMBA_ROOT_PREFIX/envs/texfig/bin/pdftoppm" ]; then
    hnl_mm create -y -p "$MAMBA_ROOT_PREFIX/envs/texfig" -c conda-forge --override-channels poppler
fi

# ----------------------------------------------------------------------------------------------- check
P=$W/check
mkdir -p "$P"
echo "tex: $(command -v latex) $(latex --version | head -1)"
echo "type1ec.sty: $(kpsewhich type1ec.sty)"
echo "sfrm1000.pfb: $(kpsewhich sfrm1000.pfb)"
export PATH=$PATH:$MAMBA_ROOT_PREFIX/envs/texfig/bin
for t in pdftoppm pdftotext pdffonts pdfinfo; do echo "$t: $(command -v $t)"; done
cd "$P"
python -B - <<'EOF'
import sys
sys.path.insert(0, "/pub/jinboz1/hnl/hnl-displaced/paper1_v4/figures")
import matplotlib as mpl
mpl.use("Agg")
import canto_style as cs
assert cs.setup(), "canto_style.setup() did not find latex + cm-super"
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(3, 2))
ax.plot([0, 1], [0, 1], label=r"$\mathrm{FR}^\star = 0.5\%$ -- test")
ax.set_xlabel(r"$d_0$ [mm]"); ax.legend()
fig.savefig("t.pdf")
print("usetex", mpl.rcParams["text.usetex"])
EOF
pdffonts t.pdf
pdftoppm -r 50 -png -singlefile t.pdf t && ls -l t.png
pdffonts t.pdf | grep -q -E "SFRM|CMR|CMMI" || { echo "no Computer Modern in the test PDF"; exit 1; }
rm -rf "$W"
echo "install_tex: OK $(date -u +%FT%TZ)"
