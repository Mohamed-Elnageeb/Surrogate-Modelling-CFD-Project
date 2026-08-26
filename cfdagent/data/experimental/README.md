# Experimental NACA0012 data

Downloaded from the NASA Turbulence Modeling Resource
(https://tmbwg.github.io/turbmodels/naca0012_val.html). Public-domain NASA data,
included here so the validation study is reproducible offline.

| File | Source | Re | Transition |
|---|---|---|---|
| `CLCD_Ladson_expdata.dat` | Ladson, NASA TM 4074 (1988) | 6e6 | **tripped** (80 grit) |
| `CL_Gregory_expdata.dat` | Gregory & O'Reilly (1970) | 3e6 | **tripped** |
| `0012.abbottdata.cl.dat`, `0012.abbottdata.cd.dat` | Abbott & von Doenhoff, *Theory of Wing Sections* | 6e6 | **untripped** (free) |
| `0012.mccroskeydata.cl.dat` | McCroskey (1988) | — | best-fit Cl |

The tripped/untripped distinction is the crux of the validation study: a tripped
model is turbulent from near the leading edge, while an untripped model runs
laminar over the forward chord. Tools that model natural transition (XFOIL,
NeuralFoil) and fully-turbulent RANS closures (SU2 with SA) are each valid for
one of these conditions and systematically wrong for the other.

Note: the Abbott data is digitized from a printed plot and the source file warns
that "digitizing only approximate, due to poor quality of plot in the book".
