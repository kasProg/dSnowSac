# Engineering log

Chronological record of what was built/changed and *why*, updated as work
happens (not written retroactively from git log). Distinct from
[NOTES.md](NOTES.md), which is upstream-bug findings specifically;
this is the "why did we make this choice" record for our own code.

---

## 2026-08-10 — Repo skeleton + snow17 submodule

**What:** `git submodule add` for `https://github.com/NOAA-OWP/snow17`,
pinned at `ea142f94a4788ddf0be7b4d7da59c1f9f552df45`. Created
`fortran/`, `tesseracts/snow17/`, `src/`, `tests/`, `notes/` per the
repo layout CLAUDE.md specifies.

**Why a submodule instead of copying files in:** CLAUDE.md's own
reasoning — "which version, with which bugs" needs to be an answerable
question, both for reproducibility and because we found (and will keep
finding) upstream quirks that are commit-specific. Vendoring by copy would
lose that traceability and make "did we modify this" ambiguous, which
matters for the Apache-2.0 NOTICE obligation.

**Why pin to a specific commit rather than track a branch:** the whole
point of a submodule here is determinism — if upstream fixes the ADC bug
or the TPREV wiring after we've built around their current behavior, we
want that to be a deliberate `git submodule update`, not something that
silently changes our test baseline.

---

## 2026-08-10 — `fortran/snow17_shim.f90`

**What:** new Fortran source, ~110 lines. Loops the vendored, *unmodified*
`EXSNOW19` (from `external/snow17/src/snow19/exsnow19.f`) over a time
series for one HRU, exposed as a single `bind(C)` entry point
(`snow17_run`).

**Why write a shim instead of calling `EXSNOW19` directly from Python/C:**
`EXSNOW19` is single-timestep, single-HRU, with ~30 positional arguments
and an explicit state array the caller must thread by hand — there's no
way to call it usefully without *some* looping wrapper. The upstream
driver (`runSnow17.f90`) already is that wrapper, but it's built around
namelists, multi-HRU config, and file I/O we don't want; reusing it would
mean linking `src/share`, which CLAUDE.md's build-scope decision
explicitly excludes (keeps the dependency footprint to exactly the 13
physics files we can audit and attribute cleanly).

**Why `bind(C)` rather than an `iso_c_binding`-free F90 interface:** the
Tesseract/PyTorch side of this project is Python. `bind(C)` gives a flat C
ABI (`ctypes.CDLL` can load it directly) instead of requiring a
Fortran-aware FFI. This is also what makes it possible to build with plain
`gfortran -shared -fPIC`, no Fortran-specific packaging.

**Why the state (`CS(19)` + `TPREV`) is threaded explicitly by the
caller, not hidden inside the shim:** two reasons. (1) `EXSNOW19` itself
already requires this — it unpacks `CS` into a COMMON block at entry and
repacks at exit, so state has to come from *somewhere* outside the
function on every call; hiding it inside the shim would just relocate the
same problem one layer down and make chained/resumable runs impossible.
(2) It's what the [determinism and state-continuity tests](test_snow17_shim.py)
actually verify — if state were implicit (e.g. a module-level variable
retained between calls), those tests would be checking the shim's
plumbing instead of the physics, and would stop being a meaningful guard
against COMMON-block leakage.

**Why `tprev_io = tmp(t)` happens *after* the call, not before:** confirmed
by reading `PACK19.f` (`DTA = TA - TPREV` is the only read site) that
`TPREV` means "the previous timestep's air temperature," and neither
`PACK19.f` nor `exsnow19.f` ever assigns it — see the state-contract note
in CLAUDE.md that flagged this as an open question. The call at step `t`
needs `TPREV = T(t-1)`, so the update to `T(t)` has to happen after, not
before, or the model would see the *current* step's temperature as its own
"previous" value. Verified against real data later — see the 2026-08-10
"reference cross-check" entry below for the divergence this choice causes
relative to upstream's actual (buggier) behavior, and why the choice was
kept anyway.

**Why `PA` is computed inside the shim from `elev`, not passed in as a
forcing:** traced every use of the internal `ELEV` variable in
`exsnow19.f` and found the line that would compute `PA` from it is
commented out — `PA` has to come from outside, exactly as CLAUDE.md's
design note said, and this was confirmed against
`external/snow17/src/share/ioModule.f90`'s `sfc_pressure` subroutine
(same formula, same constants, down to the `Anderson 2006` sourcing) so
the shim's version isn't a guess, it's a re-implementation of the same
five-constant formula the real driver uses.

**Why `ADC` is padded from 11 to 12 elements before the call:**
`exsnow19.f` declares `ADC(11)`, `PACK19.f` declares the same dummy
argument as `ADC(12)` — a real declaration-size mismatch. Traced every
site in `snow19/` that indexes `ADC` directly (`adjc19.f`, `aesc19.f`,
`aeco19.f`, `updt19.f`) and found the only path that would touch index 12
(`UPDT19`, reached only when `IUPWE`/`IUPSC` data-assimilation flags are
set) is unreachable here — `EXSNOW19` hardcodes both flags to `0` on every
call. So this padding fixes a latent mismatch that isn't live corruption
in our usage, but it costs nothing and removes the mismatch outright
rather than relying on "we checked, it's fine" holding forever as the
model is used in new ways later (e.g. if data assimilation is ever turned
on). Full trace in [NOTES.md](NOTES.md).

**Why `-fdefault-real-8` must never be added to the build:** `EXSNOW19`
and everything it calls use gfortran's default `REAL` (4 bytes). Our
`bind(C)` dummy arguments use `c_float` (4 bytes) to match. Compiling with
`-fdefault-real-8` would silently reinterpret every `REAL` in the F77
files as 8 bytes, breaking argument association with our `c_float` dummy
args without any compile error — the exact "plausible but wrong" failure
mode this whole project is trying to avoid at the state/units layer.
Called out explicitly in a source comment for this reason, not just in
CLAUDE.md.

---

## 2026-08-10 — `fortran/build.sh`

**What:** builds `libsnow17shim.so` from the 13 vendored `snow19/*.f`
files plus the shim, in one `gfortran -shared -fPIC` command.

**Why a script instead of inlining the command in the README/Makefile:**
wanted the bounds-check toggle (`SNOW17_BOUNDS_CHECK=1`) available without
having two near-duplicate build commands to keep in sync (one for
day-to-day fast builds, one with `-fcheck=bounds` for catching Fortran
array issues during development, per CLAUDE.md's suggested workflow). A
single script with an env-var switch keeps that DRY, and the Makefile
(`build` vs `build-checked`) just sets the flag rather than duplicating
compiler invocations.

---

## 2026-08-10 — `src/snow17.py`

**What:** `ctypes` wrapper (`run_snow17`) around `snow17_run`, plus
`Snow17Params`/`Snow17Output` dataclasses.

**Why dataclasses instead of passing 13 positional scalars around:** the
Fortran side has an inherent flat-scalar signature (unavoidable — that's
`EXSNOW19`'s own argument list, and the shim mirrors it 1:1 so the
Fortran-to-Python argument mapping stays easy to audit by eye). On the
Python side, callers will be constructing parameter sets from an LSTM's
output vector and from a CSV-loaded basin-attributes table, both of which
are natural to bind to named fields — the dataclass exists to stop that
mapping from being 13 unlabeled positional floats at every call site.

**Why `np.ctypeslib.ndpointer` for the arrays instead of raw
`ctypes.POINTER(c_float)`:** `ndpointer` enforces `dtype`,
contiguity, and dimensionality *before* the call, so a caller who passes a
non-contiguous slice or the wrong dtype gets a clear Python `TypeError`
immediately, not silent memory corruption or a segfault three function
calls later inside Fortran. Given the project's stated failure mode to
guard against (silent wrongness, per the `CudnnLstmModel` cautionary
note in CLAUDE.md), the cost of the extra strictness is worth it.

**Why `cs0` defaults to `None` -> zeros rather than requiring the caller
to pass it:** matches `EXSNOW19`'s own documented cold-start convention
(`CS = 0.0`, `TPREV = 0.0`, no external initialization needed — see
CLAUDE.md's state-contract section). Keeping that as the Python default
means a caller doing a fresh single-basin run doesn't have to know the
internal state layout at all; only the multi-segment / warm-start /
Tesseract-VJP-perturbation callers need to pass `cs0` explicitly.

---

## 2026-08-10 — `tests/test_snow17_shim.py`

**What:** 7 tests — determinism, state continuity, mass balance (the
three CLAUDE.md calls out as required before the Tesseract wrapper), a
reference cross-check against `test_cases/ex1`'s per-HRU output, and an
explicit "prove the snowh divergence" test.

**Why synthetic data for determinism/continuity but real ex1 data for
mass-balance/reference checks:** determinism and state-continuity are
checks on the *shim's own plumbing* (does calling the function twice give
the same answer; does splitting one call into two give the same answer) —
they don't need physically realistic forcing, just forcing that exercises
both accumulation and melt branches so state actually moves through its
different code paths. Mass balance and the reference cross-check are
checks on *physical correctness*, which synthetic data can't validate —
only real forcing with a known-good reference output can.

**Why the reference-cross-check test doesn't just loosen tolerances until
`snowh` passes:** loosening a tolerance until a test passes without
knowing why the mismatch exists is exactly the "silent wrongness" failure
mode this project is trying to avoid — a green test that's green because
the assertion doesn't actually constrain anything isn't a test, just
decoration. Instead, `test_snowh_divergence_is_explained_by_upstream_tprev_bug`
replays the exact same run with `TPREV` pinned at `0.0` (matching what
tracing the upstream driver showed it actually does for a cold start) and
asserts *that* matches the reference — which turns the mismatch from "our
shim is being sloppy" into "we understand precisely why upstream's number
differs from the physically correct one, we've proven it, and we're
deliberately not replicating their gap." See NOTES.md for the full trace.

**Why the mass-balance test computes `corrected_pcp` with a per-timestep
rain/snow split instead of a blanket `pcp * scf`:** the first version of
this test used `sum(pcp) * scf` as expected input and was off by ~25%.
Rather than loosen the tolerance, traced `PACK19.f` and found `SCF` is
only applied to the snowfall fraction (`SFALL = PXI*FRACS*SCF`,
`PACK19.f:134`), not rain — matching the real-world motivation for the
correction (gauges under-catch wind-blown snow, not rain). The test now
encodes that split explicitly rather than treating `scf` as a blanket
precip multiplier, and closes to float32 precision once that's fixed.

---

## 2026-08-10 — `NOTICE`, `README.md`, `Makefile`, `.gitignore`,
`requirements.txt`

**What:** repo scaffolding for a one-command reproduction path.

**Why `.gitignore` excludes `external/snow17/test_cases/ex1/`:** the
extracted directory is a side effect of `tar xzf`-ing a file that already
lives (compressed) inside the pinned submodule — committing the extracted
copy to *our* repo would duplicate data that's already version-controlled
upstream (in the submodule) and would go stale if the pinned commit ever
changes. `tests/test_snow17_shim.py::_ensure_ex1_extracted` extracts it on first
test run instead, so a fresh clone + `git submodule update --init` still
reproduces without a manual step — which is the actual requirement
(`README.md` promises one-command reproduction, and CLAUDE.md's
deliverables list requires it), not "commit everything so nothing can
fail."

**Why `NOTICE` explicitly states no `snow19/` source files were
modified:** required by NOAA-OWP's own README (adaptations must credit
the source repo) and is the operative Apache-2.0 obligation for a
derivative work; also directly checkable — anyone can diff
`external/snow17` against upstream at the pinned commit and confirm the
claim rather than take it on faith.

---

## 2026-08-10 — Project environment: `uv`, and the Docker access gap

**What:** `.venv/` created via `uv venv --python 3.11`, dependencies
installed via `uv pip install -r requirements.txt`. `Makefile`'s `env`
target wraps this; `test` depends on it.

**Why `uv` instead of the base conda env or a plain `venv`:** the base
Python (`/usr/local/anaconda3/bin/python3`) is a shared system
environment on this machine — installing `tesseract-core`/`tesseract-torch`
into it would affect anything else using that interpreter and makes "what
exactly does this repo need" unanswerable from the repo alone. A
project-local `.venv` fixes that; `uv` over stdlib `venv`+`pip` because
it's already available on this machine and meaningfully faster for
repeated installs during iteration (relevant since `make test` re-runs
`uv pip install` every time to stay a true one-command reproduction path).

**What we found: this account can't use Docker.** `docker ps` ->
permission denied; `getent group docker` confirms the account isn't a
member. Relevant because Tesseract's standard workflow (`tesseract
build`) packages a Tesseract as an OCI container, and that's what the
hackathon's "package it as a real Tesseract" expectation presumably means
for the final submission.

**Why this didn't block Day 3-4 work:** `tesseract_core.Tesseract` has a
documented `from_tesseract_api()` constructor -- "This does not use a
containerized Tesseract, but rather imports the Tesseract API directly.
This is useful for debugging, but requires a matching runtime environment
+ all dependencies to be installed locally." That's exactly the
constraint our own `.venv` already satisfies. All of
`tests/test_gradients.py` runs through this path -- `apply()`,
`vector_jacobian_product()`, `abstract_eval()`, and the
`tesseract_torch.apply_tesseract()` autograd integration are all verified
working without touching Docker.

**What's still unresolved:** `tesseract build` itself, needed to
containerize for actual submission. Options going forward: request
`docker` group membership from HPC admin, use a different machine with
Docker access for that one step, or check whether rootless
Docker/Podman is viable here. Flagged in `tesseract_config.yaml`'s
`build_config` comments and in `README.md` rather than silently deferred
-- the risk is discovering this gap at submission time instead of now.

---

## 2026-08-10 — `tesseracts/snow17/tesseract_api.py`

**What:** the Snow-17 Tesseract wrapper -- `InputSchema`/`OutputSchema`,
`apply()`, `vector_jacobian_product()` (forward-difference FD), and
`abstract_eval()`.

**Why scaffolded via `tesseract init --recipe base` rather than written
from scratch:** wanted the actual expected shapes of `InputSchema`,
`Differentiable[...]` annotations, and the `vjp_inputs`/`vjp_outputs`/
`cotangent_vector` signature confirmed from the installed `tesseract-core`
version rather than guessed from CLAUDE.md's prose description -- prose
can drift from what a specific pinned package version actually expects
across a piece of software still stating a beta-ish version number.
Generated both `base` and `pytorch` recipe templates to compare: `base`
uses a plain user-supplied `vector_jacobian_product`, `pytorch` wraps
`torch.func.vjp` around a user-supplied `evaluate()` and expects the
whole computation to be torch-native/traceable. Ours isn't -- gradients
come from finite-differencing calls into compiled Fortran through
`ctypes`, not from anything `torch.func` can trace -- so `base` is the
correct recipe, not `pytorch`, despite this project's autograd endpoint
being PyTorch.

**Why `ADC` (11-point areal depletion curve) is NOT in
`DIFFERENTIABLE_PARAMS`:** asked the user directly rather than resolve a
real inconsistency in CLAUDE.md silently -- the "Learnable parameters"
section lists 11 named scalars *plus* `ADC(11)` and calls it "13 scalars",
while the "Tesseract specifics" section scopes the FD VJP to "~13 scalar
parameters... ~14 forward evaluations," which is only consistent with
forward-difference over roughly 13 differentiable values, not 11+11=22.
User chose: ADC fixed for v1. `vector_jacobian_product` raises `ValueError`
if `adc` is requested as a `vjp_input`, rather than silently returning
`0.0` -- an explicit error is far cheaper to debug than a gradient that's
quietly always zero and looks like a converged/uninformative parameter
during training.

**Why the FD step is a relative step with a floor
(`max(abs(value)*1e-3, 1e-4)`), not a fixed epsilon:** the 11 differentiable
parameters span very different scales (e.g. `SI` ~500, `MBASE` often
exactly `0.0`, `PXTEMP` ~O(1)). A single fixed epsilon would be
simultaneously too large relative to small parameters (poor local-slope
estimate) and swamped by float32 rounding for near-zero ones without the
floor (`MBASE=0.0` would get `eps=0`, a zero step, dividing by which is
undefined). Values are not yet tuned against a real training loop --
flagged as worth revisiting once loss curves are in hand and it's
possible to tell whether FD noise is limiting learning, per CLAUDE.md's
"verify the VJP against a manual perturbation before trusting it," which
this file's tests do, but "trusted for correctness" and "tuned for
optimization" are different bars.

**Why `vector_jacobian_product` explicitly casts to `float64` before
subtracting `perturbed - base`, rather than working in the native
`float32` of the rollout outputs:** found empirically, not designed in
up front -- see the `tests/test_gradients.py` entry below. `perturbed`
and `base` are two nearly-identical float32 arrays (they differ only by
one small parameter perturbation), so summing each in float32 and then
subtracting the two sums hits catastrophic cancellation. Subtracting
element-wise first (while both operands are still close to each other,
so the subtraction itself is well-conditioned) and only then summing --
in float64, so the *accumulation* of many small differences doesn't
itself lose precision -- avoids it. This is generic numerical-methods
practice for finite differences of near-identical values, not something
specific to this codebase, but it wasn't in the first version of this
function and its absence was the actual root cause of an apparent test
failure (see below), not a bug in the test.

---

## 2026-08-10 — `tests/test_gradients.py`

**What:** `apply()`-matches-shim-directly, `abstract_eval()`-matches-`apply()`
shapes, `vector_jacobian_product()`-vs-manual-perturbation (parametrized
over all 11 differentiable params), linearity-in-cotangent, ADC-rejection,
and a documented near-zero-gradient check for `PXTEMP`.

**Why the synthetic fixture spans 200 days from day-of-year 0, not a
shorter window:** the first version used `n=60`. Checked (didn't assume)
whether that window actually exercises melt: `(tmp > 0).sum()` over that
window was `0/60` -- deep winter only, by construction of a `sin`-based
seasonal temperature curve starting at day 0. That makes any melt-only
parameter's (e.g. `MFMAX`) true gradient exactly zero in that window --
correctly computed as zero by both the VJP and by manual perturbation
during ad hoc verification, but a real `0 == 0` isn't a meaningful test of
whether the VJP's *wiring* (right parameter perturbed, right output read,
right cotangent applied) is correct, since a wiring bug could also
produce zero by accident. Extended to `n=200`, matching the window
`tests/test_snow17_shim.py`'s own synthetic series already uses and had already
confirmed reaches positive temperatures.

**Why `test_vjp_matches_manual_perturbation` reads `base_value` from a
validated `InputSchema` instance rather than the raw input dict:** first
attempt used the raw Python float (e.g. `1.05`) to compute the FD step,
while `vector_jacobian_product` internally reads the *pydantic-coerced*
`float32` value (`np.float32(1.05) == 1.0499999523...`) -- a different
starting point for the same nominal parameter. Investigated in case this
alone explained an observed mismatch; it turned out not to be the
dominant effect (the real cause was the cancellation issue below), but
using the coerced value is still the correct apples-to-apples comparison
and was kept.

**Why this test almost shipped with a wrong "manual perturbation" formula,
and how it was caught:** the first version computed
`(perturbed["raim"].sum() - base["raim"].sum()) / eps` using the raw
float32 rollout outputs. For 7 of 11 parameters (specifically the
melt/lag-related ones that actually got exercised once the fixture
covered real melt) this disagreed with `vector_jacobian_product`'s answer
by up to ~0.5% -- above a `rtol=1e-4` tolerance that was deliberately
tight because both sides were supposed to be doing the identical
computation. Rather than loosen the tolerance (which would have quietly
hidden either a real bug or a real numerical issue, indistinguishably),
reproduced `vector_jacobian_product`'s internal arithmetic line by line
in a scratch script and confirmed it matched the wrapper's own output
bit-for-bit -- meaning the wrapper was correct and the test's independent
check was not, specifically because it summed float32 arrays before
subtracting (catastrophic cancellation) rather than subtracting
element-wise in float64 first. Fixed the test to match the numerically
correct approach (see the `tesseract_api.py` entry above) rather than
either loosening the tolerance or leaving a subtly-wrong "independent"
check in place -- a manual check that's itself numerically unsound isn't
independent verification, it just moves the bug to a different file.

**Why `test_pxtemp_gradient_is_structurally_near_zero` exists as a
passing test rather than being left undocumented:** CLAUDE.md already
identifies that `PXTEMP`'s hard threshold means its gradient is ~zero
almost everywhere until the sigmoid relaxation lands. Rather than leave
that as a fact someone has to remember from a planning doc, encoded it as
a test that will fail loudly (forcing an update, not a silent pass) once
the relaxation actually changes this behavior -- the assertion compares
`PXTEMP`'s gradient magnitude against `MFMAX`'s (a parameter confirmed to
respond continuously in the same window) rather than against a fixed
threshold, so it stays meaningful regardless of the absolute scale of
gradients produced by future changes elsewhere in the pipeline.

**Why `test_backward_through_apply_tesseract` exists as its own test,
separate from the `vector_jacobian_product`-level ones:** first checked
this ad hoc (not committed) and it worked -- `d(loss)/d(scf)` via
`tesseract_torch.apply_tesseract()` + `.backward()` matched the manual-FD
value to ~0.02%. But CLAUDE.md's whole argument is that Tesseract is
*load-bearing*: gradients have to actually cross the Fortran boundary
through PyTorch's autograd graph, not just be computable by calling
`vector_jacobian_product` directly (which never proves the graph-splicing
machinery itself works -- a broken `apply_tesseract()` integration could
coexist with a perfectly correct `vector_jacobian_product`). Left as
scratch verification, that distinction isn't captured anywhere durable.
Promoted into a committed test that builds a real `torch` graph, calls
`.backward()`, and asserts both that the resulting gradient matches
`vector_jacobian_product` *and* that it's nonzero -- the second assertion
specifically to catch the `CudnnLstmModel`-style failure mode CLAUDE.md
names as the cautionary example (an integration that runs without error
but silently returns zero gradients, which "gradients match" alone
wouldn't catch if both sides happened to independently return zero).

---

## 2026-08-11 — CLAUDE.md rewrite: SAC-SMA replaces HBV, two Tesseracts required

**What:** user rewrote CLAUDE.md — HBV is now the Aug-20 fallback only;
SAC-SMA is the primary downstream model, chained Snow17 -> SAC-SMA via
RAIM, each its own Tesseract (judging criterion 1 requires composing two
or more). Renamed `tests/test_shim.py` -> `tests/test_snow17_shim.py` and
`tests/test_tesseract_api.py` -> `tests/test_gradients.py` to match the
new repo layout CLAUDE.md specifies, fixed all internal references
(README, docstrings, other notes files) accordingly, and updated
README/HBV mentions to reflect SAC-SMA as the primary plan rather than
rewriting history in already-dated log entries above this one.

**Why rename now instead of leaving it for later:** about to add the
SAC-SMA equivalents (`test_sacsma_shim.py`, and eventually a
`tesseracts/sacsma/` alongside a differently-scoped `test_gradients.py`).
Adding new files in the new naming convention while the Snow17 ones
still used the old names would mean drifting further from CLAUDE.md's
layout with every commit instead of converging on it.

---

## 2026-08-11 — `external/sac-sma` submodule + archaeology

**What:** vendored `https://github.com/NOAA-OWP/sac-sma`, pinned at
`975902e3d44785f3b3503f29adfb5755120f5bf`. Read `src/sac/ex_sac1.f90`
(the `EXSAC` entry point, single-timestep/single-HRU, mirrors
`EXSNOW19`'s role exactly), `src/sac/sac1.f90` (the physics), and
`src/sac/sac_data_mod.f90` (the only module either of those `USE`s).
Confirmed via sac-sma's own `CMakeLists.txt` (`MODEL_SOURCES`) that the
physics-core build needs exactly these 3 files — smaller and simpler
than Snow17's 13-file F77 core.

**Why DOUBLE PRECISION, not the default REAL snow17 uses:** read the
actual type declarations rather than assuming symmetry with the Snow17
shim — `ex_sac1.f90`/`sac1.f90` declare every real value
`DOUBLE PRECISION` explicitly (`sac_data_mod.f90`'s `dp =
SELECTED_REAL_KIND(15, 307)` confirms 8-byte). `fortran/sacsma_shim.f90`
and `src/sacsma.py` use `c_double`/`float64` throughout — mixing this up
with Snow17's `c_float`/`float32` convention would be exactly the kind
of silent argument-association corruption the "never add
-fdefault-real-8" note on the snow17 shim exists to prevent, just from
the opposite direction.

**Why state is a flat 6-element array (`UZTWC, UZFWC, LZTWC, LZFSC,
LZFPC, ADIMC`), not a packed/indexed structure like Snow17's `CS(19)`:**
`EXSAC` itself passes these as 6 individually-named `INTENT(INOUT)`
arguments — there's no packed array to unpack/repack, no `NEXLAG`-style
timestep-dependent indexing. `state_io(6)` in the shim is purely a
calling-convention choice (uniform with `cs_io` on the Python side, one
array in/out instead of 6 loose scalars) — the physics itself has no
`CS`-equivalent bookkeeping to get wrong.

**Why `TMP` is still threaded through the shim/wrapper despite having
zero effect on output:** confirmed by reading `SAC1`'s body — `TA` (its
internal name for `TMP`) is only read inside `FROST1`, called only `IF
(IFRZE .GT. 0)`, and `EXSAC` hardcodes `IFRZE = 0` on every call. Same
dead-parameter shape as Snow17's `ELEV1`. Kept in the interface anyway
(rather than dropped) so the signature doesn't need to change if frozen-
ground support is ever turned on later — cheap to keep, and mirrors
`EXSAC`'s own argument list exactly, which keeps the Fortran-to-Python
mapping easy to audit by eye (same reasoning as the snow17 shim's
1:1-with-`EXSNOW19` design).

**Why `DUAMEL` (unit hydrograph routing) is NOT part of the shim:**
checked whether the reference driver (`runSac.f90`'s `solve_sac`) calls
it before assuming routing is needed to get from `EXSAC`'s `TCI` output
to something loss-comparable — it doesn't. `solve_sac` calls `exsac(...)`
directly and does its own mass-balance bookkeeping straight off `TCI`.
`DUAMEL` also isn't in `CMakeLists.txt`'s `MODEL_SOURCES` at all. So
`TCI` (exposed as `q` in `sacsma_shim.f90`/`src/sacsma.py`) is "runoff"
for the NSE loss, matching CLAUDE.md's diagram, with no separate routing
Tesseract needed for v1. Worth a caveat in the eventual writeup: `TCI` is
unrouted total channel inflow, not timing-lagged to match a USGS gauge
hydrograph the way `DUAMEL` output would be — a real limitation, not
hidden, just out of scope for now (CLAUDE.md's own 16-parameter SAC-SMA
list doesn't include `DUAMEL`'s unit-hydrograph shape parameters either,
so this reading is consistent with the plan as written, not a unilateral
scope cut).

---

## 2026-08-11 — `bypass_ratio_check` implicit-SAVE bug: found, proven, patched

**What:** found a live (not dead-code-gated) state-leakage bug in
`sac1.f90` — full writeup in [NOTES.md](NOTES.md). Fixed via
`patches/sac1_bypass_ratio_check_save_fix.patch`, applied to a
build-time-only copy of the file by `fortran/sacsma_build.sh`.

**Why patch a vendored dependency at all, given CLAUDE.md's Apache-2.0
framing leans on "Snow17 and SAC-SMA are unmodified upstream
dependencies":** there was no alternative that didn't sacrifice either
correctness or the whole point of having a fast FD-based Tesseract.
`bypass_ratio_check` has no argument, `COMMON` block, or module interface
exposing it — nothing our shim could poke from outside to reset it, the
way `TPREV` or `CS` could be threaded correctly for Snow17. The only
other options were: (a) don't fix it and hope basin data / parameter
search never triggers the precondition -- rejected outright, "hope it
doesn't trigger" is precisely the silent-wrongness failure mode this
whole project's testing philosophy exists to rule out, and gradient-based
parameter search will explore extreme parameter values that a calibrated
parameter set never would; (b) reload the shared library (or spawn a
fresh process) before every single `EXSAC` call to force a true reset of
all static state -- technically addresses it but destroys the performance
FD-based VJPs depend on (already budgeted at up to ~45 forward runs per
gradient per CLAUDE.md's cost analysis; per-call process spawns would
dominate that cost by orders of magnitude). A one-line source patch,
fully disclosed, applied only to a build-time copy, was the only option
that fixes the actual defect without either of those costs.

**Why apply the patch to a staged copy (`fortran/_sacsma_patched/`,
gitignored) instead of committing an edited file into `external/sac-sma`
directly:** the submodule pin is supposed to answer "which version, with
which bugs" unambiguously (same reasoning as the original decision to
vendor via submodule at all, logged above under "Repo skeleton + snow17
submodule") — editing the submodule's working tree directly would make
`git submodule status` show a dirty/detached state that doesn't match
any real upstream commit, and `git submodule update` would silently wipe
the edit on the next sync. A patch file applied at build time keeps
`external/sac-sma` byte-identical to the pinned commit *and* keeps the
fix auditable as its own diffable, reviewable artifact — closer to how a
Linux distro package patches upstream source than to silently forking it.

**Why the "unmodified dependency" README/NOTICE language needs updating,
and what it should say instead:** it's simply no longer accurate for
SAC-SMA. Apache-2.0 explicitly permits modification given attribution
(that's the license working as intended, not a workaround) — the honest
framing is "SAC-SMA is vendored unmodified as a pinned submodule; one
disclosed, minimal, build-time-only patch is applied to fix a confirmed
upstream defect, documented in `notes/NOTES.md` and
`patches/sac1_bypass_ratio_check_save_fix.patch`." Updated NOTICE and
README to say exactly that rather than leave the stronger "unmodified"
claim standing.

**How the patch itself was generated, and why not hand-written directly
as a unified diff:** wrote the target fix as a small Python
find-and-replace against a copy of the real file, then generated the
patch via `diff -u` against the original -- guarantees byte-exact
context lines (a hand-written patch's context has to match whitespace
and line content exactly or `patch` rejects it; got this wrong on the
first hand-written attempt -- `patch --dry-run` failed on a malformed
hunk before the second, diff-generated version applied cleanly).

---

## 2026-08-11 — `tests/test_sacsma_shim.py`

**What:** the same three required tests as `test_snow17_shim.py`
(determinism, state continuity, mass balance -- using `runSac.f90`'s own
mass-balance formula, read directly from its `derived%mass_balance`
computation rather than re-derived from scratch), plus a dedicated test
proving the `bypass_ratio_check` patch fixes the exact scenario in
NOTES.md through the real compiled shim (not just the standalone repro
program used to first confirm the bug), plus a reference cross-check
against ex1's `output.sacbmi.HHWM8I{L,U}.txt`.

**Why the mass-balance test's tolerance is `1e-2 * total_precip` rather
than something tighter:** checked the reference's own `mass_bal.csv`
before picking a number rather than guessing -- it shows non-zero
residuals itself (up to a few tenths of a mm on some days, computed by
upstream's own driver), so demanding a tighter closure from our shim
than upstream's own reference achieves would be testing against a
standard the reference doesn't meet either.

**Why `test_matches_upstream_reference` passing tightly isn't treated as
"the patch doesn't matter" evidence:** checked directly (see NOTES.md) --
`UZTWC` never hits exactly `0.0` anywhere in the 46-year HHWM8 state
trace, so the branch that sets `bypass_ratio_check` never fires for this
basin at all. A passing cross-check here demonstrates the shim's
plumbing is correct, not that the bug was inconsequential -- those are
different claims, and NOTES.md states which one is actually supported by
the evidence.

---

## 2026-08-11 — `src/coupling.py`: cross-container gradient coupling, prototyped first

**What:** `CoupledTwoStageFunction`, a single `torch.autograd.Function`
implementing CLAUDE.md's "option 1.5" gradient-coupling design, plus
`FDConfig` (per-parameter relative-step, floored, central-difference
config). Validated against cheap toy stand-ins
(`tests/test_coupling_toy.py`) before ever touching the real Tesseracts,
per CLAUDE.md's explicit "prototype on day 3-4 before committing."

**Why one `Function` instead of the split I originally proposed (custom
autograd for theta_A, normal `apply_tesseract()` composition for
theta_B):** user correction, and right — that split leaves a real open
question about how two structurally different gradient computations
(one from a hand-rolled `backward()`, one from Tesseract's own VJP
machinery composed automatically) correctly merge into the same upstream
network's `.backward()` without double-counting or sign errors. Folding
both parameter blocks into one `Function`'s `backward()` makes that
question not exist: both blocks reach the shared LSTM/MLP through the
exact same, single autograd node, and both are computed with the same
manual FD-sweep code path (`_fd_vjp_block`) -- theta_B's block is just a
shorter version of theta_A's (no `stage_a` re-run, RAIM held at a cached
base), not a different mechanism entirely. "Symmetry over hybrid," per
the correction.

**Why the `Function` returns `runoff` (via a real VJP: perturb, rerun,
dot the resulting *output* series against the incoming cotangent) rather
than differencing the loss directly:** second correction, also right.
Differencing the loss directly (`(loss(theta+eps) - loss(theta))/eps`)
would have baked a specific loss function into the coupling layer --
swapping NSE for KGE, adding a regularizer, or differentiating something
downstream of runoff (a routing step, a multi-basin aggregation) would
all require editing `coupling.py`. Returning `runoff` as a normal
`torch.Tensor` node and computing `d(runoff)/d(theta)` (dotted with
whatever cotangent arrives from upstream, which is exactly what a VJP
is) keeps the loss entirely in ordinary autograd-land, outside this
file, and is the honest mathematical object rather than a hidden
shortcut -- worth stating plainly in the writeup, per the user's note.
Cost is identical either way: each perturbed run already produces the
full `runoff` series, so keeping it (instead of collapsing straight to
a scalar loss inside the sweep) costs nothing extra, just a dot product.

**Why `FDConfig` supports per-parameter `rel`/`floor` arrays instead of
one scalar step size:** real Snow17/SAC-SMA parameters span very
different scales in the same gradient computation (`UZTWM` ~O(100) mm,
`UZK` ~O(0.1)) -- a single fixed `eps` is simultaneously too coarse
for the small-scale parameters (poor local-slope estimate) and
needlessly fine for the large-scale ones. This mirrors why
`tesseracts/snow17/tesseract_api.py`'s own `_fd_step` already does
relative-step-with-floor for its 11 parameters; `FDConfig` generalizes
that so `coupling.py` doesn't reinvent it differently when the real
wrappers get plugged in.

**Why central differences, not forward, for both blocks:** user's
correction -- Snow17 is float32, and forward differences carry a
leading O(eps) truncation error that's especially noisy at float32
precision; central differences (`(f(x+eps)-f(x-eps))/2eps`) kill that
leading term at the cost of one extra evaluation per parameter (26->52
runs for the real 13-parameter theta_A block -- still cheap). Applied
central to the theta_B block too, for the "symmetry over hybrid"
reasoning above, even though SAC-SMA's float64 precision would tolerate
forward differences better on its own -- consistency of method across
both blocks was judged more valuable than the small extra saving,
and `FDConfig(central=False)` remains available (with the cached base
output reused, not recomputed) if that tradeoff needs revisiting once
real timing numbers exist.

**How the toy prototype validates it, and why three independent
checks:** `tests/test_coupling_toy.py` builds cheap numpy stand-ins for
both stages (a sigmoid-relaxed rain/snow threshold for "Snow17," a
leaky-reservoir recursion with real memory for "SAC-SMA," matching the
real problem's shape on purpose -- nonlinear, stateful, heterogeneous
parameter scales including one deliberately near zero to exercise the
FD floor) plus torch-native equivalents of the identical math, so
autograd gives an exact ground-truth gradient. `CoupledTwoStageFunction`'s
FD-based gradient is checked against that autograd ground truth *and*
against an independently hand-rolled brute-force central difference of
the whole pipeline's loss, computed at a different eps than what
`FDConfig` itself uses in production for that run. The "different eps"
part is the point, per the user's correction: comparing an FD
computation to itself at the same step size is tautological (a wiring
bug -- wrong sign, wrong parameter, wrong output slice -- doesn't
necessarily shrink or grow with eps, so self-consistency at one eps
doesn't rule it out); two independently-implemented computations at two
different step sizes converging to the same answer is real evidence.
Measured the actual achieved precision before picking a test tolerance
rather than guessing one (relative errors were 1e-14 to 6e-5 across both
eps values tested) and set `rtol=1e-3` -- tight enough to actually catch
a moderate-sized regression, not just loose headroom that would pass
almost anything.

**Why there's also a call-counting structural test
(`test_raim_never_requires_grad_on_theta_b_path`), separate from the
numerical gradient checks:** a numerical match alone wouldn't catch a
version of this that "accidentally" got the right answer while still
being expensive -- e.g., if `stage_a` (Snow17) were silently re-invoked
during the theta_B sweep for some unrelated reason, gradients could
still come out numerically correct while defeating the entire point of
option 1.5 (avoiding N re-runs of the expensive stage on the cheap
parameter block). Wrapping both stages in a call-counting shim and
asserting exact expected call counts (`1 + 2*n_a` for stage_a, matching
forward + the central-diff theta_A sweep, never touching stage_a again
during the theta_B block) makes the cost claim itself a tested
invariant, not just an assumption from reading the code.

**What's still open going into the real wrapper integration:** the toy
uses float64 throughout for numerical cleanliness in the ground-truth
comparison; the real integration has Snow17 predicting/consuming
float32 and SAC-SMA float64 (see the snow17.py vs sacsma.py dtype notes
above), so `coupling.py`'s dtype handling at that boundary needs
explicit attention once `tesseracts/sacsma/` exists and both wrappers
get plugged in as the real `stage_a`/`stage_b`.

---

## 2026-08-11 — `tesseracts/sacsma/tesseract_api.py`

**What:** SAC-SMA's Tesseract wrapper, mirroring
`tesseracts/snow17/tesseract_api.py`'s structure closely -- `apply()`,
forward-difference `vector_jacobian_product()`, `abstract_eval()`. All
16 named parameters are differentiable (unlike Snow17, there's no extra
curve parameter to exclude -- CLAUDE.md's named list and the real ex1
parameter file agree exactly on 16, no ambiguity to resolve this time).
Tests added to `tests/test_gradients.py` (not a new file) -- same file,
new section, since the README already describes it as covering
"VJP vs. manual perturbation + torch autograd integration, per Tesseract"
(plural), and duplicating the whole apply/abstract_eval/VJP/linearity/
backward test structure into a second file would just be the same tests
with `Snow17Params` swapped for `SacSmaParams`.

**Why this wrapper's `vector_jacobian_product` is a genuinely separate
mechanism from `src/coupling.py`'s internal FD sweep, not a redundant
one:** this endpoint exists so the SAC-SMA Tesseract is independently
testable and reusable on its own -- consistent with the "a standalone
Tesseract works with any downstream model" reusability argument for
having two Tesseracts at all (CLAUDE.md's "why two Tesseracts" section).
The actual coupled training path in this project never calls it --
`src/coupling.py` calls `apply()` directly. Worth stating explicitly in
the module docstring (and here) so nobody reading both files later
assumes they're duplicated logic that should be unified; they solve
different problems (standalone SAC-SMA differentiability vs. the
cross-container coupling that specifically avoids needing SAC-SMA's own
VJP w.r.t. RAIM).

---

## 2026-08-11 — Real dtype bug caught by wiring in the real Tesseracts

**What:** `src/coupling.py`'s `forward()`/`backward()` originally stored
one `ctx.out_dtype = theta_A.dtype` and used it for the output tensor
*and both* gradients. Fixed to track `theta_A`'s and `theta_B`'s
dtype/device independently, and to return the output tensor in a fixed
`float64` (matching SAC-SMA's true precision) rather than borrowing
theta_A's dtype.

**Why `tests/test_coupling_toy.py` never caught this:** every toy test
used `float64` for both `theta_A` and `theta_B` -- convenient for the
ground-truth autograd comparison, but it meant `theta_A.dtype ==
theta_B.dtype` was true in every single toy test, so using theta_A's
dtype for theta_B's gradient was silently correct by coincidence. The
real integration has Snow17 at float32 and SAC-SMA at float64 (see
notes/NOTES.md), which is precisely the case that distinguishes "uses
each leaf's own dtype" from "uses theta_A's dtype for everything." This
is exactly the kind of gap toy validation can leave -- caught only once
`src/pipeline.py` wired in the real, dtype-heterogeneous Tesseracts, not
by the toy suite itself. Added
`test_mixed_dtype_thetas_get_matching_gradient_dtypes` to
`tests/test_coupling_toy.py` (mixed float32/float64 thetas, asserts each
gradient comes back in its own leaf's dtype) so this stays a cheap,
fast regression check rather than something only the slow real pipeline
would catch going forward.

---

## 2026-08-11 — `src/pipeline.py` + `tests/test_pipeline_hhwm8.py`:
## the real chain, and CLAUDE.md's "loss goes down" checkpoint

**What:** `CoupledNWSStack` wires both real Tesseracts as
`CoupledTwoStageFunction`'s `stage_a`/`stage_b`, for a single HRU's fixed
forcing. `tests/test_pipeline_hhwm8.py` runs it on real HHWM8 data (same
basin/period both vendored `ex1` test cases ship) and shows a loss
decreasing under gradient-based optimization -- CLAUDE.md's Day 5-6
"Loss goes down = submittable" checkpoint.

**Why SAC-SMA's own forcing (`tmp`, `etp`, `dtm`, `state0`) is closed
over rather than threaded through `CoupledTwoStageFunction`'s
signature:** that `Function` only has one `forcings` slot, deliberately
-- it's for `stage_a`'s (Snow17's) non-differentiable inputs specifically,
since `stage_a` is called with fresh perturbed parameters repeatedly and
needs its forcing on every call. `stage_b` (SAC-SMA) only ever needs its
*own* static forcing plus whatever RAIM it's handed -- since that forcing
never changes within one `CoupledNWSStack` instance, closing over it
(`self._sacsma_forcing`, read inside the bound `_stage_sacsma` method)
avoids widening `CoupledTwoStageFunction`'s already-tested, already-
committed signature for a second `forcings` argument that would be
`None`/unused on the `theta_A`-block calls anyway.

**Why synthetic-target recovery instead of real observed streamflow:**
checked, not assumed -- grepped both vendored `test_cases/` trees for
`obs`/`flow`/`discharge`/`streamflow` file names and found nothing; these
are NWS model-development test cases, not CAMELS evaluation data with
ground-truth gauge records. Running the real chain once at HHWM8's actual
calibrated parameters to generate a synthetic "observed" series, then
optimizing back to it from a perturbed start, is scoped honestly in the
test's own docstring as proving gradient flow + optimizer usability, not
real calibration skill -- that needs actual CAMELS streamflow, which is
CLAUDE.md's Day 7-10 milestone (parameter network, 30-50 basins), not
this one.

**Why the first version of this test produced NaN gradients, and why the
fix is a reparametrization, not a smaller learning rate:** the first
attempt ran plain `Adam([theta_A, theta_B], lr=0.02)` directly on raw
physical-unit parameters. `SI` (~1500), `MBASE` (~0), and `PXTEMP`
(~0.7, must stay near a sensible range) are all in the same tensor;
Adam's per-step update magnitude is roughly `lr` in each parameter's own
raw units regardless of that parameter's sensible range, so one step
barely nudged `SI` while shoving `MBASE`/`PXTEMP` into physically invalid
territory the Fortran call couldn't handle, producing NaN on the very
next gradient. A smaller global `lr` would have papered over this by
making all parameters converge glacially slowly rather than fixing the
actual mismatch (heterogeneous natural scales sharing one step size).
Fixed by optimizing `z = theta / scale` (dimensionless, every parameter
starting at O(1) regardless of its physical units) and reconstructing
`theta = z * scale` each forward pass -- ordinary differentiable
multiplication, so autograd chains the gradient back through it for
free; no changes needed to `coupling.py` or `pipeline.py`, this is
purely how the test's optimization loop is set up. Standard practice for
optimizing physical parameters at heterogeneous scales directly, not a
one-off workaround.

**What the actual result looked like, for the record:** loss (`1 -
NSE`) started at `0.020` (NSE `0.980`) from a 15%-perturbed initial
guess and dropped to `~0.0003` (NSE `0.9997`) within the first couple of
gradient steps on a real HHWM8 water year -- strong evidence the
gradients carry real, usable optimization information end to end through
both Fortran models, not just that they're finite and non-zero.

---

## 2026-08-11 — `tesseract build`: solved via CI, not local Docker access

**What:** investigated fixing Docker access on this dev machine directly
before deciding against it. `docker ps` -> permission denied (confirmed
earlier); this session additionally checked: no passwordless `sudo`
(`sudo -n true` fails); `podman` not installed; `dockerd-rootless-setuptool.sh`
*is* present (rootless Docker is installed) but needs `/etc/subuid` and
`/etc/subgid` entries for this account, which don't exist and require
root to add; the `docker` group itself appears AD-managed
(`docker:...:tkb5476,cxs1024,kel33@AD.PSU.EDU,lgl5139` -- all
`@AD.PSU.EDU` accounts), so even a one-off `sudo usermod -aG docker` (if
sudo were available) might not persist through the next AD sync. Every
path that fixes this *on this machine* needs an admin ticket. Asked the
user to choose between that, a generate-only-then-build-elsewhere
workaround, or CI; user picked CI.

**Why CI over the other two options:** GitHub-hosted runners have Docker
preinstalled with no group-membership restriction -- fixes the blocker
immediately, no ticket, no waiting. It's also strictly better than the
generate-only workaround for this project's actual goal: an automated,
reproducible, from-scratch container build triggered by every push is
stronger evidence for judging criterion 6 ("reproducibility and
communication") than "here's a Dockerfile, go build it yourself
somewhere," which would've been the generate-only path's end state
regardless of where the manual build eventually happened.

**Why this needed real code changes first, not just a CI YAML file:**
tried to get `tesseract_config.yaml`'s `build_config` right on the first
attempt rather than guess-and-iterate through several slow CI round
trips, by reading `tesseract_core.sdk.engine.py`'s
`prepare_build_context` and the `Dockerfile.base` Jinja template
directly instead of assuming. Two real findings from that reading, not
from a failed build:

1. **`tesseract_api.py` lands flat at `/tesseract/tesseract_api.py`
   inside the container** -- not nested 3 directories deep the way
   `tesseracts/snow17/tesseract_api.py` sits locally. Both wrappers'
   `_REPO_ROOT = Path(__file__).resolve().parent.parent.parent` would
   have resolved to the filesystem root inside a real container --
   never caught locally, since `Tesseract.from_tesseract_api()` (all
   local dev/testing) imports the file from its real, nested-3-deep
   location and the relative-path logic is correct there. Fixed by
   reading a `TESSERACT_PROJECT_ROOT` env var first, falling back to the
   local-dev relative-path computation when unset -- `env:
   TESSERACT_PROJECT_ROOT: "/tesseract"` in both `tesseract_config.yaml`s
   supplies it in-container; nothing changes for local dev, where the
   env var is simply absent.
2. **`package_data`/`custom_build_steps` run as root, after
   `extra_packages` are installed, before the container drops to a
   non-root user** (confirmed from `Dockerfile.base`'s literal
   instruction ordering) -- meaning the exact same build scripts local
   dev already uses (`fortran/build.sh`, `fortran/sacsma_build.sh`) can
   run unmodified inside the container via `custom_build_steps`, with
   `gfortran`/`patch` available via `extra_packages`. Deliberately did
   NOT ship a host-built `.so` into the image -- same glibc/arch-mismatch
   reasoning already documented for why the shims get built from source
   in the first place, just now applying to "host machine" vs.
   "container base image" instead of "dev machine" vs. "whatever runs
   the tests."

Also confirmed `package_data` source paths can point outside the
Tesseract's own directory (`../../fortran`, `../../external/snow17/src/snow19`,
etc., relative to `tesseracts/snow17/`) -- `prepare_build_context` stages
anything outside `src_dir` into a sibling `__package_data__` directory and
rewrites the path automatically, so there was no need to restructure the
repo or duplicate files into each tesseract's own directory. Scoped each
`package_data` entry to exactly what each shim's build script needs
(`external/snow17/src/snow19`, not the whole submodule; `external/sac-sma/src/sac`
+ `patches/`, not the whole sac-sma submodule) rather than copying entire
submodules into the image.

**What's still unverified:** none of this has been exercised against a
real Docker daemon yet -- this dev machine still can't run one. The
actual proof is the first CI run after this commit; `tests/test_tesseract_build.py`
(skips locally, runs in CI) calls real `apply()` against the built images
specifically so a successful `docker build` isn't mistaken for a working
container -- a build can succeed while runtime path resolution or a
missing dependency still breaks the actual endpoint.

---

## 2026-08-11 — Multi-basin training: CAMELS data, ParamNet, train.py

**What:** CLAUDE.md's Day 7-10 milestone -- a parameter network trained
across many snow-dominated CAMELS basins with held-out evaluation. Real
result: 35 training basins, 10 held-out, 25 epochs, train NSE
`-1.62 -> +0.50`, held-out NSE `-0.46 -> +0.54`. Held-out tracked training
closely throughout, no overfitting observed at this scale.

**Where the data comes from and why the full archive, not per-basin
downloads:** CAMELS v2.0, via its DOI (`10.5065/D6MW2F4D`) which
redirects to a Zenodo mirror
(`https://zenodo.org/records/15529996`) -- the original UCAR/RAL page
doesn't expose direct file links, had to follow the DOI to find them.
`basin_timeseries_v1p2_metForcing_obsFlow.zip` (3.4GB) is the only way
to get forcing + observed streamflow -- there's no per-basin download
option, so the full archive is downloaded once and only ~45 basins'
worth is ever used (`data/download_camels.sh`, not run by `make test` --
deliberately separate from the fast local loop). The small attribute
text files (`camels_clim/topo/soil/vege/geol.txt`, all under 130KB) are
downloaded separately and give the 39 static features `src/paramnet.py`
takes as input.

**Why `frac_snow` (from `camels_clim.txt`) for basin selection, not a
new metric:** it's the CAMELS authors' own climatology summary --
fraction of precipitation falling as snow -- not something derived here.
`data/select_basins.py` takes the top 45 by `frac_snow` (all `>0.6`,
range 0.6-0.91), filtered to a minimum 20 sq km area first to avoid
tiny/flashy headwater catchments that would be numerically finicky for
reasons unrelated to what's actually being tested here, split 35
train / 10 held-out with a fixed seed. Geographically diverse in
practice (CO/WY Rockies, WA Cascades, CA Sierra, ID, NV, UT, MT) without
that being an explicit selection criterion -- it falls out of picking
real high-frac_snow US basins.

**Why PET had to be derived, not used directly from CAMELS:** checked
the actual Daymet forcing file columns directly (`dayl(s) prcp(mm/day)
srad(W/m2) swe(mm) tmax(C) tmin(C) vp(Pa)`) -- no PET column. CLAUDE.md's
own plan says "CAMELS provides daily PET estimates," which doesn't hold
for the actual daily timeseries (only a basin-level *mean* PET exists in
`camels_clim.txt`, presumably the original paper's own annual-aggregate
estimate -- not usable as daily forcing). Derived via Hamon (1963),
sourced from USACE's HEC-HMS technical reference (an authoritative,
unambiguous source for the exact constants -- several algebraically
equivalent-looking variants with different surface constants circulate
in the literature, easy to get subtly wrong by mixing conventions from
two sources): `PET = 0.1651 * (N/12) * P_t`, `P_t = 216.7 * e_s / (T +
273.3)`, `e_s = 6.108 * exp(17.27*T / (T + 237.3))`. `N` (day length,
hours) comes directly from Daymet's own `dayl(s)` column -- no separate
latitude-based day-length calculation needed, Daymet already computed it
from the basin's actual location. Sanity-checked before trusting it:
T=20degC, N=12hr gives PET~=2.85 mm/day, in the expected range for a
temperate summer day.

**Why the training window is a fixed WY1991-1993 (3 years), not each
basin's full ~35-year record:** checked actual data coverage across all
45 basins before picking a window, rather than assuming CAMELS' nominal
1980-2014 span holds uniformly. Found 6 of 45 basins fail a WY1981-1983
window (missing entirely or >30% NaN streamflow) and 5 still fail
WY1986-1988 -- some gauges' usable records start later than CAMELS'
nominal coverage. WY1991-1993 is the first window with zero basins
failing (<1095 days or >5% missing). Kept the window short (3 years, not
the full record) specifically for training-loop cost: at the coupled
pipeline's measured cost (~0.15-0.2s/basin/epoch for a 3-year series),
35 basins/epoch stays around 5s/epoch -- a full 25-epoch, 45-basin run
finished in about 2 minutes. A full 35-year window per basin would have
made even a modest epoch count impractically slow for this stage, for
no benefit to what's actually being tested here (whether the network
learns useful basin-to-parameter mappings at all).

**Why `pipeline.py`'s `CoupledNWSStack` needed refactoring before any of
this:** it originally took `snow17_forcing`/`sacsma_forcing` at
construction time (fine for the single-basin proof in
`tests/test_pipeline_hhwm8.py`, where there's only ever one basin).
Multi-basin training reuses ONE `CoupledNWSStack` across 35+ basins per
epoch -- constructing a new instance per basin would reload both
Tesseract clients (`Tesseract.from_tesseract_api()`, a basin-independent,
non-trivial-cost step) 35+ times per epoch for no reason. Refactored so
`.__init__()` only loads the Tesseract clients, and `.run()` takes
forcing per call. SAC-SMA's forcing specifically is captured via a fresh
closure built inside each `.run()` call (`_make_stage_sacsma`), not an
instance attribute -- keeps different basins' calls from being able to
cross-contaminate each other's forcing even under interleaving, without
needing to widen `CoupledTwoStageFunction`'s already-tested signature
for a second `forcings` slot (that Function's single `forcings` argument
is deliberately for `stage_a` only -- see the original design entry
above).

**Why `ParamNet`'s output is a bounded sigmoid per parameter, not raw
network output passed through:** direct empirical precedent, not
caution in the abstract -- `tests/test_pipeline_hhwm8.py` already showed
what unconstrained direct optimization does to parameters spanning
wildly different scales (`SI~1500` next to `MBASE~0`): one bad step,
NaN. A trained network's raw output has no reason to respect physical
ranges any better than that manual optimizer did. Bounding at the
network's own last layer makes it structural -- true for every training
step, automatically, rather than depending on a carefully-tuned learning
rate holding for the entire run. Parameter bounds themselves: Snow17
from the state-contract notes plus real calibrated HHWM8 values seen
throughout this project, widened to plausible operational ranges; SAC-SMA
from standard NWS/SCE-UA calibration bounds in the literature (Duan,
Sorooshian & Gupta 1992) -- not tuned for this project specifically, a
reasonable-effort default given hackathon scope.

**Why full-batch gradient descent over basins (one `.backward()` per
epoch, across all 35 training basins at once), not per-basin SGD:**
`ParamNet`'s forward pass is batched (all 35 basins' attributes through
the MLP in one call, cheap), but each basin's `CoupledTwoStageFunction`
call is inherently per-basin (Snow17/SAC-SMA are single-HRU) and
expensive (real Fortran calls via Tesseract). Averaging all 35 basins'
losses into one scalar before a single `.backward()` means each
optimizer step reflects the whole training set's gradient, not one
basin's noisy individual estimate of it -- standard full-batch practice,
and it lets PyTorch's autograd handle the "backward through 35 separate
custom-Function nodes plus the shared MLP" graph in one pass rather than
35 separate ones with their own bookkeeping.

**On the actual result:** train NSE went negative-to-positive
(`-1.62 -> +0.50`) and held-out basins tracked training NSE closely the
entire run (`-0.46 -> +0.54`, ending slightly ABOVE training NSE, though
with only 10 held-out basins that's within plausible sampling noise, not
read as "generalizes better than train"). NSE ~0.5 from an MLP that
never sees a hydrograph directly -- only static basin attributes -- after
25 epochs, producing parameters for two chained legacy Fortran models
trained via finite-difference cross-container gradients, is a genuine
result: it's the actual proof this project's whole architecture works at
the scale CLAUDE.md's plan calls for, not just on the one hand-tuned
HHWM8 basin from the Day 5-6 checkpoint. Not polished/tuned (25 epochs,
default Adam lr, no learning-rate schedule, no regularization sweep) --
that's legitimate further work (CLAUDE.md's Day 11-14 is figures/writeup,
not more model development, so this may be close to final scope), not a
claim that this is a finished, optimized result.

---

## 2026-08-11 — LSTM upgrade + Hargreaves PET: licensing boundary, and a better result

**What:** user asked to look at MHPI's `dPLHBVrelease` for LSTM design
ideas, switch PET from Hamon to Hargreaves, and build "a better LSTM."
Result: `src/paramnet.py`'s MLP became an LSTM-encoder-plus-MLP-head
(12-month climatology -> LSTM -> concat with static attrs -> bounded
parameter head), PET switched to Hargreaves-Samani, cached to disk. Real
result improved over the prior MLP+Hamon run: train NSE
`-1.62 -> +0.50` became `-0.05 -> +0.62`; held-out NSE `-0.46 -> +0.54`
became `+0.23 -> +0.58` -- notably, the LSTM+Hargreaves run also starts
near NSE 0 instead of deeply negative, suggesting the richer climate
context gives sensible parameter guesses even before training moves
anything.

**Why `dPLHBVrelease` itself was never fetched, only its license
checked:** confirmed directly (fetched the raw LICENSE file) that it's
PSU Non-Commercial -- "You may not use Software for commercial purposes
without prior written consent from PSU" -- the exact restriction
CLAUDE.md already flags for `generic_deltamodel`/`hydrodl2` (same MHPI
lineage; the repo even contains a `hydroDL-dev/` directory, confirming
it's the same codebase family, not a coincidentally-similar name).
"Learn from it, don't copy" is fine copyright-wise for general
architectural ideas (an LSTM predicting physical-model parameters from
basin attributes is well-documented published science, not a copyrightable
expression unique to one repo), but reading actual source under a
restrictive license and then writing "your own" version raises the real
risk of unconsciously mirroring structure/naming/logic closely enough to
complicate the "original work" claim this submission's Apache-2.0
license depends on. Decided not to fetch any of its files at all, out of
that caution -- not because the user's request was improper, but because
there was a strictly safer way to get the same design benefit.

**Why NeuralHydrology instead, and why it's not just a legal
workaround:** checked its license directly too (BSD-3-Clause, genuinely
permissive) before treating it as a safe reference. It's also arguably
the *better* reference regardless of licensing -- the most actively
maintained, widely-cited LSTM rainfall-runoff framework in this exact
research area (Kratzert et al.'s foundational work), not a fallback
chosen only because the first option was off-limits.

**Why the LSTM's output stays a static (not time-varying) parameter
vector, despite now having a genuine sequence model in the loop:** this
is a hard constraint from `src/coupling.py`'s whole design, not an
oversight. A time-varying-parameter output would mean SAC-SMA's Tesseract
receiving a different parameter value per timestep, which reintroduces
exactly the "one FD run per RAIM timestep" cost explosion `coupling.py`
was built specifically to avoid (see that module's docstring and the
original coupling-design log entry above). The LSTM's role here is
narrower and compatible with that constraint: encode a *richer input
representation* (actual monthly seasonal patterns, not just pre-computed
summary statistics like `p_mean`/`aridity`) into a fixed-size embedding,
concatenated with static attributes, feeding the same kind of bounded
static-parameter head the MLP version used. More expressive input,
identical output contract.

**Why monthly climatology (12 steps) rather than the raw daily training
window (1095 steps) as the LSTM's input:** two reasons. First, cost is
irrelevant here specifically because the LSTM itself is ordinary native
PyTorch autograd (cheap, fast) -- the actual expensive part of every
epoch is the Fortran/Tesseract calls, unaffected by the LSTM's own
sequence length. Second, with only 35 training basins, a 12-step
climatological summary is a more robust signal than a 1095-step raw
daily series for a model this data-limited -- less surface area to
overfit noise in, and it's computed from each basin's FULL available
record (up to 35 years, not just the 3-year training window), so it's
also seeing more climatological history than the differentiable rollout
itself ever touches.

**Why PET is now precomputed and cached to
`data/camels/pet/{gauge_id}_pet.csv` instead of recomputed inline every
load:** user request, and a real improvement independent of that --
`camels_loader.load_basin_timeseries()` previously recomputed Hargreaves
PET from scratch on every call (cheap individually, but redundant across
the multiple places a basin's data gets loaded -- once for the training
window, once for climatology). `data/build_pet.py` now precomputes it
explicitly for every selected basin up front; `camels_loader.get_pet()`
still falls back to computing+caching on first access if a basin's cache
is missing, so the pipeline doesn't silently break if a step is skipped,
but the intended flow is cache-first. Plain CSV (date, pet), not a binary
format -- specifically so the values are inspectable on their own, not
just an implementation detail buried inside a training run (verified:
`data/camels/pet/11264500_pet.csv` reads back byte-identical to what
gets computed fresh, confirming the cache is correct, not just fast).

**Hargreaves-Samani formula, verified against an authoritative source
before trusting it (same discipline as the Hamon verification earlier):**
`ETo = 0.0023 * (Tmean + 17.8) * sqrt(Tmax - Tmin) * 0.408 * Ra`, with
`Ra` (extraterrestrial radiation) via the full FAO-56 (Allen et al. 1998)
procedure -- solar declination, sunset hour angle, inverse
relative-distance Earth-Sun. Cross-checked against PyETo (MIT-licensed
FAO-56 reference implementation; formula/constants only, no code
reused) and sanity-checked numerically before trusting it: `et_rad()`
at 40N gives Ra ranging ~13.5 MJ/m^2/day in deep winter to ~41.9 at
summer solstice, matching the expected mid-latitude seasonal curve
shape. Hargreaves PET came out roughly 2x Hamon's for the same basin
(e.g. 1.88 vs 0.92 mm/day mean, one basin checked directly) --
expected, not a bug: different temperature-only PET methods are known to
differ by this much depending on climate/elevation, particularly at the
high-elevation sites this project's basin selection is full of (high
clear-sky radiation drives Hargreaves' Ra term up more than Hamon's
day-length-only approach accounts for).

---

## 2026-08-11 — Pure LSTM benchmark: a real, controlled comparison

**What:** `src/benchmark_lstm.py` -- a pure data-driven LSTM (daily
forcing sequence + static attributes -> streamflow directly, no
physical model), built specifically to benchmark the hybrid
Snow17+SAC-SMA+ParamNet model, not as part of the submission's core
pipeline. Controlled for a fair comparison: same 35 train / 10 held-out
basins, same WY1991-1993 window, same masked-NSE loss/eval protocol,
same 3 forcing variables (prcp, tmean, PET) as what actually drives the
physical models -- so any performance difference is attributable to the
modeling approach, not to more data or more information given to one
side.

**Result:**

| model | epochs | train NSE | held-out NSE | train/held-out gap |
|---|---|---|---|---|
| hybrid (Snow17+SAC-SMA+ParamNet) | 25 | +0.62 | **+0.58** | 0.04 |
| pure LSTM benchmark | 150 | +0.68 | **+0.34** | 0.34 |

The pure LSTM reaches slightly higher training NSE but its held-out
performance plateaus well below the hybrid model's, with a steadily
widening train/held-out gap over training (heldout NSE: `-0.07 -> +0.09
-> +0.21 -> +0.30 -> +0.34`, visibly saturating well before train NSE
does, not still climbing in step with it). This is the expected,
well-documented failure mode for a flexible black-box model trained on
few basins (35 here, vs. the hundreds NeuralHydrology-style models are
normally trained on) -- it has nothing but the training data itself to
constrain it, so it overfits basin-specific noise the training set
happens to contain. The physically-constrained hybrid model can't do
that in the same way: its outputs are bounded to physically sensible
Snow17/SAC-SMA parameter ranges by construction (`src/paramnet.py`'s
bounded-sigmoid head), so there's real hydrologic structure carrying
generalization to unseen basins that a pure black-box has to learn (and,
here, largely fails to learn) from scratch out of 35 examples.

**Why this is worth taking seriously as a real result, not just a
favorable-looking number:** it's not a strawman comparison -- both
models see identical basins, identical forcing variables, identical
time window, identical loss function, and the benchmark was given far
more epochs (150 vs. 25) and reached higher *training* NSE, i.e. it
isn't simply undertrained relative to the hybrid model. The gap is
specifically a held-out generalization gap, which is exactly the
property differentiable parameter learning's methodological lineage
(Feng et al. 2022 and related work, cited in `src/paramnet.py`'s
docstring) claims as its actual value proposition over pure black-box
sequence models in data-limited settings. This project now has its own
controlled number for that claim rather than only a citation to
someone else's.

**Caveat, stated plainly:** 35/10 basins and a 3-year window is a small
data regime for ANY model, including the LSTM benchmark -- this result
shouldn't be read as "LSTMs don't work for rainfall-runoff modeling"
(they clearly do, at the hundreds-of-basins/decades-of-data scale
NeuralHydrology-style work normally operates at), only as "with the
specific, deliberately-matched small data budget this project's
FD-gradient-cost constraints impose, physical constraints generalize
better than an unconstrained black box does." That's a fair, honestly
scoped claim, not an overreach.

**Reproducibility check, not just the one run above:** the numbers
above came from an unseeded exploratory run. Added `torch.manual_seed`
+ JSON history saving to both `src/train.py` and `src/benchmark_lstm.py`
(`save_path` argument) and re-ran both with a fixed seed --
`results/hybrid_lstm_hargreaves_history.json` and
`results/benchmark_lstm_history.json`, see `results/README.md`. The
seeded hybrid run landed at train/held-out `+0.663/+0.519` (a 0.14 gap,
vs. the exploratory run's `+0.62/+0.58`, a 0.04 gap) and the seeded
benchmark run at `+0.676/+0.335` (a 0.34 gap, matching the exploratory
run's gap almost exactly). The precise gap size moved a bit between
runs (expected -- different random init, same architecture/data), but
the actual claim -- hybrid's gap is a small fraction of the pure LSTM's
-- held up under a second, independent, saved, reproducible run rather
than being a one-off artifact of a lucky unseeded initialization.

## Parked for later (separate branch): dHBV-scale SOTA comparison

Discussed replacing `ParamNet`'s 12-month climatology input with a raw
daily sequence (rho=365, dHBV/dPL terminology for the LSTM lookback
window) fed straight into the LSTM. Conclusion of that discussion:
genuinely useful direction, but NOT a strict upgrade to adopt on faith --
we already have direct evidence (the pure-LSTM benchmark above) that a
similarly long raw daily sequence overfits badly at this basin count
(35), and switching away from climatology trades a decades-averaged,
robust signal for a single-realization one. Treat as an ablation to
measure (climatology vs. rho=365 vs. both), not a default replacement.

The actual motivation surfaced during that discussion: the user wants
this evaluated against state-of-the-art differentiable hydrology models
(dHBV / dPL-style) properly -- not just our own 35-basin/3-year
controlled comparison above, but the standard CAMELS-671-basin
benchmark protocol those papers use (all 671 basins, and the same
temporal train/test period dHBV trains/tests on, not our 45-basin
snow-dominated spatial-holdout subset). That's a materially bigger data
pull (full 671-basin CAMELS forcing/streamflow + attributes, all
already downloaded via `data/download_camels.sh`'s bulk archive -- no
new download needed, just no longer restricting to the 45 snow-dominated
IDs), a different split (temporal, not spatial -- see the
`configs/split/temporal.yaml` stub added in the Hydra refactor below,
built specifically so this experiment has somewhere to plug in later),
and likely a longer training window matching dHBV's published period.

**Explicitly deferred, not dropped.** Do on a separate branch, after the
hackathon-scope pipeline (45-basin spatial-holdout, current
`results/`) is finished and written up. Do not let this pull focus from
the finishable submission -- CLAUDE.md's own "flag scope creep hard"
instruction applies directly here.

## Config-driven training/inference refactor (Hydra)

Requested separately from the above: not the 671-basin experiment
itself, but the infrastructure to run *any* experiment (this project's
current 45-basin spatial-holdout setup, or the parked 671-basin/temporal
one later, or anything in between) from a config file instead of
hardcoded constants (`WINDOW_START`/`WINDOW_END` module globals in the
old `src/train.py`, a second copy of the same training-loop shape in
`src/benchmark_lstm.py`).

Design:
- `configs/data/*.yaml` -- which CAMELS derived files to load (currently
  only `camels_snow35.yaml`, the existing 45-basin selection; a future
  `camels_full671.yaml` for the parked experiment above is a natural
  same-shape addition later, not built now).
- `configs/split/{spatial,temporal}.yaml` -- **the actual point of this
  refactor**: split mode is now a first-class, swappable config, not an
  assumption baked into the code. `spatial` = current behavior (fixed
  window, basins partitioned train/heldout via `data/select_basins.py`'s
  own `split` column -- prediction in ungauged basins). `temporal` = new
  mode, same basin set for both groups, different date windows (train
  vs. test period -- prediction in ungauged *period*, the shape the
  parked dHBV comparison will need). `temporal.yaml` is a real, working
  code path as of this refactor, just not yet exercised by any saved
  `results/` run.
- `configs/model/{hybrid,benchmark_lstm}.yaml`, `configs/train/*.yaml` --
  architecture/optimization hyperparameters, previously scattered across
  each script's `main()` kwargs.
- `src/data_module.py` -- `BasinExample` now takes its window as an
  argument (was a module-level constant); `build_split(cfg)` dispatches
  on `cfg.split.mode` and returns train/test examples + basin ID lists
  for either split type from one code path.
- `src/train.py` -- one Hydra-driven CLI for both models (`model=hybrid`
  or `model=benchmark_lstm` switches the whole run); `run_training(cfg)`
  is a plain function underneath the `@hydra.main`-decorated `cli()`, so
  tests and notebook-style calls don't need Hydra's compose/multirun
  machinery, only a manually built config object.
- `src/infer.py` -- new: loads a saved checkpoint + composes a
  (possibly different) data/split/model config, runs forward-only, saves
  per-basin simulated streamflow + NSE. Didn't exist before this
  refactor -- training only ever produced a history JSON, never a
  reloadable model checkpoint (`torch.save(net.state_dict())` added as
  part of this same change, in `run_training()`).

Both existing saved results were regenerated through the new CLI with
the same seed (`results/runs/hybrid_spatial_seed0/`,
`results/runs/benchmark_lstm_spatial_seed0/`, replacing the old flat
`results/hybrid_lstm_hargreaves_history.json` /
`results/benchmark_lstm_history.json` files) to confirm the refactor
didn't silently change model behavior. Confirmed exactly, not
approximately: hybrid epoch 1 `train=-0.1576 test=+0.0737` and epoch 25
(final) `train=+0.6627 test=+0.5194` match the pre-refactor numbers to
4 decimal places; benchmark epoch 1 `train=-0.3372 test=-0.0632` and
epoch 150 (final) `train=+0.6756 test=+0.3352` likewise. This is the
strongest evidence available that the refactor (module restructuring,
config plumbing, window now passed as an argument instead of a
module-level constant) changed *only* the surface, not any actual
computation -- `torch.manual_seed(0)` reproducing the identical
trajectory epoch-for-epoch through a full rewrite of the surrounding
code is a much stronger check than either file's tests passing in
isolation would have been.

Checkpoints are small (SAC-SMA/Snow17's parameters are the only thing
being learned, not a large network): 128KB (hybrid) / 220KB
(benchmark) -- fine to commit directly, no LFS/external storage needed.

Also fixed while touching this: `Makefile`'s `env` target
unconditionally ran `uv venv .venv --python 3.11`, which errors if
`.venv` already exists -- meaning `make test` (the README's documented
entrypoint) worked on a fresh clone but failed on every subsequent
invocation. Guarded with `[ -d .venv ] ||` so it's idempotent. Unrelated
to the Hydra work but found by literally running the README's own
reproduce instructions while updating them, and directly relevant to
this task's "anyone can reproduce" goal.

## 10-year window: primary hybrid-vs-benchmark_lstm comparison

`configs/split/spatial.yaml`'s window extended from WY1991-1993 (3
years) to WY1991-1999 (9 years, `results/runs/model_10yrs_spatial/` and
`results/runs/lstm_10yrs_spatial/` -- run dirs call it "10yrs" for the
round number, the exact window is 1990-10-01 to 1999-09-30). Same 45
selected basins (35 train / 10 heldout), same coverage guarantee
(<=5% missing streamflow) reconfirmed to hold over the longer range
before committing to it. `configs/train/hybrid.yaml`'s `n_epochs` moved
25 -> 150 to match -- the 3-year pilot's 25 epochs was tuned for that
shorter window's convergence, not a fixed budget; the longer window
needed more epochs to reach a comparable fit. `configs/train/benchmark_lstm.yaml`
was already 150 epochs, so this run is the first time both models get
the *same* epoch budget -- the 3-year pilot gave benchmark_lstm 6x the
hybrid model's epochs (a deliberate choice at the time, to rule out the
hybrid model simply being undertrained relative to the benchmark; see
that entry's own reasoning, still valid, just superseded as the
project's headline comparison by this cleaner apples-to-apples setup).

Result: a substantially sharper version of the 3-year pilot's finding,
not a new finding. Train/test gap: hybrid 0.14 (unchanged from the
3-year pilot, interestingly -- the same relative generalization
behavior held at 3x the window length) vs. benchmark_lstm 0.63 (up from
0.34 over 3 years -- more training data let the pure LSTM fit the 35
training basins tighter without transferring any better to the 10
heldout ones). The hybrid model now also has final training NSE
+0.84, exceeding the LSTM's +0.78 -- previously (7b61b27) the two
were within 0.01 of each other, an example used to argue the comparison
"isn't a strawman" (benchmark not simply undertrained). That argument is
even stronger here, and no longer close: the hybrid model wins on
training fit, held-out fit, and generalization gap simultaneously, on
identical basins/window/loss/epoch-budget.

This is now `results/README.md`'s primary comparison; the original
3-year pilot (`runs/hybrid_spatial_seed0/`/`runs/benchmark_lstm_spatial_seed0/`)
is kept, not deleted, as an earlier, independently-seeded data point
showing the same qualitative effect at a smaller scale. See
`results/README.md` for the numbers side by side and
`results/compare_runs.py` for the comparison script (also gained a
mean-vs-median `agg` column + mismatch warning around this same time,
from the unrelated theta_B-VJP change -- these two runs are the first to
carry median-aggregated `history.json`; the 3-year pilot's are still
mean-aggregated, `compare_runs.py` flags the difference rather than
silently mixing them).

## External LSTM check: split mismatch found and fixed, honest result kept

Prompted by a question worth asking: NeuralHydrology's commonly-cited
LSTM benchmarks land around median NSE 0.65-0.75, far above our own
`benchmark_lstm`'s 0.15 held-out NSE (see the 10-year-window entry
above). Read `neuralhydrology/training/loss.py`'s `MaskedNSELoss` first,
suspecting the loss formula (basin-variance-weighted MSE with a fixed,
precomputed per-basin std) might explain it. It doesn't: worked through
the math and `masked_nse_loss` (`src/data_module.py`) is nearly
equivalent to NH's loss for our specific training regime (full-batch,
fixed window, no minibatch subsequence sampling -- both reduce to a
per-timestep MSE weighted by 1/variance of that basin's observed flow).
The one real difference is stabilizer strength (NH's `eps=0.1` on `std`
vs our `clamp(sum_sq_dev, min=1e-6)`) -- a minor hygiene gap, not
something that explains a 0.5 NSE difference.

The actual explanation, found by inspecting an existing NH benchmark run
(`hackathon_bench/daymet_3008_195232/` in the neuralhydrology repo, not
this one): its `train_basin_file`/`test_basin_file`
(`examples/snow_basins_{train,test}.txt`) assign basins almost
**inversely** to `data/camels/selected_basins.csv`'s `train`/`heldout`
split, despite drawing from the identical 45-basin pool and window. Set
comparison: 9 of that run's 10 "test" basins are in *our* training set;
9 of *our* 10 heldout basins were in *its* training set. Only
`13023000` was held out in both. Its median NSE of ~0.69 (itself a
median masking a mean of -0.067, one basin scored -6.40 -- the same
mean-vs-median issue `compare_runs.py` now flags for our own runs) says
nothing about generalization to *our* heldout basins, because the model
had already seen 9 of them.

Fixed by editing those two basin-list files in place to exactly match
`data/camels/selected_basins.csv`'s split (verified with a direct set
comparison before trusting anything downstream of it -- exactly the
"what test would prove it doesn't" discipline this project tries to
hold itself to) and retraining from scratch
(`hackathon_bench/daymet_3008_224057/`, same config otherwise:
`commit_hash: e4329c3`, 256 hidden units, 5 forcing variables, 30
epochs, ~20 min on one GPU). Result, copied into this repo as
`results/external/neuralhydrology_lstm_pub/` (config + test_metrics.csv
+ a README with the full writeup -- NeuralHydrology is BSD-3-Clause, so
citing its output is fine; no source code copied): **median held-out
NSE 0.795**, ahead of the hybrid model's 0.70.

Decision (discussed directly): don't delete `src/benchmark_lstm.py` /
`results/runs/lstm_10yrs_spatial/` / `results/runs/benchmark_lstm_spatial_seed0/`
-- keep them as a from-scratch ablation and a marker for future work
(scaling capacity/forcing/training recipe toward NeuralHydrology's, to
see whether the hybrid model still wins once the LSTM side is no longer
a strawman), but stop treating "beats our own weak LSTM" as evidence
that physical structure beats black-box ML in general. `README.md` and
`results/README.md` now report both comparisons side by side, plainly:
the internal ablation (hybrid beats our LSTM, real and unchanged) and
the external check (a competent LSTM currently beats the hybrid model,
also real). Neither claim is hidden behind the other. The project's
actual Tesseract-composition contribution doesn't depend on either
number -- see CLAUDE.md's judging criteria, none of which require
beating LSTM SOTA on NSE.

### Reversed: benchmark_lstm removed entirely, not kept

The decision immediately above (keep `src/benchmark_lstm.py` as a
demoted ablation/future-work marker) was revisited and reversed within
the same session: removed entirely instead --
`src/benchmark_lstm.py`, `configs/model/benchmark_lstm.yaml`,
`configs/train/benchmark_lstm.yaml`, `tests/test_benchmark_lstm.py`,
and both its saved run directories
(`results/runs/lstm_10yrs_spatial/`, `results/runs/benchmark_lstm_spatial_seed0/`).
Reasoning: keeping a model in the repo whose own results section had to
carry a paragraph explaining why its headline number shouldn't be
trusted was net-negative for clarity, not net-positive for "future
work" -- a reader skimming would still see "hybrid beats an LSTM" before
reaching the caveat. `results/external/neuralhydrology_lstm_pub/` (the
honest, apples-to-apples comparison against a properly-engineered LSTM)
stays; it doesn't have this problem since it isn't presented as a claim
this project gets credit for.

`src/train.py`/`src/infer.py`/`src/model_factory.py` lost the
`cfg.model.name == "benchmark_lstm"` branches (down to a single
`"hybrid"` path each); `configs/config.yaml`'s example CLI overrides and
`.gitignore`'s run-dir allowlist lost their benchmark_lstm entries
accordingly. `results/compare_runs.py` needed no code changes -- it
compares arbitrary run directories generically, only its docstring's
example paths referenced the now-deleted runs.

## Down to one canonical run: model_9yrs_spatial replaces model_10yrs_spatial and the 3-year pilot

`src/train.py` gained periodic resume checkpoints and a final-epoch
`test_predictions.json` (per-basin simulated streamflow + NSE on the
test set). Reran the primary training job with that code under a
corrected name -- `results/runs/model_9yrs_spatial/`, not
`model_10yrs_spatial/`: the window is 1990-10-01 to 1999-09-30, nine
water years, not ten; "10yrs" was a round-number mislabel flagged but
not fixed back when that run dir was first created. Same seed, same
config otherwise -- numbers match `model_10yrs_spatial/` exactly
(median train/test NSE 0.84/0.70 at epoch 150, 0.38/0.28 at epoch 1),
confirming this is a rename-and-refresh, not a different result.

Removed both `results/runs/model_10yrs_spatial/` (superseded exactly
by the above) and `results/runs/hybrid_spatial_seed0/` (the earlier
3-year pilot) from the repo -- `git rm`, not a history rewrite, so both
remain recoverable from earlier commits if ever needed. Decision to
also drop the pilot (not just the exact duplicate) was a direct choice,
not a default: one clean canonical result, not history in the repo
tree, going into the submission's final state. `results/README.md`,
the main `README.md`, `results/compare_runs.py`'s example usage, and
`results/external/neuralhydrology_lstm_pub/README.md` all updated to
point at `model_9yrs_spatial/` -- the numbers cited everywhere were
already this run's numbers, so no figure changed, only which directory
name and which files back them.

---

## 2026-09-22 — Coupling switched from custom reverse-mode Function to forward-mode Tesseract composition

**What:** Replaced `CoupledTwoStageFunction` (the "option 1.5"
reverse-mode `torch.autograd.Function` that hand-rolled the
cross-container gradient) with forward-mode automatic differentiation
over the two Tesseracts. Each `tesseract_api.py` now also implements
`jacobian_vector_product` (forward mode) alongside its existing
`vector_jacobian_product`; SAC-SMA's `pcp` forcing became a
`Differentiable` input so the RAIM tangent can flow into it.
`src/coupling.py` is now `run_physics` — it seeds a unit tangent on each
of the 27 parameters, runs the `apply_tesseract(snow17) ->
apply_tesseract(sacsma)` chain under `torch.autograd.forward_ad`, and
reads each runoff tangent as one column of `d(runoff)/d(theta)`, then
hands runoff back as an ordinary autograd tensor carrying that Jacobian
so a downstream `loss.backward()` reaches the network. `src/pipeline.py`
lost its hand-written `_stage_*`/`_make_vjp_sacsma` numpy plumbing;
the composition is now Tesseract's own.

**Why:** The old design achieved the right *cost* (avoid the dense
`d(runoff)/d(RAIM)` Jacobian) but did it with bespoke code that had to
explain, at length, why no single Tesseract's VJP could span the
downstream container. Forward mode is the textbook-correct framing for
this graph — few inputs, one wide intermediate flux, scalar loss — and
tesseract-torch already dispatches PyTorch forward-mode AD to the
`jacobian_vector_product` endpoints and chains them across the container
boundary automatically. So the cross-container gradient path is now the
framework's own machinery, not ours. Same evaluation count as before
(one pass per parameter); the win is a cleaner narrative and the deletion
of the entire hand-rolled coupling layer, not speed.

**The one seam that remains:** forward mode differentiates the physics,
but the parameters come from a network that trains by reverse mode, and
the two modes don't fuse in a single `.backward()`. `run_physics` bridges
them via a thin `_PhysicsRunoff` autograd Function: forward-mode builds
the parameter Jacobian, reverse-mode contracts it against the loss
gradient and flows into the network. Documented in `src/coupling.py`'s
module docstring.

**Validation:** `tests/test_coupling_toy.py` rewritten to check
`run_physics` against autograd ground truth (rtol 1e-6) and an
independent brute-force FD, plus an evaluation-count assertion proving
the dense RAIM Jacobian is never materialized (physics called exactly
len(A)+len(B) times). `tests/test_gradients.py` gained JVP/VJP
adjoint-identity checks (`<u, J@e> == (J^T@u)[p]`), a JVP-linearity
check, and — the load-bearing one — a directional-derivative check of
SAC-SMA's `pcp` JVP against an independent central FD, plus a
forward-mode-chaining proof that a `pcp` dual tangent survives
`apply_tesseract`. `tests/test_pipeline_hhwm8.py` now checks the coupled
forward-mode gradient against a brute-force FD through both real Fortran
models (5% tolerance — a float32-Snow17 FD-vs-FD comparison; a sign flip
or dropped tangent misses by >100%). Full suite: 78 passed, 5 skipped
(Docker/CAMELS-gated).

**One subtlety found and fixed:** the `pcp` directional derivative
initially used a large one-sided FD step and disagreed with a converged
reference at a few timesteps — SAC-SMA's storage-full/percolation
conditionals are kinks in `pcp`, and a large step secants across them.
Switched that path to a small central difference (stable to ~1e-8 across
step sizes); the scalar parameters keep the cheaper one-sided step. Same
class of hard-threshold issue already documented for Snow-17's PXTEMP.


---

## 2026-09-22 (later) — FD endpoints now delegate to tesseract-core experimental helpers

**What:** Replaced the hand-rolled finite-difference code inside both
`tesseract_api.py` files (`_scalar_column`, `_param_column`,
`_pcp_directional_derivative`, the accumulation loops) with
tesseract-core's experimental helpers,
`tesseract_core.runtime.experimental.finite_difference_jvp` and
`finite_difference_vjp`. Each endpoint is now a few lines: build a
per-input `eps` mapping and call the helper. ~120 lines of bespoke
numerics deleted across the two files.

**Why:** They're a drop-in for exactly what we wrote, they get maintained
upstream, and leaning on them strengthens the "leverage Tesseract's own
features" story. Notably `finite_difference_jvp` already implements the
directional-derivative trick (perturb along the tangent, one central pair,
O(1) rollouts regardless of series length) that the `pcp`/RAIM coupling
depends on — so the piece I was most careful about by hand is the helper's
default behavior.

**Two things we had to supply, not accept as default:**
1. *Relative step sizes.* The helpers take `eps` as an ABSOLUTE
   perturbation (a scalar applied unscaled to every input). Snow17/SAC-SMA
   parameters span orders of magnitude (SI ~1500 next to MBASE ~0; UZTWM
   ~100 next to LZPK ~0.01), so we pass a per-path `eps` mapping built from
   the old relative-with-floor `_fd_step`. The helper validates that the
   mapping names every differentiated path and no others — a typo is an
   error, not a silent fallback.
2. *pcp directional step.* `finite_difference_jvp` perturbs `pcp + eps *
   tangent`; we size `eps` to `1e-4 * ||pcp|| / ||tangent||` so the
   effective step matches the series magnitude regardless of the incoming
   tangent's norm (`_pcp_eps`). Same small-central-difference reasoning as
   before — SAC-SMA's storage-full kinks — just expressed as an eps rather
   than a hand-written central difference.

**Algorithm change:** everything is now central difference (the helper
default), including the VJP, which was previously one-sided forward. More
accurate; the only fallout was the two `test_*_vjp_matches_manual_perturbation`
tests, which compared against a manual FORWARD difference — updated to a
central difference through `apply()` (still an independent recomputation,
just matching the endpoint's algorithm).

**Caveat, stated plainly:** these helpers are marked experimental ("API
may change in future releases"). We depend on the per-path `eps` mapping
and the JVP directional-derivative behavior. If a future tesseract-core
bumps that API, these six endpoint functions are where it lands — small
and well-tested. Judged worth it for the code deletion and the cleaner
framing; the full gradient test suite (incl. the coupled brute-force-FD
check through both real Fortran models) is the safety net and stays green
(78 passed, 5 skipped).

## 2026-09-27 — Zero-tangent inputs dropped before finite differencing; tesseract-core pinned >=1.14

**Found on review of the forward-mode switch:** gradients were correct
(matched the old reverse-mode path to ~1e-7 on Snow17 parameters, ≤3e-3 on
SAC-SMA's, the gap being one-sided vs. central differencing) but one
forward+backward on the HHWM8 water year cost **513 Snow17 + 729 SAC-SMA
rollouts** (2.84 s), against 23 + 40 (0.10 s) before.

**Cause:** `_parameter_jacobian` seeds one parameter per pass, but
tesseract-torch passes every differentiable input to
`jacobian_vector_product`, zero tangents included — reseeding the other
parameters as plain tensors doesn't help, the tangents arrive as zeros
either way. `finite_difference_jvp` groups inputs by `eps` and spends a
central pair per group; with relative steps, nearly every parameter is its
own group. So each pass nudged ~11 (Snow17) or ~13 (SAC-SMA) groups where
only one had a non-zero tangent.

**Fix:** both `jacobian_vector_product` endpoints drop inputs whose tangent
is all zeros before calling the helper, and return exact zeros without a
rollout when nothing is left (Snow17 on SAC-SMA-parameter passes). Result:
**49 + 77 rollouts (0.66 s)**, gradients bitwise identical to the unfiltered
version. `test_coupled_gradient_rollout_budget` guards it — verified to fail
(513/729) with the filter removed.

**Honest remaining cost:** 126 rollouts vs. the old 63. Each of the 27
passes re-runs both models un-nudged (the old code ran that once and
reused it), SAC-SMA-parameter passes re-run Snow17 though its output can't
change, and SAC-SMA parameters now take a central pair instead of a
one-sided step. Recovering the 63 would mean caching around Tesseract's
composition — i.e. reintroducing hand-written orchestration. Not done:
~2x rollouts is the price of the gradient crossing containers on
Tesseract's own machinery. Arguably the filter belongs upstream (tesseract-
torch skipping zero tangents, or the helper skipping zero-tangent groups).

**Pin:** per-path `eps` mappings only exist from tesseract-core 1.14.0; on
1.11.0 every endpoint using them fails. `requirements.txt` and the CI
install step now say `>=1.14`.
