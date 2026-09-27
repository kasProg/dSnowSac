# Project: Differentiable NWS Operational Stack (Snow17 + SAC-SMA) via Tesseract

## What this is

A submission for the **Pasteur Labs Tesseract Hackathon 2026** (Aug 3–31, 2026;
submissions close Aug 31, 11:59 PM Anywhere on Earth). **Track 03: Hybrid ML +
mechanistic models.**

**Goal:** Make the NWS operational forecasting stack — Snow17 feeding SAC-SMA —
differentiable end to end, and train a neural network to predict both models'
parameters from basin attributes by backpropagating through both Fortran models.

```
basin attributes (CAMELS)
      |
      v
   [ LSTM / MLP ]  predicts parameters for BOTH models   (native PyTorch)
      |
  +---+--------------------------+
  v                              v
[Tesseract A: Snow17]            [Tesseract B: SAC-SMA]
 NOAA-OWP Fortran-77              NOAA-OWP Fortran-77
 finite differences    --RAIM-->  finite differences
                                       |
                                       v
                                  runoff --> NSE loss vs. CAMELS observed
                                       |
   <------ gradients flow back through both containers ------+
```

- **RAIM** (rain-plus-melt, mm/day) is the coupling flux. This is the real
  operational coupling: NWS runs Snow17 -> SAC-SMA in production. The container
  boundary sits where a genuine seam already exists — separate codebases,
  separate repos, separate operational BMI modules.
- **HBV is OUT of the project.** SAC-SMA replaces it (it is the operational
  soil-moisture model Snow17 actually pairs with). HBV appears in this document
  only as the fallback plan (see Timeline).
- **Do not manufacture boundaries.** An earlier design (HBV written in JAX just
  to create a framework boundary) was rejected: writing a module in a different
  language *in order to* need Tesseract is a costume, not a constraint. The
  Fortran models are real constraints — decades of validated operational code
  that is not ours to rewrite.

### The why-Tesseract argument (writeup core, criterion 3)

PyTorch's `.backward()` walks a recorded graph. Snow17 and SAC-SMA are compiled
Fortran — nothing is recorded, backprop stops, the network can never learn their
parameters. Tesseract wraps each in a container exposing `apply` and
`vector_jacobian_product` (by finite differences), and `tesseract-torch` splices
them into the autograd graph as normal differentiable layers. Secondary framing:
BMI (NextGen's interoperability standard) lets these models exchange fluxes but
not gradients; Tesseract gives interoperability *and* differentiability.

### Why two Tesseracts and not one merged container (a judge will ask)

1. **The seam is real.** Merging Snow17+SAC-SMA into one container would erase
   an existing operational boundary; NOAA maintains them as separate modules.
2. **Reusability.** A standalone Snow17 Tesseract works with any downstream
   rainfall-runoff model. A fused blob works only in this pipeline.
3. **The rules.** Criterion 1 requires "two or more composed Tesseracts."
4. **FD cost, honestly stated:** see next section. The saving is real but
   modest once the coupling is priced; do not oversell it.

### Finite-difference cost analysis (be honest about this in the writeup)

One FD gradient direction = one extra forward run. Snow17 has ~13 learnable
parameters, SAC-SMA ~17 (verify exact count from its parameter file).

- **Merged container (rejected):** 30 nudges x (both models) ~= 60 model-runs
  per gradient; 17 of those pointlessly re-run Snow17 for parameters that
  cannot affect it.
- **Naive split:** 13 x Snow17 + 17 x SAC-SMA = 30 runs — BUT requires
  d(runoff)/d(RAIM), SAC-SMA's sensitivity to its input *time series*
  (~3,650 values for 10 years daily). Naive FD on that = ~3,650 runs.
  Catastrophic. Do not do this.
- **ADOPTED (option 1.5):** Snow17's parameter nudges flow through BOTH models,
  reading the change in final runoff (13 x both = 26 runs). SAC-SMA's parameter
  nudges run SAC-SMA only (17 runs). **~45 runs per gradient.** Never
  materializes the per-timestep flux Jacobian. Trade-off: the theta_A gradient
  path must be able to invoke the downstream model — document this coupling
  honestly.
- **Stretch (best-engineering territory):** exploit causality — RAIM(t) only
  affects runoff(t' >= t) and SAC-SMA memory is finite, so windowed re-runs
  could restore the elegant 30-run split. Only if ahead of schedule.

**OPEN DESIGN QUESTION** (resolve before writing the Tesseract wrappers): where
does the option-1.5 "nudge through both models" logic live within Tesseract's
endpoint contract? Likely answer: a thin orchestration layer in the training
code (src/coupling.py) implements the theta_A chain manually — calling
Tesseract A's `apply` with perturbed parameters and piping each result through
Tesseract B's `apply` — rather than relying on tesseract-torch's automatic
composition for that block. Prototype on day 3-4 before committing.

---

## Hard constraints

### License — non-negotiable

- Submission must be **Apache-2.0**. Terms also require: original work created
  during the hackathon period (Aug 3-31); no third-party IP infringement.
- snow17 and sac-sma (both NOAA-OWP) are Apache-2.0. Fine as dependencies,
  pinned as git submodules. Their READMEs ask that adaptations credit the
  source — **NOTICE file in the first commit.**
- **Do NOT use MHPI's `generic_deltamodel` (dMG) or `hydrodl2`** (PSU
  Non-Commercial license — incompatible with an Apache-2.0 submission and with
  the prize's collaboration component; Pasteur Labs is a company). Cite as
  methodological lineage only.
- NOAA-OWP's CONTRIBUTING releases all contributions to their repos into the
  public domain. **Never paste project code into a NOAA-OWP issue or PR** —
  describe bugs in prose, quote their code only.
- Dependency vs. authorship: using snow17/sac-sma as pinned dependencies is
  fine ("original work" applies to the submission, not the dependency tree).
  One README line settles it: "Snow17 and SAC-SMA are unmodified upstream
  dependencies (NOAA-OWP, Apache-2.0, pinned at commits X, Y). The shims,
  Tesseract wrappers, gradient endpoints, and training pipeline are original
  work written during the hackathon period."

### Rules and judging (verified against the official page)

- **Criterion 1 requires composing TWO OR MORE Tesseracts** across a real
  boundary ("the heart of the challenge"). Stated three times on the page.
- Criteria in order: (1) composition across a real boundary, (2) gradients
  doing real work — a visible objective improving on a problem unsolvable
  piecewise, (3) the why-Tesseract case — load-bearing, not costume, (4) a
  real application — undemoed domains score higher (**their gallery has no
  hydrology/climate**), (5) execution and technical depth (incl. forking
  Tesseract itself), (6) reproducibility and communication.
- Organizers, verbatim: "We'd rather see a modest problem where Tesseract is
  clearly load-bearing than an ambitious one where it's a costume."
- Deliverables: public GitHub repo + reproducible README; 2-4 page writeup (or
  detailed README) that **names the track**; optional <=5 min demo video.
- Submission form: https://tally.so/r/KYNZMg. Then a **LinkedIn post tagging
  Pasteur Labs & ISI and Tesseract** — a required part of submission.
- No rules about AI-assisted coding. Use Claude Code freely; understand every
  line (winners get feedback sessions and possible research collaboration).
- Prizes: $8k grand, $5k second, $1k x5 best-in-track, $1k best
  engineering/Tesseract hack, $1k best visual. Grand prize includes research
  collaboration with Pasteur Labs.
- Resources: cookiecutter-tesseract starter template, tesseract-torch, weekly
  office hours, forum at si-tesseract.discourse.group.
- Repo is currently **private** — flip public + submit form + LinkedIn post by
  **Aug 29** (calendar reminder set; hard deadline Aug 31 AoE).

---

## Snow17: everything established (one evening of archaeology — do not redo)

Upstream: `https://github.com/NOAA-OWP/snow17` (Apache-2.0, beta status).

### Use `EXSNOW19` directly — not the standalone driver, not BMI

Core physics = 13 plain FORTRAN-77 files in `src/snow19/`:

```
exsnow19.f  PACK19.f  SNOWPACK.f  SNEW.f     SNOWT.f   SNDEPTH.f  adjc19.f
aeco19.f    aesc19.f  melt19.f    rout19.f   updt19.f  zero19.f
```

`EXSNOW19` = single-timestep, single-HRU entry point, everything passed as
explicit arguments. No derived types, no namelist, no file I/O, no allocatables.
**Link these 13 files only** — never `src/share/`, `src/bmi/`, `src/driver/`
(that is where every bug found so far lives, and none of it is needed).

### Signature

```fortran
SUBROUTINE EXSNOW19(IDTS,IDT,IDA,IMN,IYR,
     &    PCP,TMP,RAIM,SNEQV,SNOW,SNOWH,          ! forcing in / diagnostics out
     &    ALAT,SCF,MFMAX1,MFMIN1,UADJ1,SI,NMF1,TIPM1,
     &    MBASE,PXTEMP,PLWHC,DAYGM,ELEV1,PA,ADC,  ! parameters
     &    CS,TPREV)                               ! carryover state (in/out)
```

- `RAIM` = rain-plus-melt, **mm depth per timestep**. The coupling flux to
  SAC-SMA.
- `SNEQV` = SWE in metres (TWE/1000) — diagnostic; expose it for validation
  plots.
- Learnable: `SCF, MFMAX1, MFMIN1, UADJ1, SI, NMF1, TIPM1, MBASE, PXTEMP,
  PLWHC, DAYGM` + `ADC(11)` depletion curve. `ALAT`, `ELEV1`, `PA` are basin
  attributes, not learnable.

### State contract — complete and explicit (verified in source)

```
CS(1..10)          = WE, NEGHS, LIQW, TINDEX, ACCMAX, SB, SBAESC, SBWS, STORGE, AEADJ
CS(11..10+NEXLAG)  = EXLAG(1..NEXLAG)      where NEXLAG = 5/IDT + 2  (integer div)
CS(11+NEXLAG)      = SNDPT
CS(12+NEXLAG)      = SNTMP
TPREV              = previous air temperature
```

Daily forcing => IDT=24 => NEXLAG=2 => CS uses indices 1-14 of 19.
`EXSNOW19` unpacks CS into COMMON `/SNCO19/` at entry (~lines 89-106) and
repacks at exit (~lines 168-184) — **verify these line numbers against the
fresh clone before relying on them.** `CS(19)` + `TPREV` is the COMPLETE
state; thread them explicitly and the call is deterministic. `/SNUP19/`
correction factors (MFC, UADJC, SFALLX, WINDC) are set to 1.0 by EXSNOW19
itself on every call; IUPWE=IUPSC=LMFV=LAEC=0 likewise. Cold start = CS=0.0,
TPREV=0.0.

### Units and calling convention (from runSnow17.f90 ~lines 158-170)

- `IDTS` = timestep in **seconds** (86400 daily); `IDT` = **hours** (24 daily)
- `PCP` in **mm depth per timestep**; `RAIM` returned in **mm depth**.
  Recommendation: mm/day everywhere; skip the driver's mm/s conversions; be
  explicit about units in the Tesseract schemas.
- `ELEV1` in **metres** (EXSNOW19 scales internally).
- **Melt factors internally rescaled:** MFMAX=(MFMAX1*IDT)/6.0 => x4 at daily.
  Learned MFMAX1 is in **mm/degC/6hr**. Critical for the recovered-parameter
  figure.
- Surface pressure computed in the shim (Anderson 2006; constants verified
  against constants.f90):
  `pa = 33.86 * (29.9 - 0.335*(elev/100.0) + 0.00022*((elev/100.0)**2.4))`

### KNOWN BUG — ADC dimension mismatch (workaround required)

`exsnow19.f` declares `REAL ADC(11)`; `PACK19.f` declares `DIMENSION ADC(12)`;
the upstream caller passes an 11-element slice => out-of-bounds read every
timestep, silently. **Workaround in shim:** `adc12(1:11)=adc11; adc12(12)=1.0`.
Before filing upstream: check whether PACK19 actually reads element 12, and
reproduce on the HPC toolchain (Linux, older gfortran).

### Other upstream findings (macOS arm64, gfortran 16.1.0 — may be compiler artifacts)

1. `test_cases/ex1` namelist: `start_datehr`/`end_datehr` supplied unquoted but
   declared CHARACTER(len=10) => "Missing quote while reading item 7". Fix:
   quote them.
2. After (1): `read_snow17_parameters` fails on an invalid `adc` array
   descriptor despite initParams allocating it. **Unresolved; recheck on
   Linux.**
3. `ioModule.f90` contains commented-out transposed adc indexing (incomplete
   refactor).
4. README documents build target `snow17_bmi`; actual target is `snow17bmi`.
   **Issue + PR filed** (URLs in notes/NOTES.md).
5. Upstream PR #47 / issue #44 fix a different bug class (implicit-SAVE from
   initialization-at-declaration) in dateTimeUtilsModule/ioModule. Not in our
   dependency path, but a warning: check `snow19/*.f` for DATA-initialized
   variables that are also assigned during execution.

Bugs 1-3 do not affect our build (we never link src/share/). Validation
reference: `test_cases/ex1/output/snowout.orig.HHWM8.txt.dev` (output from the
original NCAR version).

---

## SAC-SMA: archaeology NOT yet done (day 1-2 on HPC)

Upstream: `https://github.com/NOAA-OWP/sac-sma` (Apache-2.0). Sibling repo to
snow17 — same owp-open-source-project-template, same namelist conventions, same
test basin (HHWM8), Makefile-based build (`Makefile.local`; gfortran
supported). Snow17's CMakeLists even contains a stray `${SAC_LIB_NAME_CMAKE}`
reference — shared ancestry. Expect the same architecture: an F77 physics core
plus a modern-Fortran share/BMI wrapper (with the same bug classes).

**Repeat the Snow17 methodology exactly:**

1. Find the F77 physics core (the historical NWS entry point is a subroutine
   named ~`SAC1`/`FLAND1` — locate the EXSNOW19-equivalent with explicit
   arguments).
2. Reverse-engineer the state contract. SAC-SMA states: UZTWC, UZFWC, LZTWC,
   LZFSC, LZFPC, ADIMC (upper/lower zone tension/free water contents). Find
   how they thread (the equivalent of CS/TPREV).
3. Identify learnable parameters (~16: UZTWM, UZFWM, UZK, PCTIM, ADIMP, RIVA,
   ZPERC, REXP, LZTWM, LZFSM, LZFPM, LZSK, LZPK, PFREE, SIDE, RSERV). Verify
   against their parameter file.
4. Establish unit conventions for the input flux (RAIM in — units, per
   timestep?) and outputs (runoff components: surface, interflow, baseflow,
   total; plus ET handling — SAC-SMA needs PET forcing; decide source:
   CAMELS provides daily PET estimates).
5. Check the same bug classes: array dimension mismatches at call sites,
   DATA-initialized variables assigned during execution, COMMON-block state
   not covered by explicit state arguments.
6. Write `fortran/sacsma_shim.f90` mirroring the snow17 shim: loop the kernel
   over a time series, state threaded explicitly, bind(C).
7. **Same three tests** (below) before anything else consumes its output.

The shared HHWM8 test case may include a worked Snow17->SAC-SMA chain — if so,
that is the validation target for the coupled pipeline.

---

## Fortran shims — pattern established for Snow17; replicate for SAC-SMA

`fortran/snow17_shim.f90`:

```fortran
subroutine snow17_run(n, idt, idts, iyr, imn, ida,             &
                      pcp, tmp,                                &
                      alat, elev, scf, mfmax, mfmin, uadj, si, &
                      nmf, tipm, mbase, pxtemp, plwhc, daygm,  &
                      adc11,                                   &
                      cs_io, tprev_io,                         &
                      raim, sneqv) bind(C, name="snow17_run")
```

Settled design decisions:
- **Dates as arrays** (iyr(n), imn(n), ida(n)) built in pandas — no Fortran
  calendar arithmetic.
- pa computed in the shim from elev (formula above).
- adc11 -> adc12 expansion (bug workaround above).
- F77 default REAL is 4-byte: use real(c_float); never -fdefault-real-8.
- **OPEN:** check whether PACK19 writes back to TPREV (grep PACK19.f for TPREV
  assignments) before deciding if the shim sets tprev=tmp(t) after each call.

Build (adjust for HPC; prefer building inside the container that will ship):

```bash
gfortran -shared -fPIC -O2 -o fortran/libsnow17shim.so \
  external/snow17/src/snow19/*.f fortran/snow17_shim.f90
```

Use `-fcheck=bounds` during development (it will flag the ADC issue —
expected); drop it for release once the 12-element workaround is confirmed.

### THREE TESTS — write before the Tesseract wrappers, for EACH shim

Each catches a failure mode that produces *plausible but wrong* output:

1. **Determinism.** Two identical calls => bitwise-identical output. Catches
   COMMON-block leakage; without it, FD gradients are silently garbage (the
   perturbed run inherits state from the base run).
2. **State continuity.** One 100-day call == two chained 50-day calls,
   bitwise. The sharper version of (1); trust this one most.
3. **Mass balance.** Water year: sum(raim) + delta_SWE ~= sum(pcp)*SCF for
   Snow17; sum(runoff) + delta_storage ~= sum(raim) - ET for SAC-SMA. Catches
   unit errors — the likeliest defect given the mm/s vs mm/step conventions
   being deliberately skipped.

Context: a prior debugging episode (CudnnLstmModel in HydroDL) involved
silently-zero gradients that looked fine until checked directly. Same failure
shape. These tests are not optional.

---

## Tesseract specifics

- Two Tesseracts: `tesseracts/snow17/` and `tesseracts/sacsma/`, each with
  tesseract_api.py, tesseract_config.yaml, tesseract_requirements.txt.
- `apply`: forcings + parameters + initial state -> outputs + final state.
- `vector_jacobian_product`: finite differences over the scalar parameters
  (explicitly permitted: "The solver may expose its Jacobian by autodiff or by
  finite differences; the composition with the inference engine is the
  contribution").
- Gradient flow follows **option 1.5** (FD cost section): resolve the open
  design question about where the through-both-models nudge logic lives BEFORE
  writing the wrappers. Prototype day 3-4.
- **Verify each VJP against a manual perturbation** (tests/test_gradients.py).
- Wrap at **full-rollout granularity** (one call per basin per epoch), never
  per-timestep (Tesseract targets kernels running seconds+).
- `abstract_eval` is cheap; implement it.
- **PXTEMP rain/snow threshold is a hard step** => d(loss)/d(PXTEMP) ~ 0
  almost everywhere under small-eps FD. Options: (a) sigmoid-relax in the
  shim, (b) use a large FD step (~0.5-1.0 degC) as implicit smoothing, stated
  openly. SAC-SMA likely has analogous thresholds — identify during
  archaeology. Discuss the treatment in the writeup; it reads as competence.
- Do NOT attempt Enzyme/Flang differentiation of the Fortran — timeline
  killer. Mention as future work.
- Consider the `cookiecutter-tesseract` starter template for boilerplate.

---

## Repo layout

```
dsnowsac/                        # δSnowSac; renamed from dsnow17-sacsma (was dsnow17-hbv)
├── LICENSE                      # Apache-2.0, first commit
├── NOTICE                       # credit NOAA-OWP/snow17 AND NOAA-OWP/sac-sma
├── README.md                    # one-command reproduction; dependency-vs-
│                                #   authorship statement (License section)
├── external/
│   ├── snow17/                  # git submodule, pinned
│   └── sac-sma/                 # git submodule, pinned
├── fortran/
│   ├── snow17_shim.f90
│   └── sacsma_shim.f90
├── tesseracts/
│   ├── snow17/
│   └── sacsma/
├── src/
│   ├── paramnet.py              # LSTM/MLP: attributes -> both parameter sets
│   ├── coupling.py              # option-1.5 gradient orchestration
│   └── train.py
├── tests/
│   ├── test_snow17_shim.py      # three tests
│   ├── test_sacsma_shim.py      # three tests
│   └── test_gradients.py        # VJP vs manual perturbation, per Tesseract
└── notes/NOTES.md               # upstream bug log -> writeup motivation
```

---

## Timeline (from Aug 10; ~8-10 real working days; deadline Aug 31 AoE)

| Days | Milestone |
|---|---|
| 1-2 | HPC setup (modules/container; gfortran version logged). Build snow19 + shim; three tests pass. Reproduce/refute the macOS bugs; file the ADC issue upstream if confirmed. Start sac-sma archaeology. |
| 3-4 | SAC-SMA state contract + shim + three tests. Resolve the option-1.5 design question. Snow17 Tesseract with FD VJP; gradient check passes. |
| 5-6 | SAC-SMA Tesseract. Chain: Snow17 -> SAC-SMA -> NSE -> backward. Single basin (HHWM8 or a CAMELS snow basin), direct parameter optimization. **Loss goes down = submittable.** |
| 7-10 | Parameter network, 30-50 snow-dominated CAMELS basins, held-out basins. |
| 11-14 | Figures + writeup (2-4 pages, names Track 03). |
| 15-19 | Buffer. **Aug 29:** flip repo public, submit form, LinkedIn post tagging Pasteur Labs & ISI + Tesseract. |

**HARD STOP — Aug 20:** if end-to-end gradients through BOTH Tesseracts are not
flowing, drop SAC-SMA, write a minimal HBV in PyTorch (native, ~200 lines,
Feng et al. 2022 formulation), and submit Snow17-Tesseract + native-HBV
single-basin optimization with an honest note about the two-Tesseract
requirement. A truthful near-miss beats a costume. (This is the ONLY context
in which HBV appears.)

Best-visual candidates: (1) recovered MFMAX1 (mm/degC/6hr) vs. basin
elevation / mean winter temperature across CAMELS basins — the network
recovering physics, not fitting noise; (2) a gradient-flow diagram through the
two containers with the RAIM coupling annotated.

---

## How to work with me on this

- Be exacting and critical. Push back on bad ideas directly; no softening.
- Flag scope creep hard. Known failure mode #1: reaching for a new interesting
  direction while the finishable thing sits unfinished. Known failure mode #2
  (from the Snow17 evening): debugging code paths we do not depend on. Timebox
  archaeology; the dependency surface is 13 .f files + the sac-sma kernel
  equivalents.
- When something looks like it works, ask what test would prove it doesn't.
  Silent wrongness is the enemy, not loud failure.
- Verify all cited line numbers against fresh clones before relying on them.