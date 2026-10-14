# pi–hypertriton femto offline analysis workflow

This note describes the recommended offline workflow for the pi–hypertriton femto analysis using the current AO2D output tables from `HadNucleiFemto`.

The current design writes pair-level tables:

- data: `HADHYPERTABLE`
- MC: `HADHYPERMC`

Each row corresponds to one `hypertriton candidate + pair pion` pair. Same-event and mixed-event pairs are stored in the same table and are separated by the `IsMixed` column.

The current `HadNucleiFemto` hypertriton workflow uses one process switch for data and one process switch for MC:

```text
processHyper    -> data same-event and mixed-event hadron-hypertriton pairs
processMCHyper  -> MC same-event and mixed-event hadron-hypertriton pairs
```

Each process scans the input once, fills same-event rows with `IsMixed == false`, and, if mixing is enabled, fills mixed-event rows with `IsMixed == true`.

The process names are intentionally generic hadron-hypertriton names: the nominal analysis uses pions as the hadron species, but the producer side can follow the configured hadron PID species.

## 1. Input tables

### Data table

Use:

```text
HADHYPERTABLE
```

Important columns:

```text
IsMixed
ChargeChannel
MixingDepth

HypSourceCollisionId
HypGlobalBC
HypRunNumber
HypPosZ
HypCentFT0A
HypCentFT0C
HypCentFT0M
HypTrackOccupancy
HypFt0cOccupancy

PairSourceCollisionId
PairPosZ
PairCentFT0A
PairCentFT0C
PairCentFT0M
PairTrackOccupancy
PairFt0cOccupancy

SourceCandidateId
IsMatter

PtHe3
EtaHe3
PhiHe3
PtPi
EtaPi
PhiPi

DcaV0Daug
DcaHe
DcaPi
NSigmaHe
TofMass

XPrimVtx
YPrimVtx
ZPrimVtx
XDecVtx
YDecVtx
ZDecVtx
```

The table stores a full track-level QA block for the pair pion used in the femto pair.

The hypertriton daughters are not stored as separate full track-QA blocks. Their candidate-level quantities are already available through the hypertriton candidate block, for example `PtHe3`, `EtaHe3`, `PhiHe3`, `PtPi`, `EtaPi`, `PhiPi`, `DcaHe`, `DcaPi`, TPC-cluster information, TPC signal, ITS chi2, and ITS cluster sizes.

These columns are intended for offline QA, BDT input variables, and systematic variations while keeping the pair-level AO2D output reasonably compact.

### MC table

Use:

```text
HADHYPERMC
```

`HADHYPERMC` contains the same reconstructed information as `HADHYPERTABLE`, plus MC-truth columns for validation:

```text
HypMcParticleId
HypMcCollisionId
HypPdgCode
HypGenPt
HypGenEta
HypGenPhi
HypGenPtHe3
HypGenXDecVtx
HypGenYDecVtx
HypGenZDecVtx
HypIsReco
HypIsFakeHeOnITSLayer
HypIsSignal
HypIsRecoMCCollision
HypIsSurvEvSel
HypIsTwoBodyDecay
HypStatusCode

HeMcParticleId
HeMcCollisionId
HePdgCode
HeGenPt
HeGenEta
HeGenPhi
HeIsPhysicalPrimary

DecayPiMcParticleId
DecayPiMcCollisionId
DecayPiPdgCode
DecayPiGenPt
DecayPiGenEta
DecayPiGenPhi
DecayPiIsPhysicalPrimary

PairPiMcParticleId
PairPiMcCollisionId
PairPiPdgCode
PairPiGenPt
PairPiGenEta
PairPiGenPhi
PairPiIsPhysicalPrimary
PairPiStatusCode
PairPiProcess

SameMCCollision
MatchesHypRecoMCCollision
MatchesPairRecoMCCollision
IsTruthSelfCorrelation
IsTruePrimaryPiHyperPair
KstarMC
```

## 2. Offline derived variables

Several variables are intentionally not stored in the AO2D table because they can be reconstructed offline.

For each pair row, rebuild the hypertriton candidate from its daughter tracks:

```text
He3 daughter:
  PtHe3, EtaHe3, PhiHe3, mass = mHe3

decay pion:
  PtPi, EtaPi, PhiPi, mass = mPi

hypertriton candidate:
  p(H3L) = p(He3) + p(pi_decay)
```

Then derive:

```text
massH3L
ptH3L
etaH3L
phiH3L
ct
cosPA
decay radius
```

The pair pion four-vector is built from:

```text
PairPiTrackPt
PairPiTrackEta
PairPiTrackPhi
mass = mPi
```

The femto observable is:

```text
kstar = momentum of either particle in the pair center-of-mass frame
```

For MC, compare the reconstructed value against:

```text
KstarMC
```

to check the k* resolution and possible reconstruction bias.

## 3. PbPb centrality treatment

For PbPb analysis, use the hypertriton-side event centrality as the default binning variable:

```text
HypCentFT0C
```

The pair-pion-side event centrality is also stored:

```text
PairCentFT0C
```

For same-event pairs these should be identical. For mixed-event pairs they are generally close but not identical because the hypertriton and pair pion come from different events in the same mixing pool.

The recommended convention is:

```text
centrality bin = bin(HypCentFT0C)
```

This keeps the rare hypertriton candidate as the anchor of the analysis.

This convention is especially important for mixed-event pairs. In mixed events, the hypertriton candidate and the pair pion come from two different collisions:

```text
HypCentFT0C  = centrality of the hypertriton-candidate event
PairCentFT0C = centrality of the mixed pair-pion event
```

Even for mixed-event pairs, fill the femto centrality bin using:

```text
cent = HypCentFT0C
```

The mixed-event pion centrality should not define the final femto centrality bin. It should instead be used to validate the event-mixing quality.

Recommended QA:

```text
DeltaCent = PairCentFT0C - HypCentFT0C
PairCentFT0C versus HypCentFT0C
ME k* distribution in different DeltaCent windows
```

If the event-mixing centrality bins are sufficiently fine, the `DeltaCent` distribution should be narrow. Large tails indicate poor centrality matching in the mixed-event pool.

Recommended systematic checks:

```text
vary the event-mixing centrality bin width
require |PairCentFT0C - HypCentFT0C| < 2%
require |PairCentFT0C - HypCentFT0C| < 5%
```

If the final correlation function is stable against these variations, the centrality treatment in mixed events is considered safe.

## 4. Charge-channel convention

The current charge-channel definition is:

```text
0 = H3L + pi+
1 = H3L + pi-
2 = anti-H3L + pi-
3 = anti-H3L + pi+
```

Useful combinations:

```text
same sign:
  H3L + pi+
  anti-H3L + pi-

opposite sign:
  H3L + pi-
  anti-H3L + pi+
```

The recommended first check is to plot all four channels separately. If matter and antimatter are compatible, combine them later to improve statistics.

## 5. Candidate-level BDT training

The AO2D table is pair-level, so the same hypertriton candidate can appear multiple times because it is paired with many pions.

Do not train the hypertriton BDT directly on every pair row. That would duplicate candidates and bias the training toward high-multiplicity or high-pair-count events.

Instead, first build a unique candidate-level sample.

Recommended candidate key:

```text
HypRunNumber
HypGlobalBC
HypSourceCollisionId
SourceCandidateId
```

Within a single AO2D file, this may often be sufficient:

```text
HypSourceCollisionId
SourceCandidateId
```

For merged files, prefer the full key with run number and global BC.

The BDT training workflow should be:

```text
1. Read HADHYPERTABLE or HADHYPERMC.
2. Build derived hypertriton variables.
3. Deduplicate rows using the candidate key.
4. Train or apply the hypertriton BDT on the unique candidate sample.
5. Merge the BDT score back into the pair-level table using the same candidate key.
6. Apply the BDT selection on pair-level rows.
7. Build SE and ME k* distributions.
```

Candidate-level BDT variables can include:

```text
massH3L
ptH3L
ct
cosPA
decay radius
DcaV0Daug
DcaHe
DcaPi
NSigmaHe
TofMass
He3 TPC/ITS QA
decay pion candidate-level TPC/ITS QA
occupancy
centrality
```

The pair pion should not be used to train the hypertriton BDT, unless the BDT is explicitly designed as a pair-level classifier. For the nominal hypertriton selection, keep the BDT candidate-level.

## 6. Global training versus bin-dependent training

Recommended nominal strategy:

```text
Train one global BDT using the full selected sample.
Apply the trained BDT in centrality, pt, ct, and k* differential analyses.
```

This is the most stable baseline.

Possible systematic variation:

```text
Train separate BDTs in hypertriton pt or centrality bins.
```

Bin-dependent training can be useful, but it requires enough MC signal and background statistics in every bin. It also changes the pair acceptance and therefore needs MC closure checks.

Recommended treatment:

```text
Nominal:
  global BDT

Systematic:
  bin-dependent BDT or BDT-score variation
```

## 7. Same-event and mixed-event construction

Use the `IsMixed` column:

```text
same-event:
  IsMixed == false

mixed-event:
  IsMixed == true
```

For each analysis bin, fill:

```text
SE(k*) = same-event distribution
ME(k*) = mixed-event distribution
```

Then build:

```text
C(k*) = alpha * SE(k*) / ME(k*)
```

where `alpha` is a normalization factor obtained in a high-k* region.

Example normalization range:

```text
0.25 < k* < 0.45 GeV/c
```

The exact normalization range should be varied as a systematic uncertainty.

## 8. Hypertriton combinatorial background

The mixed-event distribution corrects pair acceptance, but it does not remove combinatorial hypertriton background.

Use the reconstructed invariant mass to separate:

```text
signal window
left sideband
right sideband
```

Recommended correction:

```text
C_raw(k*) = correlation from the signal mass window
C_bkg(k*) = correlation from sideband candidates
P         = signal purity from invariant-mass fit
```

Then:

```text
C_true(k*) = [C_raw(k*) - (1 - P) * C_bkg(k*)] / P
```

The signal purity should be evaluated in the same analysis binning whenever statistics allow:

```text
centrality
hypertriton pt
BDT-score region
```

If the statistics are limited, use a coarser purity extraction and propagate the difference as a systematic uncertainty.

## 9. Occupancy treatment

Both hypertriton-side and pair-pion-side occupancy information are stored:

```text
HypTrackOccupancy
HypFt0cOccupancy
PairTrackOccupancy
PairFt0cOccupancy
```

Recommended baseline:

```text
Use HypTrackOccupancy or HypFt0cOccupancy for event-level occupancy selection.
```

For mixed-event pairs, the pion comes from a partner event, so the pair-side occupancy is also available for QA:

```text
PairTrackOccupancy
PairFt0cOccupancy
```

Useful checks:

```text
SE occupancy distribution
ME occupancy distribution
occupancy dependence of mass
occupancy dependence of BDT score
occupancy dependence of C(k*)
```

Occupancy cuts should be varied as part of the systematic uncertainty.

## 10. Systematic uncertainty strategy

For each systematic variation, rerun the full offline chain:

```text
candidate selection
BDT application
SE and ME construction
ME normalization
purity or sideband correction
final C(k*)
```

Then evaluate the uncertainty per k* bin using either RMS or envelope of accepted variations.

### Candidate selection variations

Vary:

```text
DcaV0Daug
DcaHe
DcaPi
cosPA
ct
decay radius
mass window
TofMass
NSigmaHe
```

### Daughter-track QA variations

Vary:

```text
TPC clusters
TPC crossed rows
TPC PID clusters
TPC chi2
ITS chi2
ITS cluster size
hasITS
hasTPC
hasTOF
```

### Pair-pion selection variations

Vary:

```text
PairPiTrackDcaXY
PairPiTrackDcaZ
PairPiTrackTpcNClsFound
PairPiTrackTpcNClsCrossedRows
PairPiTrackTpcNSigmaPi
PairPiTrackTofNSigmaPi
PairPiTrackEta
PairPiTrackPt
```

### BDT variations

Vary:

```text
BDT score cut
training sample definition
background definition
input variable set
global BDT versus bin-dependent BDT
```

### Event mixing variations

Vary:

```text
z-vertex binning
centrality binning
mixing depth
mixed-event normalization range
occupancy selection
run-by-run splitting
```

### Mass and purity variations

Vary:

```text
signal mass window
sideband window
mass-fit range
background function
signal shape
purity extraction binning
```

### Matter and charge-channel checks

Check separately:

```text
matter
antimatter
same sign
opposite sign
all four ChargeChannel values
```

If compatible, combine matter and antimatter. If not compatible, keep them separated or include the difference as a systematic uncertainty.

## 11. MC validation

Use:

```text
HADHYPERMC
```

The MC validation should check both the hypertriton reconstruction and the femto pair construction.

### Truth signal selection

Check:

```text
HypIsSignal == true
abs(HypPdgCode) == hypertriton PDG
HypIsTwoBodyDecay == true
```

Study these distributions:

```text
massH3L
ptH3L
ct
cosPA
decay radius
BDT score
centrality
occupancy
```

### Reco-MC collision matching

Check:

```text
MatchesHypRecoMCCollision
MatchesPairRecoMCCollision
SameMCCollision
```

For same-event true pairs, `SameMCCollision` should usually be true. For mixed-event pairs, it is expected to be false in most cases.

### Self-correlation validation

Check:

```text
IsTruthSelfCorrelation
```

This should be zero or extremely small after the online table production because truth-level self-correlations are rejected before output.

The workflow also fills self-correlation QA histograms, including a dedicated multidimensional histogram:

```text
hSelfCorrelationQA
```

with axes:

```text
reason
mixed flag
k*
centrality
```

If non-zero entries are observed, inspect whether the pair pion is:

```text
the same track as the He3 daughter
the same track as the hypertriton decay pion
a daughter of the same hypertriton mother
```

### k* resolution check

Compute offline:

```text
kstarReco
```

from reconstructed pair momenta and compare against:

```text
KstarMC
```

Study:

```text
deltaKstar = kstarReco - KstarMC
deltaKstar / KstarMC
```

as functions of:

```text
kstar
centrality
hypertriton pt
pair pion pt
occupancy
```

### Purity and pair truth

Use:

```text
HypIsSignal
IsTruePrimaryPiHyperPair
PairPiIsPhysicalPrimary
SameMCCollision
```

to estimate:

```text
hypertriton signal purity
primary pair-pion purity
true primary pi-hypertriton pair fraction
```

These quantities should be checked versus:

```text
kstar
centrality
mass window
BDT score
occupancy
```

### Femto closure

Run the full offline chain on MC:

```text
1. Apply the same reconstruction-level selections as in data.
2. Build SE and ME k* distributions.
3. Apply the same ME normalization.
4. Apply the same purity or sideband correction strategy.
5. Check whether the corrected C(k*) closes to the expected MC baseline.
```

If the generator does not contain a femtoscopic final-state interaction signal, the corrected correlation function should be close to unity within uncertainties.

Significant deviations may indicate:

```text
bad event mixing
self-correlation leakage
centrality or occupancy mismatch
reconstruction-induced k* bias
BDT-induced pair acceptance distortion
incorrect purity correction
```

## 12. Recommended analysis sequence

The recommended practical workflow is:

```text
1. Produce HADHYPERTABLE for data and HADHYPERMC for MC.
2. Read the pair-level tables offline.
3. Rebuild hypertriton four-vectors from He3 + decay pion.
4. Recompute mass, pt, ct, cosPA, decay radius, and kstar.
5. Build a unique candidate-level table using:
     HypRunNumber, HypGlobalBC, HypSourceCollisionId, SourceCandidateId
6. Train or apply the hypertriton BDT on the unique candidate sample.
7. Merge the BDT score back into the pair-level table.
8. Apply candidate, track, event, occupancy, and BDT selections.
9. Fill SE and ME k* distributions using IsMixed.
10. Split by centrality and ChargeChannel.
11. Normalize ME in a high-k* region.
12. Build raw C(k*).
13. Correct for hypertriton combinatorial background using sidebands or purity.
14. Validate with MC:
      truth signal,
      self-correlation,
      k* resolution,
      pair purity,
      femto closure.
15. Evaluate systematic uncertainties with the full variation ensemble.
```

## 13. Minimal pseudocode

```python
for row in HADHYPERTABLE:
    hyp_p4 = make_p4(row.PtHe3, row.EtaHe3, row.PhiHe3, mHe3)
    hyp_p4 += make_p4(row.PtPi, row.EtaPi, row.PhiPi, mPi)

    pair_pi_p4 = make_p4(
        row.PairPiTrackPt,
        row.PairPiTrackEta,
        row.PairPiTrackPhi,
        mPi,
    )

    mass = hyp_p4.mass()
    kstar = compute_kstar(hyp_p4, pair_pi_p4)
    cent = row.HypCentFT0C
    channel = row.ChargeChannel

    if not pass_candidate_selection(row, mass):
        continue
    if not pass_pair_pion_selection(row):
        continue
    if not pass_occupancy_selection(row):
        continue
    if not pass_bdt_selection(row):
        continue

    if row.IsMixed:
        fill_ME(kstar, mass, cent, channel)
    else:
        fill_SE(kstar, mass, cent, channel)

for analysis_bin in bins:
    alpha = normalize_ME_to_SE(high_kstar_range)
    C_raw = alpha * SE / ME
    C_true = purity_correct(C_raw, sideband_correlation, purity)
```

## 14. Main rule of thumb

Use the pair-level table for the femto correlation function.

Use a deduplicated candidate-level view for hypertriton BDT training.

Use the MC table for truth validation, purity checks, self-correlation checks, k* resolution, and closure.
