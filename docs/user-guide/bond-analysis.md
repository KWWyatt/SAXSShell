# Bond Analysis

The **Bond Analysis** tool is SAXSShell's structure-analysis application for
measuring bond-length, angle, dihedral, and coordination distributions from
stoichiometry-sorted cluster folders.

## Launching the application

Open the tool from the main SAXS UI through
`Tools > Structure Analysis > Open Bond Analysis`.

When the window is launched from an active SAXS project, the current project
and cluster-folder reference are carried into the tool. If you change the
selected clusters folder there, that reference is saved back to the project.

## What the tool does

The current UI supports:

- choosing one sorted clusters directory as the analysis source
- saving results into a separate output directory
- limiting the run to checked stoichiometry labels
- defining bond-pair cutoffs directly in a table
- defining angle triplets directly in a table
- defining signed dihedral quartets directly in a table
- loading built-in presets and saving custom presets for later reuse
- reopening an existing bond-analysis output folder and browsing its saved
  distributions

The right side of the window focuses on the computed distributions. You can
refresh a results directory, select one or more saved bond-pair, angle,
dihedral, or coordination entries, and open them in a dedicated plot window.
Matching items from multiple cluster types can be overlaid together for
comparison.

## Typical workflow

1. Start from the project's sorted clusters folder.
2. Confirm or choose the bond-analysis output directory.
3. Refresh the detected cluster types and clear any stoichiometries you do not
   want to include.
4. Load a preset or define the bond pairs, angle triplets, and dihedral
   quartets manually.
5. Run the calculation and inspect the saved distributions from the results
   browser.

Dihedral quartets use adjacent-pair cutoffs: `ATOM1-ATOM2`, `ATOM2-ATOM3`,
and `ATOM3-ATOM4` must each be within their requested cutoff. The reported
values are signed degrees in `[-180, 180]`. The calculation projects the two
outer bonds onto the plane perpendicular to the middle bond and uses `atan2` to
keep the torsion sign. `-180` and `+180` are the same anti-aligned planar
torsion at the wrap boundary, while `0` is the aligned planar torsion and is
not equivalent to `180`.

When plotting dihedral distributions, the saved values are not modified, but
the plot display is recentered when helpful. The display center snaps to the
nearest clean multiple of 90 degrees (`0`, `+90`, `-90`, or `180`) based on the
circular center, so a wrapped population near `-180` / `+180` appears as one
continuous peak centered near `180` instead of being split across both plot
edges. Even when the display is internally recentered, tick labels and summary
values remain in the signed `[-180, 180]` convention. Dihedral plots also
provide a plot-style toggle between a normal histogram and a radial histogram.
The radial view draws all dihedral values on one circular axis, which is useful
for combined nearly degenerate terminal distributions such as `O-C-N-C` where
one branch may populate `0` and another may populate the `-180` / `+180`
boundary.

Each histogram CSV includes ordinary distribution statistics plus
GDS-oriented metadata. Bond-distance histograms add `gds_center_angstrom`,
`gds_sigma_angstrom`, and `gds_sigma2_angstrom_squared`. Angle and dihedral
histograms add `gds_center_degrees`, `gds_sigma_degrees`,
`gds_center_radians`, `gds_sigma_radians`, and
`gds_variance_radians_squared`. The same metadata also includes
`gds_*_variable` names and paste-ready Artemis `set` rows. The run's
`bondanalysis_results_index.json` registers those variables across the saved
cluster and aggregate distributions. Dihedral GDS centers and widths use
circular statistics, so a population split across the `-180` / `+180` histogram
edge is treated as one wrapped distribution instead of being averaged toward
`0`.

## EXAFS GDS handoff

After representative structures and bondanalysis are complete, open
`Tools > Structure Analysis > Open EXAFS GDS Mapping` from the main SAXS UI.
The mapping window loads the project representative-structure metadata and
completed bondanalysis result folders, then lets you choose a stoichiometry
representative, inspect labeled 3D and 2D absorber-scatterer path diagrams,
select bond, angle, and dihedral registry variables, preview the generated GDS,
and write a validated Artemis setup file.

The bondanalysis variables are imported as GDS-ready statistical `set` rows.
They provide the distribution centers and sigmas that can anchor the GDS
constraint setup while the EXAFS mapping tool builds the path/template side
from the selected representative structure. Hydrogen-family atoms are excluded
from the mapping diagrams and generated EXAFS paths. The absorber defaults to
Pb when Pb is present, and can be changed to another non-hydrogen coordination
center in the mapping window.

## Related pages

- [MD Extraction and Cluster Preparation](cluster-extraction.md)
- [Debye-Waller Analysis](debye-waller-analysis.md)
- [GUI Overview](gui-overview.md)
