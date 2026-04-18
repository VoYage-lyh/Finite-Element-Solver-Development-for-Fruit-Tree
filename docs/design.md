# Orchard Vibration Solver Design

## 1. Project Positioning

### 1.1 Goal

This project targets a dedicated fast solver for orchard tree vibration analysis, especially for vibration harvesting, shaker excitation response, clamp-position comparison, parameter scanning, and reduced-order modeling research.

The solver is **not** intended to replicate a general-purpose commercial FEM package. Instead, it is built around orchard-specific structural and dynamical traits:

- hierarchical trunk-branch-fruit topology,
- non-circular and spatially varying branch cross-sections,
- tissue-partitioned material behavior,
- localized strong nonlinearities at joints, clamps, and fruit attachments,
- controllable simplification for fast repeated simulation.

### 1.2 Primary Objectives

- Preserve the physical mechanisms that most affect harvest-relevant response metrics.
- Accelerate simulation for parameter sweeps, frequency-amplitude searches, and clamp-placement studies.
- Keep model reduction and error assessment as first-class architectural concerns.
- Provide a testable, extensible, research-grade codebase that can evolve from MVP to higher fidelity.

## 2. Target Scenarios

The first versions focus on the following use cases:

- Dynamic response of a tree under clamp excitation.
- Frequency response comparison under different excitation positions.
- Fruit-bearing branch response under added fruit mass and fruit-stem connection effects.
- Sensitivity analysis for branch material, section variation, and local nonlinear joint parameters.
- Offline reference-vs-fast-model comparison for error-controlled acceleration.

Out-of-scope for MVP:

- Full 3D continuum contact analysis over the entire tree.
- General CAD/mesh workflows for arbitrary solids.
- A one-size-fits-all element library for unrelated engineering domains.

## 3. Why This Is an Orchard-Specific Solver

This solver differs from general FEM in both modeling philosophy and software structure.

### 3.1 Domain-Driven Simplification

The entire tree is not treated uniformly. We retain complexity where orchard dynamics are most sensitive:

- bifurcation and branch-joint regions,
- clamp/excitation regions,
- fruit attachment locations,
- large-amplitude response regimes.

Less sensitive slender regions are modeled with lower-cost structural abstractions, while still carrying non-circular section and tissue-partition information into effective properties.

### 3.2 Orchard-Specific Structural Representation

Instead of a generic mesh-first workflow, the model starts from:

- tree topology,
- branch centerlines,
- branch-wise cross-section evolution,
- tissue partitions: xylem, pith, phloem,
- fruit locations and attachment models,
- clamp and observation definitions.

### 3.3 Response-Oriented Design

The code is optimized for outputs relevant to vibration harvesting:

- displacement/acceleration at fruit-bearing points,
- frequency-response peaks,
- transmitted energy along the branch hierarchy,
- local amplification near fruits and bifurcations,
- computation time vs. response error tradeoff.

## 4. Modeling Assumption Tiers

We explicitly classify assumptions into three levels.

### 4.1 Must Preserve

- Hierarchical tree topology and parent-child branch relations.
- Non-circular or irregular cross-section description.
- Axially varying cross-section along a branch.
- Explicit tissue partitions for xylem, pith, and phloem.
- Fruit mass and attachment coupling.
- Clamp excitation and orchard-specific boundary conditions.
- Local nonlinear behavior at joints, clamp region, and selected strong-response zones.
- Error assessment against higher-fidelity or reference models.

### 4.2 Can Be Simplified in MVP

- Detailed continuum discretization outside hotspot regions.
- Full anisotropic constitutive behavior; interface is preserved, isotropic/orthotropic placeholders may be used first.
- Advanced hysteresis and friction models; interface is preserved, simple polynomial or piecewise nonlinear laws may be used first.
- Rich damping models; equivalent viscous damping may be used initially, with extension points for viscoelastic or nonlinear damping.
- Full nonlinear geometry everywhere; initial implementation may use low-order nonlinear corrections or localized nonlinear internal-force terms.

### 4.3 Deferred Beyond MVP

- Automated high-fidelity local submodel generation from scanned geometry.
- Hyper-reduction and empirical interpolation beyond interface stubs.
- Parallel distributed solves.
- Inverse identification and uncertainty quantification pipelines.

## 5. Architecture Overview

### 5.1 Layered Architecture

```text
Input JSON/YAML
    |
    v
Model Builder / Validation
    |
    +--> geometry_topology
    +--> cross_section
    +--> materials
    +--> branches
    +--> joints_and_bifurcations
    +--> fruits
    +--> excitation_and_bc
    |
    v
Discretization / DOF Allocation
    |
    +--> low-cost global structural model
    +--> optional local enriched regions
    |
    v
solver_core
    |
    +--> assembly
    +--> time-domain solver
    +--> frequency-response solver
    +--> parameter scan driver
    |
    v
model_reduction / validation / outputs
```

### 5.2 Module Map

- `geometry_topology/`: tree graph, branch hierarchy, centerlines, observation and clamp points.
- `cross_section/`: non-circular section definition, tissue-region partition geometry, property integration.
- `materials/`: tissue material models and extensible constitutive interfaces.
- `branches/`: branch-level structural objects combining geometry, section, and material distribution.
- `joints_and_bifurcations/`: joint kinematics, stiffness, nonlinear restoring laws, local enriched models.
- `fruits/`: fruit mass and fruit-stem connection behavior.
- `excitation_and_bc/`: clamp, forcing, displacement/force/acceleration excitation.
- `discretization/`: orchard-specific low-cost discretization and hotspot enrichment.
- `solver_core/`: matrices, nonlinear forces, integrators, scans.
- `model_reduction/`: reserved MOR interfaces and error reporting.
- `validation/`: comparisons to references and regression checks.
- `apps/`: command line applications.

## 6. Data Flow

```text
model.json
  -> parse
  -> validate
  -> construct topology graph
  -> attach cross-section families and materials
  -> build branch and fruit objects
  -> define joints and boundary conditions
  -> discretize into reduced structural DOFs
  -> assemble M, C, K and nonlinear force providers
  -> run frequency or time analysis
  -> compute harvest-oriented response metrics
  -> export CSV / VTK / summary JSON
```

Key design rule: raw input data is kept separate from assembled computational state so we can support reproducibility, model mutation for scans, and future alternative backends.

### 6.1 Current Implementation Note

The current codebase now implements:

- linearized frequency-response analysis for fast scan workflows,
- Newmark time-history analysis,
- localized nonlinear spring-like connectors for joints and clamp regions,
- explicit fruit attachment DOFs,
- CSV export for both frequency and transient outputs.

## 7. Core Class Design

### 7.1 Input and Model Construction

- `ModelInput`
  - Plain data object representing parsed JSON/YAML.
  - Keeps raw branch, section, material, fruit, joint, and excitation definitions.

- `ModelValidator`
  - Checks completeness and consistency.
  - Verifies topology, section partition coverage, material references, and excitation targets.

- `TreeModelBuilder`
  - Converts validated input into runtime domain objects.
  - Owns construction order and dependency wiring.

### 7.2 Geometry and Topology

- `TreeTopology`
  - Stores nodes and parent-child branch relationships.
  - Provides traversal by hierarchy level and subtree queries.

- `TopologyNode`
  - Generic node in the orchard graph.
  - Can represent trunk base, branching junction, terminal twig node, clamp marker, or fruit attachment location.

- `BranchPath`
  - Defines branch centerline geometry.
  - Supports arc-length parameterization and local frame queries.

- `ObservationPoint`
  - Named point or DOF set for extracting results.

### 7.3 Cross-Section and Tissue Partitioning

- `CrossSectionProfile`
  - Abstract base for a branch cross-section at a given axial coordinate.
  - Supports area, centroid, inertia, and tissue-region queries.

- `ParameterizedSectionProfile`
  - Profile defined from analytic parameters such as ellipse, lobed contour, eccentric core, or offset rings.

- `ContourSectionProfile`
  - Profile defined from measured contour points.

- `MeasuredSectionSeries`
  - Branch-wise section definition along the axis.
  - Interpolates section properties between stations.

- `TissueRegion`
  - Geometry plus material-region tag: xylem, pith, phloem.

- `SectionIntegrator`
  - Numerically integrates region area, centroid, inertia, and equivalent stiffness factors.

### 7.4 Materials

- `MaterialBase`
  - Base constitutive interface.
  - Exposes density, tangent stiffness contribution, damping contribution, and optional nonlinear parameters.

- `ElasticMaterial`
  - Initial linear elastic model.

- `NonlinearElasticMaterial`
  - Polynomial or user-defined nonlinear constitutive extension.

- `DampingModel`
  - Base damping interface.

- `OrthotropicMaterialAdapter`
  - Future-ready wrapper for orthotropic behavior without forcing MVP complexity into the whole codebase.

- `SpatialMaterialField`
  - Returns position-dependent material properties along a branch.

### 7.5 Branches and Joints

- `BranchComponent`
  - Runtime representation of one branch.
  - Combines centerline, section evolution, tissue partitions, material field, discretization hints, and attached fruits.

- `BranchDiscretizationHint`
  - Encodes preferred refinement level and hotspot markers for a branch.

- `JointComponent`
  - Represents a bifurcation or branch connection.

- `JointConstitutiveLaw`
  - Interface for joint restoring moments/forces.

- `PolynomialRotationalJointLaw`
  - MVP nonlinear joint option using angle-dependent restoring terms.

- `GapFrictionJointLaw`
  - Reserved interface for future gap/friction/hysteresis behavior.

### 7.6 Fruits and Excitation

- `FruitAttachment`
  - Mass, position, attachment stiffness/damping, optional detachment criterion placeholder.

- `ExcitationDefinition`
  - Base excitation descriptor.

- `HarmonicExcitation`
  - Frequency, amplitude, phase, target location, and type.

- `ClampBoundaryCondition`
  - Clamp region definition and equivalent support properties.

### 7.7 Discretization and Solver

- `DOFManager`
  - Assigns DOFs consistently across branches, joints, fruits, and boundary conditions.

- `StructuralAssembler`
  - Builds mass, damping, stiffness, and nonlinear force providers from runtime components.

- `DynamicSystem`
  - Holds assembled operators and the state layout.

- `NonlinearForceProvider`
  - Interface for local nonlinear internal-force contributions.

- `TimeIntegrator`
  - Base interface for transient analysis.

- `NewmarkIntegrator`
  - First MVP transient integrator.

- `FrequencyResponseAnalyzer`
  - Computes linearized or weakly nonlinear frequency response.

- `ParameterScanRunner`
  - Runs batches over frequency, amplitude, clamp position, or selected material/joint parameters.

### 7.8 Reduction and Validation

- `ReducedBasis`
  - Placeholder basis container.

- `ReductionStrategy`
  - MOR interface for modal truncation, substructuring, or future hyper-reduction.

- `ErrorMetrics`
  - Defines modal error, displacement response error, peak error, and runtime gain.

- `ValidationRunner`
  - Compares fast solver outputs against references.

## 8. Numerical Strategy

### 8.1 Core Modeling Strategy

The initial solver uses a global-low-cost / local-enriched approach:

- Most slender branches are represented by specialized structural elements carrying effective section and tissue information.
- Hotspots such as joints, clamp zones, and selected fruit neighborhoods are enriched through local nonlinear components.
- Fruit bodies enter as concentrated masses with optional spring-damper attachment.

This gives a practical balance between fidelity and speed for orchard studies.

### 8.2 Assembly Strategy

The MVP assembly builds:

- global mass matrix `M`,
- global damping matrix `C`,
- global stiffness matrix `K`,
- external load vector `f(t)`,
- optional nonlinear internal force vector `g(u, v)`.

The first implementation should remain backend-agnostic:

- use lightweight dense/small sparse data structures for MVP,
- isolate matrix/vector interfaces,
- allow later swap to Eigen or PETSc without changing domain modules.

### 8.3 Analysis Modes

MVP supports:

- frequency sweep under harmonic excitation,
- transient response under harmonic or prescribed excitation,
- basic parameter scanning over excitation frequency and amplitude.

Future modes:

- nonlinear frequency continuation,
- reduced-order online/offline workflows,
- localized substructure enrichment with condensation.

## 9. MVP Scope

The MVP must demonstrate orchard-specific value without pretending to be fully general.

### 9.1 Included in MVP

- Tree topology with hierarchical branch graph.
- Non-circular section definitions using parameterized or contour-based input.
- Tissue partitions for xylem, pith, and phloem.
- Branch-wise section variation along axial stations.
- Simple branch structural discretization carrying section/material effects into equivalent operators.
- Fruit added mass and spring-damper attachment.
- Clamp boundary and harmonic excitation.
- Frequency response or transient response output.
- CLI demo that reads a model file and writes response results.
- Unit tests for geometry, materials, topology, and assembly.

### 9.2 Explicit MVP Placeholders

- Local joint nonlinearity may start with a low-order polynomial model.
- Orthotropy interface exists even if the first implementation uses isotropic parameters.
- Reduction module is interface-only at first, with one simple modal-truncation proof of concept added later.
- Validation against high-fidelity models begins with file-based reference data comparison, not fully automated external solvers.

## 10. Error Control Strategy

Every simplification should map to an error assessment plan.

### 10.1 Reference Levels

- `L0`: analytic or near-analytic toy cases.
- `L1`: higher-resolution structural reference model.
- `L2`: offline high-fidelity local or whole-tree model.
- `L3`: experimental measurement comparison when available.

### 10.2 Error Metrics

- Natural frequency relative error.
- Mode-shape correlation or modal assurance criterion.
- Peak frequency-response amplitude error.
- Time-history displacement/acceleration RMS error.
- Harvest-indicator error at target fruit nodes.
- Runtime speedup ratio.

### 10.3 Acceptance Philosophy

The project does not pursue maximum fidelity everywhere. It seeks acceptable error on harvest-oriented outputs under a documented speedup target.

## 11. First Development Roadmap

### Phase 0: Architecture Baseline

- Finalize design document and repository structure.
- Define input schema.
- Establish testing, formatting, and example conventions.

### Phase 1: Orchard-Specific MVP

- Implement topology, cross-section, materials, branch objects, excitation, and simple solver core.
- Add demo model and basic CLI.
- Add tests for geometry, parsing, topology, and matrix assembly.

### Phase 2: Targeted Nonlinearity and Better Validation

- Add nonlinear joint laws and localized clamp stiffness variation.
- Add fruit attachment dynamics and local nonlinear force providers.
- Improve reference comparison utilities and response metrics.

### Phase 3: Acceleration and Reduction

- Introduce modal reduction and substructuring.
- Add parameter sweep automation and error-vs-speed reporting.
- Prepare optional backend integration with Eigen/PETSc.

## 12. Planned Repository Structure

```text
.
|-- apps/
|-- CMakeLists.txt
|-- README.md
|-- config/
|-- docs/
|   `-- design.md
|-- examples/
|-- include/
|   `-- orchard_solver/
|       |-- geometry_topology/
|       |-- cross_section/
|       |-- materials/
|       |-- branches/
|       |-- joints_and_bifurcations/
|       |-- fruits/
|       |-- excitation_and_bc/
|       |-- discretization/
|       |-- solver_core/
|       |-- model_reduction/
|       `-- validation/
|-- scripts/
|-- src/
|   |-- geometry_topology/
|   |-- cross_section/
|   |-- materials/
|   |-- branches/
|   |-- joints_and_bifurcations/
|   |-- fruits/
|   |-- excitation_and_bc/
|   |-- discretization/
|   |-- solver_core/
|   |-- model_reduction/
|   `-- validation/
`-- tests/
```

## 13. Current Limitations of This Draft

- Numerical formulations are intentionally scoped for a fast orchard solver, not a full continuum FEM engine.
- Exact element equations for all branch behaviors are not frozen yet; the architecture is prepared for iteration.
- Material, anisotropy, and geometric nonlinearity interfaces are ahead of the MVP implementation on purpose.

This is expected: the design prioritizes correct problem framing and extensible interfaces before adding full complexity.
