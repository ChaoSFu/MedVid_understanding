# ReliVE v2 protocol-first contract

This document defines a model-independent protocol. It applies before any
segmenter, tracker, VLM, verifier, cache, or certificate run. A protocol record
contains hashes and abstract roles; it does not contain a video-specific rule,
anchor selection rule, or action vocabulary.

## Claim and evidence types

| Claim type | Required generic evidence roles | Minimum evidence form |
|---|---|---|
| `ENTITY_STATE` | `SUBJECT`, `STATE_INDICATOR` | typed entity mask or region in a frozen temporal window |
| `SPATIAL_RELATION` | `SUBJECT`, `REFERENCE_ENTITY`, `RELATION_INTERFACE` | two regions plus their interface or an explicit occlusion abstention |
| `CONTACT_ACTION` | `ACTOR`, `ACTION_TARGET`, `CONTACT_INTERFACE` | actor/target masks and a time-bound contact interface |
| `STATE_CHANGE_OR_PERSISTENCE` | `SUBJECT`, `PRE_STATE`, `POST_STATE`, `CHANGE_INTERFACE` | pre/post evidence tubes and a change interface |
| `POSTCONDITION_PERSISTENCE` | `SUBJECT`, `POST_STATE`, `PERSISTENCE_INTERFACE` | a post-interaction tube with explicit interaction-end localization |

Every `EvidenceNode` binds a frame hash, role, optional mask hash, and one
occlusion state: `VISIBLE`, `PARTIALLY_OCCLUDED`, `OCCLUDED`, or `AMBIGUOUS`.
`EvidenceTube` binds node identities to a closed temporal window. A
`RelationEdge` makes subject/reference/interface relationships explicit; no
relation is inferred merely because boxes overlap.

## Interventions and controls

An `InterventionPlan` declares a typed family and a fixed target-node set.
Supported families are defined by a versioned operator, for example
single-region occlusion, mask-union occlusion, tube occlusion, or temporal
window replacement. The operator must be able to represent the frozen geometry
without substituting a wider rectangle.

Matched controls use ordered tiers:

1. **Tier 1**: same binary-mask geometry, area, frame count and non-overlap.
2. **Tier 2**: same tube geometry and temporal extent with predeclared spatial
   displacement.
3. **Tier 3**: a separately frozen diagnostic control, excluded from certificate
   admission.

If no declared tier is constructible, the result is
`OPERATOR_UNAVAILABLE` and the evidence result is an abstention. It is neither
semantic contradiction nor missing evidence.

## Adaptation and certificates

`AdaptationDecision` contains the route, maximum round count, and a cycle key.
Routes include `TEMPORAL_REACQUIRE`, `SPATIAL_RECOMPOSE`,
`RELATIONAL_COMPOSITE`, and `STOP`. A cycle key may be consumed only once; an
exhausted maximum budget or repeated key ends in abstention.

A `VERIFIED` certificate requires all predeclared semantic intervention
conditions, passing pixel and provenance audits, a valid matched control, and a
complete immutable binding to the EvidenceProgram, model, prompt, operator,
policy, and inputs. `UNCERTAIN` means the conditions were not all established.
`ENGINEERING_FAILURE` means an implementation, pixel, binding, cache, or
operator failure. `NOT_APPLICABLE` carries no certificate authority.

Formal admission never treats human review acceptance, a legal mask, or a legal
box as semantic support. An unavailable operator cannot be converted into a
negative semantic label.

## Dataset isolation

The three immutable manifests are `protocol_dev_manifest.jsonl`,
`calibration_manifest.jsonl`, and `blind_test_manifest.jsonl`. They split by
full `source_video_sha256`; frame hashes must also be unique across splits.
Protocol development may use fixtures only. Calibration defines a frozen
operator or policy without benchmark outcomes. Blind test remains unseen until
all protocol, calibration, and implementation versions are frozen.

`automatic` mode can use only registered automatic components. `human_oracle`
mode is a separately labelled protocol mode for controlled analysis; it cannot
silently supply masks, tubes, timings, or evidence to automatic mode, and it
cannot create a benchmark certificate.

## Machine-readable contracts

`relive.v2.protocol` contains immutable dataclasses for `EvidenceProgram`,
`EvidenceNode`, `EvidenceTube`, `RelationEdge`, `InterventionPlan`,
`AdaptationDecision`, `Certificate`, and `ProtocolManifest`. All schemas use
canonical JSON hashes. The three split manifests are checked with
`validate_relive_v2_protocol_manifests.py` before use.
