# Day 3 Hook Plan

## Expected Hook Point
Future visual-token methods should attach at:

`image -> vision encoder -> projector -> visual_tokens -> EfficientInferenceMethod.process_visual_tokens() -> policy/action head`

The benchmark framework owns the method wrapper and metadata logging. GR00T model internals should remain unchanged until a later, explicit integration step.

## Expected Tensor Shape
Visual tokens are expected to use shape `[B, N, D]`:

- `B`: batch size
- `N`: visual token count
- `D`: token embedding dimension

## Metadata To Log
- `original_tokens`
- `kept_tokens`
- `keep_ratio`
- `pruning_method`

## Safety Rules
- Baseline must preserve exact behavior.
- Pruning must be disabled by default.
- No shape mutation is allowed unless `method != baseline`.
- Method wrappers must be dependency-safe and work in dry-run/mock mode without LIBERO.
- Real VLA-Pruner, SpecPrune-VLA, ADP, and CLP are out of scope for Day 3.

