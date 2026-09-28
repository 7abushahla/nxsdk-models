# Public NxTF compilation without NxSDK

The public NxTF partition optimizer and synapse encoder can run without an NxSDK installation. This fork defers the hardware mapper imports until `compileModel()` or `map()` is called. Python logging supplies the logger when NxSDK is absent. The partition search, layer compilation, resource constraints, and synapse compression algorithms remain the upstream implementations.

The exporter produces a portable `nxtf-partitions-v1` artifact from an actual `NxModel.partition()` call. It does not assign physical chip/core addresses or call the restricted `N2ASynapseCompiler`. The artifact records those boundaries in its provenance. Compilation and model simulation do not establish hardware equivalence or hardware performance.

## Environment

Use Python 3.11 and the separate environment below.

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements-offline.txt
.venv/bin/python -m unittest discover -s tests_offline -v
```

NumPy 1.23.5 is intentional. The upstream convolution compiler uses ragged-array behavior removed in NumPy 1.24. Its `VisibleDeprecationWarning` may still appear. The compiler algorithms have not been rewritten to suppress that warning. The original NxSDK hardware test suite still requires NxSDK and its original environment.

## Examples

```bash
.venv/bin/python -m nxsdk_modules_ncl.dnn.tutorials.offline_compile \
  --topology dense --reset-mode soft --output offline-output/dense-soft.json
.venv/bin/python -m nxsdk_modules_ncl.dnn.tutorials.offline_compile \
  --topology dense --reset-mode hard --output offline-output/dense-hard.json
.venv/bin/python -m nxsdk_modules_ncl.dnn.tutorials.offline_compile \
  --topology conv --reset-mode soft --output offline-output/conv-soft.json
```

The convolution example includes a virtual flatten layer before its dense layer. The examples use integer weights, threshold mantissa 255, current decay 4096, voltage decay zero, and refractory delay zero. The soft reset recurrent weight is exactly representable for this threshold. Other parameter choices require their own functional validation.

## Python API

```python
import json
from nxsdk_modules_ncl.dnn.src.offline import compile_model

manifest = compile_model(model)  # model is an already constructed NxModel
with open('partitions.json', 'w') as stream:
    json.dump(manifest, stream, indent=2)
```

Weights and biases must be finite and integer valued. The exporter rejects fractional values before invoking the compiler. The model's configured logging directory receives the upstream partition files. A fresh partition run is performed on each call. The exporter reads the selected in-memory partition chain, so virtual layers and cross-partition references retain the compiler's connectivity.

## Artifact fields

| Field | Contents |
| --- | --- |
| `schema` | `nxtf-partitions-v1` |
| `provenance` | Public compiler identity, source file hashes, false flags for SDK mapping, physical placement, synapse memory compilation, and hardware validation |
| `layers` | Logical output shape, input mode, reset mode, and the upstream core assignment and multiplicity arrays |
| `partitions` | Stable `layer:partition` identifiers, parameters, compartments, complete encoded synapse groups and formats, input/output axon groups, and resource counts |
| `edges` | Decoded compartment connectivity obtained by joining actual output axon groups to their destination input axon groups |

Each compartment has a local `id`, an upstream `global_id`, a `logical_id`, a `role`, and bias mantissa/exponent. The role is `hard`, `dendrite`, `soma`, or `dummy`. Soft reset pairs retain both compartments and their recurrent feedback connection. Compartment ordering may be interleaved. Consumers must use the explicit maps when associating compartments with logical neurons.

The `compartments` list covers every slot in the update groups selected by the public mapper's documented rounding rule. Slots without a logical neuron have `logical_id=null` and role `dummy`. An upstream padding ID is preserved when present; additional holes and update padding have `global_id=null`. `configured` distinguishes configured register indices from trailing default slots. The original compartment arrays remain under `compartment_group`.

`resource_counts.compiled_compartments` counts the original compartment group. `configured_compartments` counts register indices through its highest local index. `update_slots` and `num_update_groups` include rounding to groups of four. Upstream resource counters such as `numSynMemWords` are public compiler estimates. They are not measurements of restricted NxSDK binary memory encoding.

Raw synapse formats, entries, and axon fields retain their upstream names. Runtime object hashes are replaced by stable references. No physical placement is inferred from `chipCounter`, which is the partitioner's planning counter. A simulator or external mapper must supply its own explicit placement.

Each decoded edge carries source and destination partition/local compartment identifiers, weight mantissa and exponent, weight precision, sign mode, delay, and a soft reset flag. Raw compressed entries are retained alongside this view. The exporter performs no simulator-specific arithmetic conversion, routing, scheduling, or neuron-model substitution.
