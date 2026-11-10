# Provenance record (`lexsi_provenance.json`)

CircuitKIT writes a `lexsi.provenance/1` record into every checkpoint folder it exports
(`lexsi_provenance.json`) and embeds one under `provenance` in every `*_scores.json`.
The writer is `circuitkit.provenance.make_provenance` / `write_provenance`.

## `base_model` is the immediate parent

`base_model` names the model this step loaded: a Hub id or a local path. It is always
the `ref` of a `kind: "model"` entry in `inputs`. It is not the original foundation
model. To find that, follow the parent's own record, which is embedded under that
input's `provenance`, until it is `null`.

`base_model` is `null` only when `inputs` has no model entry.

Example: `gpt2` is pruned and exported to `out/pruned`, then that folder is loaded,
quantized and exported to `out/quantized`. The second record:

```json
{
  "schema": "lexsi.provenance/1",
  "library": "circuitkit",
  "base_model": "out/pruned",
  "base_model_relation": "quantized",
  "method": "export_checkpoint.quantization",
  "inputs": [
    {
      "kind": "model",
      "ref": "out/pruned",
      "provenance": {
        "schema": "lexsi.provenance/1",
        "library": "circuitkit",
        "base_model": "gpt2",
        "base_model_relation": "finetune",
        "method": "export_checkpoint.pruning",
        "inputs": [{"kind": "model", "ref": "gpt2", "provenance": null}],
        "params": {}
      }
    }
  ],
  "params": {}
}
```

(`version`, `git_sha` and `created_at` are left out here.)

## `base_model_relation`

Optional. One of `finetune`, `adapter`, `quantized`, `merge`. The key is left out when
the step did not produce a model or the relation is not known; it is never `null`.

| Written by | `method` | `base_model_relation` |
|---|---|---|
| `ck.export_checkpoint` / `save_pruned_checkpoint` (pruned or edited weights) | `export_checkpoint.pruning` | `finetune` (same architecture, changed weights) |
| `save_quantized_checkpoint` | `export_checkpoint.quantization` | `quantized` |
| `*_scores.json` from discovery | the algorithm | not written: discovery reads the model and produces no model |

## Limits

- `make_provenance` raises `ValueError` when `base_model` is passed and is not the `ref`
  of a model entry in `inputs`, or when the relation is not one of the four values.
- The record compares strings. It cannot tell whether the named model is the one that
  was really loaded; for a model built in memory the `ref` comes from its config.
- Records written before this rule can name the parent's parent in `base_model` when
  the loaded folder carried its own record. Their `inputs` are correct, so read the
  parent from the model entry in `inputs` for those files.
