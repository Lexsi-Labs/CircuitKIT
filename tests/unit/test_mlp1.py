from circuitkit.api import discover_circuit

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))


def main():
    config = {
        "model": {"name": "gpt2", "precision": "bfloat16"},
        "pruning": {
            "target_sparsity": 0.4,
            "scope": "mlp",
            "random": True,
            "intervention": "patching",
        },
        "discovery": {
            "algorithm": "eap-ig",
            "task": "ioi",
            "level": "neuron",
            "batch_size": 2,
            "ig_steps": 2,
            "evaluate": True,
            "mlp_hook": "post_act",
            "data_params": {"num_examples": 200},
        },
        "output_path": "results/test_mlp1_gpt2.pt",
    }

    print("Running EAP-IG Discovery with mlp1/post_act...")
    result = discover_circuit(config)

    print("\n--- Discovery Result Summary ---")
    if "mlp" in result:
        import json

        with open("results/pruned_indices.json", "w") as f:
            # Convert keys to strings for JSON
            json.dump({str(k): [int(i) for i in v] for k, v in result["mlp"].items()}, f)
        print("Saved pruned indices to results/pruned_indices.json")

        mlp_dict = result["mlp"]
        # Find the maximum neuron index pruned across all layers
        all_mlp_neurons = []
        for layer, neurons in mlp_dict.items():
            all_mlp_neurons.extend(neurons)

        if all_mlp_neurons:
            max_idx = max(all_mlp_neurons)
            min_idx = min(all_mlp_neurons)
            print(f"Total MLP neurons pruned: {len(all_mlp_neurons)}")
            print(f"MLP Neuron Index Range: {min_idx} to {max_idx}")
            print("Expected bounds for GPT-2 d_mlp: 0 to 3071")

            if max_idx >= 768:
                print(
                    "SUCCESS: Pruned neurons above d_model (768), confirming we are targeting d_mlp (3072)."
                )
            else:
                print(
                    "WARNING: Max index is < 768. Might still be targeting d_model, or sparsity was too low."
                )
        else:
            print("No MLP neurons pruned.")
    else:
        print("'mlp' key not found in result.")


if __name__ == "__main__":
    main()
