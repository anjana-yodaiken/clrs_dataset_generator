import itertools
from constants import MODEL_PATH, MPNN_MODELS_ROOT_DIR

from pathlib import Path
from main import main_validate_model
import jax.numpy as jnp

if __name__ == "__main__":
    add_virtual_nodes = [True, False]
    layer_norms = [True, False]
    mid_dims = [192, 256]
    reductions = ["max", "sum"]
    disable_edge_updates_ = [True, False]
    apply_attentions = [False]
    number_of_attention_heads_ = [3]

    mpnn_variants = itertools.product(
        add_virtual_nodes,
        layer_norms,
        mid_dims,
        reductions,
        disable_edge_updates_,
        apply_attentions,
        number_of_attention_heads_,
    )

    for mpnn_variant in mpnn_variants:
        add_virtual_node = mpnn_variant[0]
        layer_norm = mpnn_variant[1]
        mid_dim = mpnn_variant[2]
        reduction = mpnn_variant[3]
        disable_edge_updates = mpnn_variant[4]
        apply_attention = mpnn_variant[5]
        number_of_attention_heads = mpnn_variant[6]

        if reduction == "max":
            reduction_fn = jnp.max
        elif reduction == "sum":
            reduction_fn = jnp.sum
        elif reduction == "mean":
            reduction_fn = jnp.mean
        else:
            raise Exception("Unknown reduction.")

        if apply_attention:
            model_path = Path(
                MPNN_MODELS_ROOT_DIR,
                f"vn-{add_virtual_node}-ln-{layer_norm}-mid_dim-{mid_dim}-reduction-{reduction}-disable_edge_updates-{disable_edge_updates}-apply_attention-{apply_attention}-number_of_attention_heads-{number_of_attention_heads}.pkl",
            )
        else:
            model_path = Path(
                MPNN_MODELS_ROOT_DIR,
                f"vn-{add_virtual_node}-ln-{layer_norm}-mid_dim-{mid_dim}-reduction-{reduction}-disable_edge_updates-{disable_edge_updates}-apply_attention-{apply_attention}.pkl",
            )
        main_validate_model(
            model_path=model_path,
            encoder_decoder_path=MODEL_PATH,
            add_virtual_node=add_virtual_node,
            layer_norm=layer_norm,
            mid_dim=mid_dim,
            reduction=reduction_fn,
            disable_edge_updates=disable_edge_updates,
            apply_attention=apply_attention,
            number_of_attention_heads=number_of_attention_heads,
            processor_type="aligned_mpnn",
            batch_size=64,
            eval_batch_size=6,
        )
