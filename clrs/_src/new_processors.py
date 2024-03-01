from copy import deepcopy
from typing import Any, Callable, List, Optional

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

_Array = chex.Array
_Fn = Callable[..., Any]
BIG_NUMBER = 1e6


class MPNNLayer(hk.Module):
    def __init__(
        self,
        nb_layers: int,
        out_size: int,
        mid_size: Optional[int] = None,
        mid_act: Optional[_Fn] = None,
        reduction: _Fn = jnp.max,
        msgs_mlp_sizes: Optional[List[int]] = None,
        name: str = "mpnn_mp",
    ):
        super().__init__(name=name)
        self.nb_layers = nb_layers
        if mid_size is None:
            self.mid_size = out_size
        else:
            self.mid_size = mid_size
        self.out_size = out_size
        self.mid_act = mid_act
        self.reduction = reduction
        self._msgs_mlp_sizes = msgs_mlp_sizes

    def __call__(
        self,
        node_fts: _Array,
        edge_fts: _Array,
        graph_fts: _Array,
        adj_mat: _Array,
    ) -> _Array:
        node_tensors = node_fts
        edge_tensors = edge_fts
        graph_tensors = graph_fts

        m_1 = hk.Linear(self.mid_size)
        m_2 = hk.Linear(self.mid_size)
        # m_e = hk.Linear(self.mid_size)
        # m_g = hk.Linear(self.mid_size)

        o1 = hk.Linear(self.out_size)
        o2 = hk.Linear(self.out_size)

        msg_1 = m_1(node_tensors)
        msg_2 = m_2(node_tensors)
        # msg_e = m_e(edge_tensors)
        # msg_g = m_g(graph_tensors)

        msgs = (
            jnp.expand_dims(msg_1, axis=1)
            + jnp.expand_dims(msg_2, axis=2)
            # + msg_e
            # + jnp.expand_dims(msg_g, axis=(1, 2))
        )
        if self._msgs_mlp_sizes is not None:
            msgs = hk.nets.MLP(self._msgs_mlp_sizes)(jax.nn.relu(msgs))

        if self.mid_act is not None:
            msgs = self.mid_act(msgs)

        if self.reduction == jnp.mean:
            msgs = jnp.sum(msgs * jnp.expand_dims(adj_mat, -1), axis=1)
            msgs = msgs / jnp.sum(adj_mat, axis=-1, keepdims=True)
        elif self.reduction == jnp.max:
            maxarg = jnp.where(jnp.expand_dims(adj_mat, -1), msgs, -BIG_NUMBER)
            msgs = jnp.max(maxarg, axis=1)
        else:
            msgs = self.reduction(msgs * jnp.expand_dims(adj_mat, -1), axis=1)

        h_1 = o1(node_tensors)
        h_2 = o2(msgs)

        ret = h_1 + h_2

        return ret, edge_tensors


class AlignedMPNN(hk.Module):
    def __init__(
        self,
        nb_layers: int,
        out_size: int,
        mid_size: Optional[int] = None,
        mid_act: Optional[_Fn] = None,
        activation: Optional[_Fn] = jax.nn.relu,
        reduction: _Fn = jnp.max,
        msgs_mlp_sizes: Optional[List[int]] = None,
        name: str = "mpnn_mp",
        num_layers: int = 1,
    ):
        super().__init__(name=name)
        self.nb_layers = nb_layers
        if mid_size is None:
            self.mid_size = out_size
        else:
            self.mid_size = mid_size
        self.out_size = out_size
        self.mid_act = mid_act
        self.activation = activation
        self.reduction = reduction
        self._msgs_mlp_sizes = msgs_mlp_sizes
        self.num_layers = num_layers

    def __call__(
        self,
        node_fts: _Array,
        edge_fts: _Array,
        graph_fts: _Array,
        adj_mat: _Array,
        hidden: _Array,
        e_hidden: _Array,
        **kwargs,
    ) -> tuple[list[_Array], _Array]:
        node_tensors = jnp.concatenate([node_fts, hidden], axis=-1)
        edge_tensors = jnp.concatenate([edge_fts, e_hidden], axis=-1)
        graph_tensors = graph_fts

        # encode edge tensors
        edge_tensors = hk.Linear(self.mid_size)(edge_tensors)
        # VIRTUAL NODE

        # NODE FEATURES
        # add features of 0
        virtual_node_features = jnp.zeros(
            (node_tensors.shape[0], 1, node_tensors.shape[-1])
        )
        node_tensors = jnp.concatenate([node_tensors, virtual_node_features], axis=1)

        # EDGE FEATURES
        # add features of 0
        # column
        virtual_node_edge_features_col = jnp.zeros(
            (edge_tensors.shape[0], edge_tensors.shape[1], 1, edge_tensors.shape[-1])
        )
        edge_tensors = jnp.concatenate(
            [edge_tensors, virtual_node_edge_features_col], axis=2
        )

        # row
        virtual_node_edge_features_row = jnp.zeros(
            (
                edge_tensors.shape[0],
                1,
                edge_tensors.shape[2],
                edge_tensors.shape[-1],
            )
        )
        edge_tensors = jnp.concatenate(
            [edge_tensors, virtual_node_edge_features_row], axis=1
        )

        # ADJ MATRIX
        # add connection between VN and all other nodes
        virtual_node_adj_mat_row = jnp.ones((adj_mat.shape[0], 1, adj_mat.shape[-1]))
        adj_mat = jnp.concatenate([adj_mat, virtual_node_adj_mat_row], axis=1)
        virtual_node_adj_mat_col = jnp.ones((adj_mat.shape[0], adj_mat.shape[1], 1))
        adj_mat = jnp.concatenate([adj_mat, virtual_node_adj_mat_col], axis=2)

        layers = []
        for _ in range(self.num_layers):
            layers.append(
                MPNNLayer(
                    nb_layers=self.nb_layers,
                    out_size=self.out_size,
                    mid_size=self.mid_size,
                    reduction=self.reduction,
                )
            )

        for layer in layers:
            node_tensors, edge_tensors = layer(
                node_tensors, edge_tensors, graph_tensors, adj_mat
            )

        return (
            node_tensors[:, : node_tensors.shape[1] - 1, :],
            edge_tensors[
                :, : edge_tensors.shape[1] - 1, : edge_tensors.shape[2] - 1, :
            ],
            None,
        )

    @property
    def inf_bias(self):
        return False

    @property
    def inf_bias_edge(self):
        return False


class LinearAttentionLayer(hk.Module):
    def __init__(
        self, out_size, num_heads=1, mid_size=None, name="LinearAttentionLayer"
    ):
        super().__init__(name=name)
        self.out_size = out_size
        self.num_heads = num_heads
        self.mid_size = mid_size
        if mid_size is None:
            self.mid_size = out_size

    def __call__(
        self,
        node_fts: _Array,
        edge_fts: _Array,
        graph_fts: _Array,
        adj_mat: _Array,
    ):
        node_tensors = node_fts
        edge_tensors = edge_fts
        graph_tensors = graph_fts

        # Linear projections for Q, K, V
        Q = hk.Linear(self.out_size * self.num_heads)(node_tensors)
        K = hk.Linear(self.out_size * self.num_heads)(node_tensors)
        V = hk.Linear(self.out_size * self.num_heads)(node_tensors)

        # Reshape for multi-head attention
        B, N, _ = node_fts.shape
        Q = Q.reshape(B, N, self.num_heads, self.out_size)
        K = K.reshape(B, N, self.num_heads, self.out_size)
        V = V.reshape(B, N, self.num_heads, self.out_size)

        # Compute scaled dot-product attention
        attention_scores = jnp.einsum("bnhd,bmhd->bhnm", Q, K) / jnp.sqrt(self.out_size)
        attention_scores += adj_mat[:, None, :, :] * (
            -1e9
        )  # Masking with adjacency matrix
        attention_probs = jax.nn.softmax(attention_scores, axis=-1)
        attention_output = jnp.einsum("bhnm,bmhd->bnhd", attention_probs, V)

        # Combine heads
        attention_output = attention_output.reshape(B, N, -1)

        return attention_output, edge_tensors


save_name = {
    0: None,
    1: "input_node_features",
    2: "input_hidden_node_features",
    3: "input_edge_features",
    4: "input_hidden_edge_features",
    5: "input_graph_features",
    6: "input_hidden_graph_features",
    7: "input_adjacency_matrix",
    8: "out_node_features_0",
    9: "out_node_features_1",
    10: "out_node_features_2",
    11: "out_edge_features",
    12: "out_graph_features",
}

GLOBAL_SAMPLE_COUNTER = 0

from pathlib import Path
from jax import pure_callback


def save_input(save_features):
    global GLOBAL_SAMPLE_COUNTER
    save_directory_name = f"dataset/{save_name[0]}/{GLOBAL_SAMPLE_COUNTER}"
    Path(save_directory_name).mkdir(exist_ok=True, parents=True)

    for filename_idx, array in save_features:

        np.save(
            f"{save_directory_name}/{save_name[filename_idx.item()]}.npy",
            array,
        )
    if filename_idx == len(save_name) - 1:
        GLOBAL_SAMPLE_COUNTER = GLOBAL_SAMPLE_COUNTER + 1
    return np.array(0, dtype=np.int32)  # dummy variable


class LinearGraphTransformer(hk.Module):
    def __init__(
        self,
        nb_heads: int,
        out_size: int,
        save_emb_sub_dir: str,
        save_embeddings: str = False,
        name: str = "linear_gt",
        num_layers: int = 1,
        mid_size=None,
    ):
        super().__init__(name=name)
        self.out_size = out_size
        self.nb_heads = nb_heads
        self.num_layers = num_layers
        self.mid_size = mid_size
        self.save_emb_sub_dir = save_emb_sub_dir
        global save_name
        save_name[0] = self.save_emb_sub_dir
        self.save_embeddings = save_embeddings

    def __call__(
        self,
        node_fts: _Array,
        edge_fts: _Array,
        graph_fts: _Array,
        adj_mat: _Array,
        hidden: _Array,
        e_hidden: _Array,
        **kwargs,
    ):

        if (
            not (type(adj_mat) == jax._src.interpreters.partial_eval.DynamicJaxprTracer)
            and self.save_embeddings
        ):
            result = pure_callback(
                save_input,
                jax.ShapeDtypeStruct(shape=(), dtype=np.int32),
                [
                    (1, np.array(deepcopy(node_fts))),
                    (2, np.array(deepcopy(hidden))),
                    (3, np.array(deepcopy(edge_fts))),
                    (4, np.array(deepcopy(e_hidden))),
                    (5, np.array(deepcopy(graph_fts))),
                    (7, np.array(deepcopy(adj_mat))),
                ],
            )
        node_tensors = node_fts
        node_tensors = jnp.concatenate([node_tensors, hidden], axis=-1)
        edge_tensors = jnp.concatenate([edge_fts, e_hidden], axis=-1)

        # encode edge_tensors
        edge_tensors = hk.Linear(self.mid_size)(edge_tensors)

        graph_tensors = graph_fts

        layers = []
        for _ in range(self.num_layers):
            layers.append(
                LinearAttentionLayer(
                    out_size=self.out_size,
                    num_heads=self.nb_heads,
                    mid_size=self.mid_size,
                )
            )

        for i, layer in enumerate(layers):
            node_tensors, edge_tensors = layer(
                node_tensors, edge_tensors, graph_tensors, adj_mat
            )
            if (
                not (
                    type(adj_mat)
                    == jax._src.interpreters.partial_eval.DynamicJaxprTracer
                )
                and self.save_embeddings
            ):
                result = pure_callback(
                    save_input,
                    jax.ShapeDtypeStruct(shape=(), dtype=np.int32),
                    [
                        (i + 8, np.array(deepcopy(node_tensors))),
                    ],
                )

        if (
            not (type(adj_mat) == jax._src.interpreters.partial_eval.DynamicJaxprTracer)
            and self.save_embeddings
        ):

            result = pure_callback(
                save_input,
                jax.ShapeDtypeStruct(shape=(), dtype=np.int32),
                [
                    (11, np.array(deepcopy(edge_tensors))),
                    (12, np.array(deepcopy(graph_tensors))),
                ],
            )

        return node_tensors, edge_tensors, None

    @property
    def inf_bias(self):
        return False

    @property
    def inf_bias_edge(self):
        return False
