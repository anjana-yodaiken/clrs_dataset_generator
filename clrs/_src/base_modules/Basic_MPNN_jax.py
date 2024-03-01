from typing import Any, Callable, List, Optional

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

_Array = chex.Array
_Fn = Callable[..., Any]
BIG_NUMBER = 1e6


class Basic_MPNN(hk.Module):
    def __init__(
        self,
        nb_layers: int,
        out_size: int,
        mid_size: Optional[int] = None,
        mid_act: Optional[_Fn] = None,
        activation: Optional[_Fn] = jax.nn.relu,
        reduction: _Fn = jnp.max,
        msgs_mlp_sizes: Optional[List[int]] = None,
        use_ln: bool = False,
        name: str = "mpnn_mp",
        **kwargs,
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
        self.use_ln = use_ln

    def __call__(
        self,
        node_fts: _Array,
        edge_fts: _Array,
        graph_fts: _Array,
        adj_mat: _Array,
        hidden: _Array,
        **kwargs,
    ) -> _Array:

        node_tensors = node_fts
        edge_tensors = edge_fts
        graph_tensors = graph_fts

        m_1 = hk.Linear(self.mid_size)
        m_2 = hk.Linear(self.mid_size)
        m_e = hk.Linear(self.mid_size)
        m_g = hk.Linear(self.mid_size)

        o1 = hk.Linear(self.out_size)
        o2 = hk.Linear(self.out_size)

        msg_1 = m_1(node_tensors)
        msg_2 = m_2(node_tensors)
        msg_e = m_e(edge_tensors)
        msg_g = m_g(graph_tensors)

        msgs = (
            jnp.expand_dims(msg_1, axis=1)
            + jnp.expand_dims(msg_2, axis=2)
            + msg_e
            + jnp.expand_dims(msg_g, axis=(1, 2))
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

        if self.activation is not None:
            ret = self.activation(ret)

        if self.use_ln:
            ln = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
            ret = ln(ret)

        return ret, None, None

    @property
    def inf_bias(self):
        return False

    @property
    def inf_bias_edge(self):
        return False


class MPNNLayer(hk.Module):
    def __init__(
        self,
        nb_layers: int,
        out_size: int,
        mid_size: Optional[int] = None,
        mid_act: Optional[_Fn] = None,
        activation: Optional[_Fn] = jax.nn.relu,
        reduction: _Fn = jnp.max,
        msgs_mlp_sizes: Optional[List[int]] = None,
        use_ln: bool = False,
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
        self.activation = activation
        self.reduction = reduction
        self._msgs_mlp_sizes = msgs_mlp_sizes
        self.use_ln = use_ln

    def __call__(
        self,
        node_fts: _Array,
        edge_fts: _Array,
        graph_fts: _Array,
        adj_mat: _Array,
        hidden: _Array,
    ) -> _Array:
        node_tensors = node_fts
        edge_tensors = edge_fts
        graph_tensors = graph_fts

        m_1 = hk.Linear(self.mid_size)
        m_2 = hk.Linear(self.mid_size)
        m_e = hk.Linear(self.mid_size)
        m_g = hk.Linear(self.mid_size)

        o1 = hk.Linear(self.out_size)
        o2 = hk.Linear(self.out_size)

        msg_1 = m_1(node_tensors)
        msg_2 = m_2(node_tensors)
        msg_e = m_e(edge_tensors)
        msg_g = m_g(graph_tensors)

        msgs = (
            jnp.expand_dims(msg_1, axis=1)
            + jnp.expand_dims(msg_2, axis=2)
            + msg_e
            + jnp.expand_dims(msg_g, axis=(1, 2))
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

        if self.activation is not None:
            ret = self.activation(ret)

        if self.use_ln:
            ln = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
            ret = ln(ret)

        return ret, msg_e


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
        use_ln: bool = False,
        name: str = "mpnn_mp",
        num_layers: int = 3,
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
        self.use_ln = use_ln

    def __call__(
        self,
        node_fts: _Array,
        edge_fts: _Array,
        graph_fts: _Array,
        adj_mat: _Array,
        hidden: _Array,
        e_hidden: _Array,
        num_layers: int = 3,
        **kwargs,
    ) -> tuple[list[_Array], _Array]:
        node_tensors = jnp.concatenate([node_fts, hidden], axis=-1)
        edge_tensors = jnp.concatenate([edge_fts, e_hidden], axis=-1)
        graph_tensors = graph_fts

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
        for _ in range(num_layers):
            layers.append(
                MPNNLayer(
                    nb_layers=self.nb_layers,
                    out_size=self.out_size,
                    mid_size=self.mid_size,
                    activation=self.activation,
                    reduction=self.reduction,
                )
            )

        for layer in layers:
            node_tensors, edge_tensors = layer(
                node_tensors, edge_tensors, graph_tensors, adj_mat, hidden
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
