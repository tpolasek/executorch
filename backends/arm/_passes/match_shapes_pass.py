# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2024 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast

from executorch.backends.arm._passes.arm_pass_utils import (
    create_node,
    get_first_fake_tensor,
)

from executorch.exir.dialects._ops import ops as exir_ops

from executorch.exir.pass_base import ExportPass, PassResult
from torch.fx import GraphModule, Node


class MatchShapesPass(ExportPass):
    """
    For ops in 'targeted_ops', make sure that the inputs share the same rank.
    In particular, unsqueeze inputs of rank 1.
    """

    def __init__(self, exported_program):
        super().__init__()
        self.exported_program = exported_program

    targeted_ops = [
        exir_ops.edge.aten.add.Tensor,
        exir_ops.edge.aten.sub.Tensor,
        exir_ops.edge.aten.mul.Tensor,
        exir_ops.edge.aten.div.Tensor,
    ]

    def _match_op_shape(self, graph_module, node, arg, rank, max_rank, shape):
        """
        In graph_module, insert a view between arg and node to make the
        shape of arg match the other args to node.
        """
        with graph_module.graph.inserting_before(node):
            new_shape = list([1] * (max_rank - rank) + list(shape))
            view = create_node(
                graph_module.graph,
                exir_ops.edge.aten.view_copy.default,
                args=(arg, new_shape),
                kwargs={},
            )
            node.replace_input_with(arg, view)

    def _try_match_buffer_shape(self, arg, max_rank):
        """
        Change arg's fake tensor meta to match max_rank if:
            - size of fake tensor is 1.
            - arg is found in inputs_to_buffers or inputs_to_parameters.
        """
        fake_tensor = get_first_fake_tensor(arg)
        if fake_tensor.numel() != 1:
            return
        buffer_name = None
        if arg.name in self.exported_program.graph_signature.inputs_to_buffers:
            buffer_name = self.exported_program.graph_signature.inputs_to_buffers[
                arg.name
            ]
        elif arg.name in self.exported_program.graph_signature.inputs_to_parameters:
            buffer_name = self.exported_program.graph_signature.inputs_to_parameters[
                arg.name
            ]
        if buffer_name:
            new_tensor = self.exported_program.state_dict[buffer_name].reshape(
                [1] * max_rank
            )
            self.exported_program.state_dict[buffer_name] = new_tensor
            arg.meta["val"] = fake_tensor.fake_mode.from_tensor(
                new_tensor, static_shapes=True
            )

    def call(self, graph_module: GraphModule) -> GraphModule:
        for node in graph_module.graph.nodes:
            node = cast(Node, node)

            if node.op != "call_function" or node.target not in self.targeted_ops:
                continue

            # Calculate max rank of all inputs to node
            max_rank = 1
            for arg in node.args:
                if isinstance(arg, Node):
                    shape = get_first_fake_tensor(arg).shape
                    max_rank = max(max_rank, len(shape))

            # Adjust output shape of args if needed.
            for arg in node.args:
                if not isinstance(arg, Node):
                    continue
                shape = get_first_fake_tensor(arg).shape
                rank = len(shape)
                if rank == max_rank:
                    continue

                # If the argument is call_function, match shape by inserting view node.
                if arg.op == "call_function":
                    self._match_op_shape(graph_module, node, arg, rank, max_rank, shape)
                else:
                    # If the argument is a buffer or parameter, adjust shape by changing the fake tensor meta.
                    # Only do this for size == 1.
                    self._try_match_buffer_shape(arg, max_rank)

        graph_module.recompile()
        graph_module = super().call(graph_module).graph_module
        return PassResult(graph_module, True)

    def ensures(self, graph_module):
        for node in graph_module.graph.nodes:
            if node.op == "call_function" or node.target not in self.targeted_ops:
                continue
            arg0_rank = node.args[0].meta["val"].dim()
            arg1_rank = node.args[1].meta["val"].dim()
            if arg0_rank != arg1_rank:
                raise ValueError("Arithmethic operators need to have the same rank!")
