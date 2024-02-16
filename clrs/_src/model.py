# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Model base classes and utilities."""

import abc
from typing import Dict, List, Optional, Tuple, Union
import chex

from clrs._src import probing
from clrs._src import samplers
from clrs._src import specs
import numpy as np


_Array = chex.Array
Result = Dict[str, probing.DataPoint]


class Model(abc.ABC):
  """Abstract base class for CLRS3-B models."""

  def __init__(self, spec: Union[specs.Spec, List[specs.Spec]]):
    """Set up the problem, prepare to predict on first task."""
    if not isinstance(spec, list):
      spec = [spec]
    self._spec = spec

  @abc.abstractmethod
  def predict(self, features: samplers.Features) -> Result:
    """Make predictions about the current task."""
    pass

  @abc.abstractmethod
  def feedback(self, feedback: Optional[samplers.Feedback]):
    """Advance to the next task, incorporating any available feedback."""
    pass


def evaluate_hints(
    hints: Tuple[probing.DataPoint],
    lengths: _Array,
    hint_preds: List[Result],
) -> Dict[str, _Array]:
  """Evaluate hint predictions."""
  evals = {}
  for truth in hints:
    assert truth.name in hint_preds[0]
    eval_along_time = [_evaluate(truth, p[truth.name], hints,
                                 idx=i+1, lengths=lengths)
                       for (i, p) in enumerate(hint_preds)]
    evals[truth.name] = np.sum(
        [x * np.sum(i+1 < lengths)
         for i, x in enumerate(eval_along_time)]) / np.sum(lengths - 1)
    evals[truth.name + '_along_time'] = np.array(eval_along_time)

  # Unlike outputs, the hints sometimes include scalars, which don't have
  # a meaningful eval score. So we don't compute a global 'hint score' as we
  # do for outputs.
  return evals


def evaluate(
    outputs: Tuple[probing.DataPoint],
    predictions: Result,
) -> Dict[str, float]:
  """Evaluate output predictions."""
  evals = {}
  for truth in outputs:
    assert truth.name in predictions
    pred = predictions[truth.name]
    evals[truth.name] = _evaluate(truth, pred, outputs)
  # Return a single scalar score that is the mean of all output scores.
  evals['score'] = sum([v.item() for v in evals.values()]) / len(evals)
  return evals


def _evaluate(truth, pred, full_truth, idx=None, lengths=None):
  """Evaluate single prediction of hint or output."""
  assert pred.name == truth.name
  assert pred.location == truth.location
  assert pred.type_ == truth.type_
  mask_name = f'{truth.name}_mask'
  if mask_name in full_truth:
    assert False
    mask = full_truth[mask_name].data
    return np.mean((pred.data[mask].flatten() - truth.data[mask].flatten())**2)
  else:
    if truth.type_ not in _EVAL_FN:
      raise ValueError('Invalid type')
    truth_data = truth.data
    pred_data = pred.data
    if idx is not None:
      if np.all(idx >= lengths):
        return 0.
      truth_data = truth_data[idx][idx < lengths]
      pred_data = pred_data[idx < lengths]
    return _EVAL_FN[truth.type_](pred_data, truth_data)


def _eval_one(pred, truth):
  mask = np.all(truth != specs.OutputClass.MASKED, axis=-1)
  return np.sum(
      (np.argmax(pred, -1) == np.argmax(truth, -1)) * mask) / np.sum(mask)


def _mask_fn(pred, truth):
  """Evaluate outputs of type MASK, and account for any class imbalance."""
  mask = (truth != specs.OutputClass.MASKED).astype(np.float32)

  # Use F1 score for the masked outputs to address any imbalance
  tp = np.sum((((pred > 0.5) * (truth > 0.5)) * 1.0) * mask)
  fp = np.sum((((pred > 0.5) * (truth < 0.5)) * 1.0) * mask)
  fn = np.sum((((pred < 0.5) * (truth > 0.5)) * 1.0) * mask)

  # Protect against division by zero
  if tp + fp > 0:
    precision = tp / (tp + fp)
  else:
    precision = np.float32(1.0)
  if tp + fn > 0:
    recall = tp / (tp + fn)
  else:
    recall = np.float32(1.0)

  if precision + recall > 0.0:
    f_1 = 2.0 * precision * recall / (precision + recall)
  else:
    f_1 = np.float32(0.0)

  return f_1

_EVAL_FN = {
    specs.Type.SCALAR:
        lambda pred, truth: np.mean((pred - truth)**2),
    specs.Type.MASK: _mask_fn,
    specs.Type.MASK_ONE:
        _eval_one,
    specs.Type.CATEGORICAL:
        _eval_one,
    specs.Type.POINTER:
        lambda pred, truth: np.mean((pred == truth) * 1.0)
}
