"""Fast policy construction for pi0.5 / FlashVLA.

``PreTrainedPolicy.from_pretrained`` (LeRobot) first instantiates the policy from its
config and *then* loads the safetensors checkpoint.  For a 3B-parameter pi0.5 that
instantiation randomly initialises every weight on the default device, which on an
Ascend host costs 7-13 minutes of CPU time and a large RSS spike -- all of it thrown
away moments later by the checkpoint load.

``transformers`` ships a context manager that patches ``torch.nn.init.*`` functions (and
``PreTrainedModel.init_weights``) to no-ops for the duration of a ``with`` block:
``transformers.initialization.no_init_weights`` (transformers >= 5), with the older
``transformers.modeling_utils.no_init_weights`` as a fallback.  Every parameter is
overwritten by the subsequent ``load_state_dict``, so the weights are identical.

Usage::

    from flashvla.startup import no_random_init

    with no_random_init():
        policy = make_policy(cfg=cfg.policy, env_cfg=cfg.env, rename_map=cfg.rename_map)

Keep the context narrowly around the construction: any module created inside it that is
*not* restored from the checkpoint would keep uninitialised memory.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext

logger = logging.getLogger(__name__)


def no_random_init():
    """Context manager that turns random weight initialisation into a no-op.

    Falls back to a no-op context if transformers provides no such helper, so callers can
    use it unconditionally.
    """
    try:
        from transformers.initialization import no_init_weights
    except ImportError:
        try:
            from transformers.modeling_utils import no_init_weights
        except ImportError:
            logger.warning(
                "transformers provides no no_init_weights(); policy construction keeps "
                "its full random initialisation cost"
            )
            return nullcontext()
    return no_init_weights()
