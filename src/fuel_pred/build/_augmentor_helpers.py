"""Shared helpers for ``census_augment.Pipeline.augment`` callers.

Used by both ``enrich_census`` (cross-sectional, per-station) and
``enrich_panel_temporal`` (temporal, per-(station, date)). Both call
sites hit the same upstream bug shape:

    UPSTREAM_PRESET_COLLISION: when ANY direct ``<NAMESPACE>.<field>``
    ref appears in the source-field list of a requested ``PRESET.<id>``,
    a single ``Pipeline.augment(...)`` call fails inside the PRESET
    evaluator with::

        ValueError: cannot reindex on an axis with duplicate labels

    The split-then-merge workaround works in both modes — sending the
    colliding direct refs in a separate augment call and merging the
    output column-wise.

This module owns the splitter so the two call sites stay in lockstep
when upstream lands the proper fix.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def split_for_preset_collision(variables: dict[str, str]) -> list[dict[str, str]]:
    """Group ``variables`` into PRESET-collision-safe subsets.

    Returns a list of dicts; pass each to a separate ``Pipeline.augment``
    call, then merge the resulting frames column-wise. The common no-
    collision case returns ``[variables]`` (one group, one call) so
    callers don't pay any overhead.

    Detection: a non-PRESET ref (any ``<NS>.<field>``) collides when it
    appears in ``features.get(preset_id).source_fields()`` of any
    requested ``PRESET.<id>``. Colliding direct refs get split into a
    second group so each ``Pipeline.augment`` sees a non-colliding set.

    Args:
        variables: friendly_name → ``<NS>.<field>`` dict (e.g. the
            ``AUGMENTOR_VARIABLES_*`` config dicts).

    Returns:
        ``[variables]`` if there's no collision (the fast path), or
        ``[non_colliding_variables, colliding_direct_refs]`` otherwise.
    """
    try:
        from census_augment.features import features
    except ImportError:  # augmentor not installed (e.g. some test envs)
        return [variables]

    # Direct refs = anything that isn't a PRESET. v2.0+ cross-dataset
    # PRESETs use DSS sources, so the splitter is namespace-agnostic:
    # any direct ref is a collision candidate.
    direct_refs: dict[str, str] = {}  # friendly -> "<NS>.<field>"
    preset_ids: list[str] = []
    for friendly, ref in variables.items():
        if ref.startswith("PRESET."):
            preset_ids.append(ref[len("PRESET.") :])
        elif "." in ref:
            direct_refs[friendly] = ref

    if not direct_refs or not preset_ids:
        return [variables]

    colliding_friendlies: set[str] = set()
    for preset_id in preset_ids:
        try:
            sources = set(features.get(preset_id).source_fields())
        except KeyError:
            continue
        for friendly, ref in direct_refs.items():
            if ref in sources:
                colliding_friendlies.add(friendly)

    if not colliding_friendlies:
        return [variables]

    pass_a = {f: r for f, r in variables.items() if f not in colliding_friendlies}
    pass_b = {f: r for f, r in variables.items() if f in colliding_friendlies}
    logger.info(
        "augmentor split: %d non-colliding vars + %d colliding direct vars (%s) "
        "to work around upstream PRESET-collision bug",
        len(pass_a),
        len(pass_b),
        sorted(colliding_friendlies),
    )
    return [pass_a, pass_b]


def merge_augmented_frames(
    frames: list,
    *,
    primary_key_cols: tuple[str, ...] = (),
) -> "object":
    """Column-wise merge a list of augmented DataFrames.

    The first frame is the canonical scaffold (provides ``sa2_code``,
    ``sa2_name``, any other bookkeeping columns); subsequent frames
    contribute additional ``sa2_*`` value columns by positional
    alignment. Bookkeeping cols listed in ``primary_key_cols`` are
    NEVER overwritten — used to protect e.g. ``sa2_code``, ``sa2_name``,
    ``*_release`` etc. when merging temporal-mode output.

    Args:
        frames: list of DataFrames returned by separate
            ``Pipeline.augment`` calls on the same input rows. Must have
            the same row count (positional merge).
        primary_key_cols: column names from the first frame that must
            not be overwritten by subsequent frames. ``sa2_code`` and
            ``sa2_name`` are always protected; pass extras for temporal
            mode (e.g. ``("seifa_release", "erp_by_sa2_release")``).

    Returns:
        The merged DataFrame. When ``len(frames) == 1``, returns that
        frame unchanged.
    """
    if len(frames) == 1:
        return frames[0]
    protected = {"sa2_code", "sa2_name"} | set(primary_key_cols)
    merged = frames[0].copy()
    for extra in frames[1:]:
        for col in extra.columns:
            if col.startswith("sa2_") and col not in protected:
                merged[col] = extra[col].values
    return merged
