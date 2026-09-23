"""B runtime candidate boundary from A review R1/R2; no mapping changes."""
from evaluations.iclr2027.native6_v3.object_mapping import (
    active_roots, canonical_task_object, normalize_coppelia_name,
)


def contact_candidate(mapping, task, variation, record):
    # Reject non-physical scene helpers BEFORE ancestry normalization.
    if record['object_type'].lower() != 'shape':
        return None
    return canonical_task_object(mapping, task_id=task, variation=variation,
                                 object_name=record['name'], ancestry_names=record['parent_chain'])


def attachment_candidate(mapping, task, variation, record):
    if record['object_type'].lower() != 'shape':
        return None
    name = normalize_coppelia_name(record['name'])
    matches = [r['canonical_id'] for r in active_roots(mapping, task, variation)
               if r['scene_root_name'] == name and 'attachment' in r['relation_sources']]
    if len(matches) > 1:
        raise ValueError('Ambiguous exact attachment root')
    return matches[0] if matches else None


def inventory_valid(mapping, probe):
    task, variation = probe['task'], probe['variation']
    included, excluded = probe['included_task_objects'], probe['excluded_task_objects']
    handles = [r['handle'] for r in included + excluded]
    if len(handles) != len(set(handles)):
        return False
    if any(r.get('runtime_candidate') is not True or
           contact_candidate(mapping, task, variation, r) != r.get('canonical_id') or
           r.get('canonical_id') is None for r in included):
        return False
    if any(r.get('runtime_candidate') is not False or
           contact_candidate(mapping, task, variation, r) is not None for r in excluded):
        return False
    return True
