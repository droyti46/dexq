"""Direct 512×512 SpineNet baseline: two extreme centers and a fixed 5° limit."""
from __future__ import annotations


def measure_axis(landmark_result: dict) -> dict:
    """Return the decoded native-coordinate angle without calibration or gates."""
    base = landmark_result.get('measurement')
    output = {
        'method': 'spinenet_direct_512_extreme_centers',
        'endpoint_policy': 'first_and_last_detected_candidate_by_native_y',
        'fixed_anatomical_level_pair': None,
        'threshold_deg': 5.0,
        'score': None,
        'score_threshold': 5.0,
        'score_semantics': 'absolute native-coordinate angle in degrees',
        'angle_deg': None,
        'signed_angle_deg': None,
        'violation': None,
        'top_xy': None,
        'bottom_xy': None,
        'status': 'unavailable',
        'reason': 'fewer_than_two_spinenet_centers',
        'model_training': False,
    }
    if base is not None:
        output.update(base)
        output.update(score=base['angle_deg'], status='research_prediction',
                      reason='fixed_5_degree_threshold_on_extreme_centers')
    return output
