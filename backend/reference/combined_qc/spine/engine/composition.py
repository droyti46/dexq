"""Three independent lumbar quality outputs with explicit anatomical unknowns.

This composition layer does not fit models, read target labels, infer vertebral
levels from candidate order, or gate one criterion using another criterion.
"""
from __future__ import annotations

import math

def _score_output(scores: dict, key: str, method: str) -> dict:
    score = scores.get(key)
    if score is None:
        return {'method': method, 'score': None, 'violation': None,
                'threshold': 0.5, 'status': 'unavailable'}
    if isinstance(score, bool):
        raise ValueError(f'Expected a finite score, not bool: {key}')
    score = float(score)
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError(f'Invalid classifier score: {key}')
    return {'method': method, 'score': score, 'violation': score >= 0.5,
            'threshold': 0.5, 'status': 'research_prediction',
            'score_semantics': 'uncalibrated criterion violation score'}


def _landmarks(result: dict) -> dict:
    width, height = result['image_width'], result['image_height']
    if type(width) is not int or type(height) is not int or min(width, height) <= 0:
        raise ValueError('Positive integer native image dimensions required')
    points = []
    for candidate in result.get('candidates', []):
        xy = candidate['xy']
        if len(xy) != 2:
            raise ValueError('Expected a native (x,y) center')
        x, y = map(float, xy)
        confidence = float(candidate['confidence'])
        if not all(map(math.isfinite, (x, y, confidence))):
            raise ValueError('Nonfinite center/confidence')
        if not (0 <= x < width and 0 <= y < height and 0 <= confidence <= 1):
            raise ValueError('Center outside native frame or invalid confidence')
        corners = candidate.get('corners_xy')
        if corners is not None:
            if len(corners) != 4 or any(len(p) != 2 for p in corners):
                raise ValueError('Expected four corner coordinates')
            corners = [[float(v) for v in p] for p in corners]
            if not all(math.isfinite(v) for p in corners for v in p):
                raise ValueError('Nonfinite corner coordinates')
        points.append({'center_xy': [x, y], 'detector_score': confidence,
                       'corners_xy': corners, 'anatomical_level': None,
                       'visibility': 'unknown', 'annotation_status': 'model_prediction_unvalidated'})
    points.sort(key=lambda p: (p['center_xy'][1], p['center_xy'][0]))
    for index, point in enumerate(points):
        point['candidate_id'] = f'candidate_{index:02d}'
    return {'image_width': width, 'image_height': height,
            'coordinate_system': 'native_pixels_x_right_y_down',
            'vertebral_candidates': points, 'anatomical_levels_available': False,
            'iliac_crests': {'image_left': None, 'image_right': None},
            'source_preprocessing': result.get('preprocessing', {}).get('method'),
            'anatomical_localization_validated': False}


def compose_spine_result(image_id: str, primary_landmarks: dict,
                         coverage_result: dict, axis_result: dict, criterion_scores: dict) -> dict:
    """Compose manual coverage, anonymous centers and artifact classifier.

    The axis uses only the direct-512 detector and its native-coordinate angle.
    """
    if not isinstance(image_id, str) or not image_id:
        raise ValueError('Nonempty image_id required')
    shared = _landmarks(primary_landmarks)
    axis = dict(axis_result)
    if axis.get('violation') is not None and type(axis['violation']) is not bool:
        raise ValueError('Axis result decision must be bool or None')
    coverage = dict(coverage_result)
    if type(coverage.get('violation')) is not bool or not math.isfinite(float(coverage.get('score'))):
        raise ValueError('Manual coverage result must contain a finite score and Boolean decision')
    artifacts = _score_output(criterion_scores, 'p_spine_artifacts', 'resnet18_morphology_logistic_regression')
    artifacts['localization'] = None
    artifacts['localization_status'] = 'no_object_or_mask_supervision'
    flags = ['anatomical_landmarks_not_validated', 'coverage_anatomical_level_unavailable']
    if axis['violation'] is None:
        flags.append('axis_measurement_unavailable')
    if coverage['violation'] is True:
        flags.append('coverage_manual_rule_positive_review_endpoint_visibility')
    if artifacts['violation'] is True:
        flags.append('artifact_classifier_positive_review_landmark_interference')
    return {
        'schema_version': 'spine_modules_v3', 'image_id': image_id,
        'anatomical_region': 'lumbar_spine', 'region_source': 'caller_supplied',
        'shared_landmarks': shared, 'coverage': coverage, 'axis': axis, 'artifacts': artifacts,
        'review_flags': flags,
        'quality': combine_quality(coverage, axis, artifacts),
    }


def combine_quality(*criteria: dict) -> dict:
    """Boolean OR: any violation is True; any remaining unknown is None."""
    decisions = [item.get('violation') for item in criteria]
    if not decisions or any(v is not None and type(v) is not bool for v in decisions):
        raise ValueError('One or more bool/None criterion decisions required')
    value = True if any(v is True for v in decisions) else (None if any(v is None for v in decisions) else False)
    return {'violation': value, 'method': 'boolean_or_of_three_independent_criteria',
            'status': 'unavailable' if value is None else 'research_prediction'}
