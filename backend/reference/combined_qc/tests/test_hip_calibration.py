"""Study exclusion must extend through learned side preprocessing."""
import tempfile
from pathlib import Path
import unittest

import numpy as np

from combined_qc.hip.features import (
    _positioning_fold_features, assign_study_folds, crossfit_training_probabilities,
)


class HipCalibrationTests(unittest.TestCase):
    def context(self):
        groups = np.repeat([f'study{i}' for i in range(5)], 2)
        folds = np.repeat(np.arange(5), 2)
        # Two side-only images: one shares a quality study; the other is an
        # additional global study assigned to fold0. Both must be excluded.
        side_groups = np.concatenate([groups, ['study0', 'extra_study']])
        side_folds = np.r_[folds, 0, 0]
        target = np.r_[np.tile([0, 1], 5), 0, 1]
        side = np.zeros((len(target), 2048), dtype=np.float64)
        side[:, 0] = target * 2 - 1
        normal = np.full((len(groups), 6400), 10., dtype=np.float64)
        mirrored = np.full_like(normal, 20.)
        context = {'normal': normal, 'mirrored': mirrored, 'side_features': side,
                   'side_targets': target, 'side_groups': side_groups,
                   'side_folds': side_folds, 'side_ids': np.array([f'image{i}' for i in range(12)]),
                   'quality_indices': np.arange(10),
                   'quality_ids': np.array([f'image{i}' for i in range(10)]),
                   'side_cache_sha256': 'a' * 64, 'side_backbone_sha256': 'b' * 64}
        return context, folds, groups

    def test_held_side_labels_cannot_change_canonical_features(self):
        context, _, groups = self.context()
        with tempfile.TemporaryDirectory() as directory:
            first, proof = _positioning_fold_features(context, 0, groups, model_dir=Path(directory))
            changed = dict(context)
            changed['side_targets'] = context['side_targets'].copy()
            held = context['side_folds'] == 0
            changed['side_targets'][held] = 1 - changed['side_targets'][held]
            second, second_proof = _positioning_fold_features(changed, 0, groups)
            np.testing.assert_array_equal(first, second)
            np.testing.assert_array_equal(first[:, 0], np.tile([10., 20.], 5))
            self.assertEqual(proof['fit_n'], 8)
            self.assertEqual(proof['held_n'], 4)
            self.assertEqual(set(proof['held_studies']), {'study0', 'extra_study'})
            self.assertFalse(set(proof['fit_studies']) & set(proof['held_studies']))
            self.assertNotIn('image10', proof['fit_image_ids'])
            self.assertNotIn('image11', proof['fit_image_ids'])
            self.assertLessEqual(proof['numeric_probability_error'], 1e-12)
            self.assertEqual(proof['fit_image_ids'], second_proof['fit_image_ids'])
            with np.load(Path(directory) / 'fold0.npz', allow_pickle=False) as saved:
                self.assertEqual(saved['classes'].tolist(), [0, 1])

    def test_positioning_rejects_final_side_only_calibration(self):
        context, folds, groups = self.context()
        y = np.tile([0, 1], 5)
        with self.assertRaisesRegex(ValueError, 'fold-excluded automatic side'):
            crossfit_training_probabilities('positioning_rotation', context['normal'], y, folds, groups)

    def test_quality_fit_and_prediction_use_the_fold_side_features(self):
        context, folds, groups = self.context()
        y = np.tile([0, 1], 5)
        # Deliberately remove all signal from the final-side input. Only the
        # temporary side-dependent normal/mirrored views contain the label.
        final_features = np.zeros_like(context['normal'])
        probabilities, proofs = crossfit_training_probabilities(
            'positioning_rotation', final_features, y, folds, groups,
            positioning_context=context)
        np.testing.assert_array_equal(probabilities >= .5, y)
        self.assertEqual(len(proofs), 5)
        for proof in proofs:
            self.assertFalse(set(proof['validation_studies']) &
                             set(proof['automatic_side']['fit_studies']))

    def test_side_only_images_follow_global_studies(self):
        rows = [{'image_id': f'image{i}', 'study_id': f'study{i}'} for i in range(5)]
        rows.extend([{'image_id': 'side_only', 'study_id': 'study0'},
                     {'image_id': 'extra_hip', 'study_id': 'new_shared_study'},
                     {'image_id': 'extra_spine', 'study_id': 'new_shared_study'}])
        split = {'folds': {str(i): {'val': [f'image{i}']} for i in range(5)}}
        assigned = assign_study_folds(rows, split)
        self.assertEqual(assigned[5], assigned[0])
        self.assertEqual(assigned[6], assigned[7])
        self.assertEqual(assigned[6], 0)


if __name__ == '__main__':
    unittest.main()
