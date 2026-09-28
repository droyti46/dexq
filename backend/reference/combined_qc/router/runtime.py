"""One numeric classifier and one frozen ONNX backbone; no training imports."""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from .features import ROOT, FeatureExtractor, prepare_native, sha256
from combined_qc.common import Progress


class RegionClassifier:
    def __init__(self, checkpoints_dir=None, *, verbose=False, ci=False):
        self.checkpoints_dir=Path(checkpoints_dir).resolve() if checkpoints_dir else ROOT/'checkpoints'
        directory=self.checkpoints_dir/'runtime';progress=Progress('router',verbose or ci)
        manifest=json.loads((directory/'manifest.json').read_text())
        if manifest.get('format')!='region_classifier_runtime_v1':raise ValueError('Unknown region runtime format')
        if set(manifest['files'])!={'region_logreg.npz','resnet18_features.onnx'}:raise ValueError('Incomplete region model inventory')
        for name,expected in manifest['files'].items():
            path=(directory/name).resolve()
            if not path.is_relative_to(directory.resolve()) or sha256(path)!=expected:
                raise ValueError(f'Region model checksum mismatch: {name}')
        with np.load(directory/'region_logreg.npz',allow_pickle=False) as data:
            self.mean=np.asarray(data['mean'],dtype=np.float64);self.scale=np.asarray(data['scale'],dtype=np.float64)
            self.coef=np.asarray(data['coef'],dtype=np.float64);self.intercept=np.asarray(data['intercept'],dtype=np.float64)
            classes=data['classes'];self.metadata=json.loads(str(data['metadata'].item()))
        meta=self.metadata
        if meta!=manifest['model'] or meta.get('format')!='image_only_region_logreg_v1':raise ValueError('Classifier metadata mismatch')
        if meta.get('class_names')!=['hip','lumbar_spine'] or classes.tolist()!=[0,1]:raise ValueError('Unexpected region classes')
        if meta.get('representation') not in {'global_512','global_spatial_2560'}:raise ValueError('Unexpected feature representation')
        n=512 if meta['representation']=='global_512' else 2560
        if (meta.get('n_features')!=n or any(a.shape!=(n,) for a in [self.mean,self.scale,self.coef])
                or self.intercept.shape!=(1,) or any(not np.isfinite(a).all() for a in [self.mean,self.scale,self.coef,self.intercept])
                or np.any(self.scale<=0)):raise ValueError('Invalid numeric classifier parameters')
        if meta.get('preprocessing')!='trim_exact_black_border_direct320_mirror_mean_v1':raise ValueError('Preprocessing mismatch')
        if meta.get('threshold')!=.5 or meta.get('review_threshold')!=.9:raise ValueError('Unexpected decision thresholds')
        if meta.get('backbone_sha256')!=manifest['files']['resnet18_features.onnx']:raise ValueError('Classifier/backbone binding mismatch')
        progress.update('load','region_logreg.npz verified',1,2)
        self.extractor=FeatureExtractor(directory/'resnet18_features.onnx',meta['backbone_sha256'])
        progress.update('load','Frozen ImageNet ONNX loaded',2,2)
        self.provenance={'classifier_sha256':manifest['files']['region_logreg.npz'],
            'backbone_sha256':manifest['files']['resnet18_features.onnx'],'n_fit_images':len(meta['train_image_ids']),
            'fitted_on_holdout':meta['trained_on_holdout'],'representation':meta['representation']}

    def score_features(self, features):
        features=np.asarray(features)
        if features.ndim!=2 or features.shape[1]!=2560 or len(features)==0 or not np.isfinite(features).all():
            raise ValueError('Expected finite [N,2560] features')
        x=features[:,:512] if self.metadata['representation']=='global_512' else features
        try:
            with np.errstate(over='raise',invalid='raise',divide='raise'):
                z=((x.astype(np.float64)-self.mean)/self.scale)@self.coef+self.intercept[0]
        except FloatingPointError as error:
            raise ValueError('Numeric classifier normalization overflow') from error
        if not np.isfinite(z).all():raise ValueError('Nonfinite class log odds')
        # Stable sigmoid without overflow; handles finite extreme log odds.
        p=np.empty_like(z);positive=z>=0
        p[positive]=1/(1+np.exp(-z[positive]));e=np.exp(z[~positive]);p[~positive]=e/(1+e)
        if not np.isfinite(p).all():raise ValueError('Invalid region probabilities')
        return p

    def predict_array(self,array, *, source=None):
        array=np.asarray(array)
        try: frame=prepare_native(array)
        except ValueError as error:
            return {'source':str(source) if source else None,'region':None,'label':None,'confidence':None,
                'scores':None,'needs_review':True,'metadata':{'status':'invalid_image','reason':str(error),'provenance':self.provenance}}
        p=float(self.score_features(self.extractor.extract(frame[None]))[0]);label=int(p>=.5)
        confidence=max(p,1-p)
        return {'source':str(source) if source else None,'region':'lumbar_spine' if label else 'hip','label':label,
            'confidence':confidence,'scores':{'hip':1-p,'lumbar_spine':p},'needs_review':confidence<.9,
            'metadata':{'status':'classified','input_width':int(array.shape[1]),'input_height':int(array.shape[0]),
                'decision_threshold':.5,'review_threshold':.9,'score_interpretation':'Uncalibrated classifier score; not an OOD detector',
                'preprocessing':self.metadata['preprocessing'],'provenance':self.provenance}}
