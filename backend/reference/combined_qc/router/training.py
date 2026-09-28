"""Development-only model selection, followed by one sealed study holdout."""
from __future__ import annotations
import json
from pathlib import Path
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from combined_qc.common import Progress, quiet_output, write_json
from .bootstrap import prepare_backbone
from .features import DATA, ROOT, FeatureExtractor, actual_source, load_record, prepare_native, records, sha256, recipe_fingerprint, RESNET_STATE_SHA

SEED=20260927
C_VALUES=(.1,1.,10.)
REPRESENTATIONS=('global_512','global_spatial_2560')


def subset(features, representation):
    return (features[:,:512] if representation=='global_512' else features).astype(np.float64)


def fit_model(x,y,c):
    model=make_pipeline(StandardScaler(),LogisticRegression(C=c,class_weight='balanced',solver='liblinear',max_iter=2000,tol=1e-6,random_state=SEED))
    model.fit(x,y)
    return model


def metrics(y,p):
    y=np.asarray(y);p=np.asarray(p,dtype=np.float64);pred=(p>=.5).astype(int)
    if not np.isfinite(p).all() or np.any((p<0)|(p>1)):raise ValueError('Invalid class probabilities')
    precision,recall,f1,support=precision_recall_fscore_support(y,pred,labels=[0,1],zero_division=0)
    confidence=np.maximum(p,1-p);keep=confidence>=.9
    return {'n_images':len(y),'accuracy':float(accuracy_score(y,pred)),
        'macro_f1':float(f1_score(y,pred,average='macro')),
        'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
        'roc_auc_spine':float(roc_auc_score(y,p)),
        'confusion_matrix_hip_spine':confusion_matrix(y,pred,labels=[0,1]).tolist(),
        'per_class':{name:{'precision':float(precision[i]),'recall':float(recall[i]),'f1':float(f1[i]),'n':int(support[i])}
                     for i,name in enumerate(['hip','lumbar_spine'])},
        'minimum_true_class_score':float(np.where(y==1,p,1-p).min()),
        'review_threshold':.9,'review_count':int((~keep).sum()),
        'accepted_coverage':float(keep.mean()),
        'accepted_accuracy':float(accuracy_score(y[keep],pred[keep])) if keep.any() else None}


def choose_c(x,y,groups,representation):
    candidates=[]
    folds=list(StratifiedGroupKFold(3,shuffle=True,random_state=SEED+1).split(x,y,groups))
    for c in C_VALUES:
        probabilities=np.full(len(y),np.nan)
        for train,val in folds:
            assert not set(groups[train])&set(groups[val])
            model=fit_model(subset(x[train],representation),y[train],c)
            probabilities[val]=model.predict_proba(subset(x[val],representation))[:,1]
        score=metrics(y,probabilities)
        candidates.append({'C':c,'metrics':score})
    # Predeclared tie break: macro F1, balanced accuracy, AUC, stronger shrinkage.
    best=max(candidates,key=lambda r:(r['metrics']['macro_f1'],r['metrics']['balanced_accuracy'],r['metrics']['roc_auc_spine'],-r['C']))
    return best['C'],candidates


def _cache_features(rows,data,source_data,backbone,cache,progress):
    fingerprint={'manifest_sha256':sha256(Path(data)/'manifest.jsonl'),'backbone_sha256':sha256(backbone),
        'preprocessing':'trim_exact_black_border_direct320_mirror_mean_v1',
        'feature_code_sha256':sha256(ROOT/'features.py'),
        'source_files_sha256':{r['image_id']:sha256(actual_source(r,data,source_data)) for r in rows}}
    meta=cache.with_suffix('.json')
    if cache.is_file() and meta.is_file() and json.loads(meta.read_text())==fingerprint:
        with np.load(cache,allow_pickle=False) as saved:
            features=saved['features'];ids=saved['image_ids'].tolist()
        if ids==[r['image_id'] for r in rows] and features.shape==(255,2560) and np.isfinite(features).all():
            progress.update('features','Using verified image-only cache',255,255)
            return features
    extractor=FeatureExtractor(backbone)
    parts=[]
    for start in range(0,len(rows),16):
        frames=np.stack([prepare_native(load_record(r,data,source_data)) for r in rows[start:start+16]])
        parts.append(extractor.extract(frames))
        progress.update('features','ImageNet features, normal + mirror',min(start+16,len(rows)),len(rows))
    features=np.concatenate(parts)
    cache.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(cache,features=features,image_ids=np.asarray([r['image_id'] for r in rows]))
    write_json(meta,fingerprint)
    return features


def train(data=None, *, checkpoints_dir=None, device='auto', epochs=None,
          verbose=False, ci=False, resume=True, convert=True):
    if device not in {'auto','cpu'}:raise ValueError('Frozen router training uses CPU')
    if epochs is not None and (type(epochs) is not int or epochs!=1):raise ValueError('One logistic fit; epochs=None or 1')
    with quiet_output(verbose or ci):
        base=Path(checkpoints_dir).resolve() if checkpoints_dir else ROOT/'checkpoints'
        report_path=base/'training_report.json'
        if resume and report_path.is_file():
            from .conversion import load_source_classifier, convert_all
            _,spec,_=load_source_classifier(base)
            if spec.get('training_recipe_sha256')!=recipe_fingerprint():
                raise ValueError('Router training recipe changed; explicitly use resume=False')
            if spec['manifest_sha256']!=sha256((Path(data) if data else DATA)/'manifest.jsonl') or spec['split_sha256']!=sha256(ROOT/'split.json'):
                raise ValueError('Router data or sealed split changed; start a new fit')
            data_root=Path(data).resolve() if data else DATA
            cache_metadata_path=base/'source/features.json'
            if (not cache_metadata_path.is_file()
                    or sha256(cache_metadata_path)!=spec.get('feature_cache_metadata_sha256')):
                raise ValueError('Source feature metadata changed; use resume=False')
            cache_metadata=json.loads(cache_metadata_path.read_text())
            from .features import PROJECT
            input_hashes={row['image_id']:sha256(actual_source(row,data_root,PROJECT/'source_data'))
                          for row in records(data_root)}
            if input_hashes!=cache_metadata.get('source_files_sha256'):
                raise ValueError('Native training input changed; use resume=False')
            if (cache_metadata.get('feature_code_sha256')!=sha256(ROOT/'features.py')
                    or cache_metadata.get('manifest_sha256')!=spec['manifest_sha256']
                    or cache_metadata.get('backbone_sha256')!=spec['backbone_sha256']
                    or sha256(base/'source/features_backbone.onnx')!=spec['backbone_sha256']
                    or sha256(base/'source/features.npz')!=spec.get('feature_cache_sha256')):
                raise ValueError('Source features or backbone changed; use resume=False')
            report=json.loads(report_path.read_text())
            Progress('router',verbose or ci).update('train','Reusing verified single fitted source classifier')
            if convert:
                report['conversion']=convert_all(base,data=data,verbose=verbose,ci=ci)
            report['converted']=bool(convert)
            write_json(report_path,report)
            return report
        return _train(data,checkpoints_dir=base,verbose=verbose,ci=ci,convert=convert)


def _train(data=None, *, checkpoints_dir=None, verbose=False, ci=False, convert=True):
    started=time.perf_counter()
    data=Path(data).resolve() if data else DATA
    base=Path(checkpoints_dir).resolve() if checkpoints_dir else ROOT/'checkpoints'
    output=base/'outputs';output.mkdir(parents=True,exist_ok=True)
    progress=Progress('router',verbose or ci)
    split_path=ROOT/'split.json';split=json.loads(split_path.read_text())
    if sha256(data/'manifest.jsonl')!=split['manifest_sha256']:raise ValueError('Data changed after the holdout was frozen')
    rows=records(data);ids=[r['image_id'] for r in rows]
    if set(split['development_image_ids'])&set(split['holdout_image_ids']) or set(ids)!=set(split['development_image_ids'])|set(split['holdout_image_ids']):
        raise ValueError('Invalid frozen split membership')
    y=np.asarray([int(r['anatomical_region']=='lumbar_spine') for r in rows]);groups=np.asarray([r['study_id'] for r in rows])
    dev=np.asarray([i for i,v in enumerate(ids) if v in set(split['development_image_ids'])])
    held=np.asarray([i for i,v in enumerate(ids) if v in set(split['holdout_image_ids'])])
    if set(groups[dev])&set(groups[held]):raise ValueError('Holdout study leakage')
    backbone=prepare_backbone(base,verbose=verbose,ci=ci)
    from .features import PROJECT
    features=_cache_features(rows,data,PROJECT/'source_data',backbone,base/'source/features.npz',progress)
    comparisons=[]
    outer=list(StratifiedGroupKFold(5,shuffle=True,random_state=SEED).split(features[dev],y[dev],groups[dev]))
    for representation in REPRESENTATIONS:
        probabilities=np.full(len(dev),np.nan);fold_reports=[]
        for fold,(train,val) in enumerate(outer):
            x_train=features[dev[train]];y_train=y[dev[train]];group_train=groups[dev[train]]
            assert not set(group_train)&set(groups[dev[val]])
            c,inner=choose_c(x_train,y_train,group_train,representation)
            model=fit_model(subset(x_train,representation),y_train,c)
            probabilities[val]=model.predict_proba(subset(features[dev[val]],representation))[:,1]
            fold_reports.append({'fold':fold,'C':c,'n_train':len(train),'n_validation':len(val),
                'train_studies':sorted(set(group_train)),'validation_studies':sorted(set(groups[dev[val]])),
                'metrics':metrics(y[dev[val]],probabilities[val]),'inner_selection':inner})
            progress.update('development',f'{representation}: fold {fold+1}',fold+1,5)
        comparisons.append({'representation':representation,'metrics':metrics(y[dev],probabilities),
            'folds':fold_reports,'oof_probabilities':probabilities.tolist(),
            'error_ids':[ids[dev[i]] for i in np.flatnonzero((probabilities>=.5).astype(int)!=y[dev])]})
    selected=max(comparisons,key=lambda r:(r['metrics']['macro_f1'],r['metrics']['balanced_accuracy'],r['metrics']['roc_auc_spine'],r['representation']=='global_512'))
    representation=selected['representation']
    final_c,final_selection=choose_c(features[dev],y[dev],groups[dev],representation)
    selection={'representation':representation,'C':final_c,'selected_without_holdout':True,
        'split_sha256':sha256(split_path),'development_comparison':comparisons,'final_inner_cv':final_selection}
    # Persist the choice before any holdout probability is computed.
    write_json(output/'selection_before_holdout.json',selection)
    progress.update('train',f'One final logistic model: {representation}, C={final_c}, {len(dev)} images')
    model=fit_model(subset(features[dev],representation),y[dev],final_c)
    scaler=model.named_steps['standardscaler'];lr=model.named_steps['logisticregression']
    source=base/'source';source.mkdir(parents=True,exist_ok=True)
    from .bootstrap import export_recipe_sha256
    spec={'format':'image_only_region_logreg_v1','representation':representation,'n_features':int(lr.coef_.shape[1]),
        'class_names':['hip','lumbar_spine'],'threshold':.5,'review_threshold':.9,
        'preprocessing':'trim_exact_black_border_direct320_mirror_mean_v1',
        'backbone_sha256':sha256(backbone),'ImageNet_state_sha256':RESNET_STATE_SHA,
        'backbone_export_recipe_sha256':export_recipe_sha256(),
        'feature_cache_metadata_sha256':sha256(base/'source/features.json'),
        'feature_cache_sha256':sha256(base/'source/features.npz'),
        'training_recipe_sha256':recipe_fingerprint(),
        'train_image_ids':[ids[i] for i in dev],'holdout_image_ids':[ids[i] for i in held],
        'manifest_sha256':split['manifest_sha256'],'split_sha256':sha256(split_path),
        'C':final_c,'seed':SEED,'trained_on_holdout':False}
    target=source/'region_logreg.npz'
    import tempfile
    with tempfile.TemporaryDirectory(dir=source,prefix='.fit-') as temporary:
        staged=Path(temporary)/target.name
        np.savez_compressed(staged,mean=scaler.mean_.astype(np.float64),scale=scaler.scale_.astype(np.float64),coef=lr.coef_[0].astype(np.float64),intercept=lr.intercept_.astype(np.float64),classes=lr.classes_,metadata=np.asarray(json.dumps(spec)))
        staged.replace(target)
    from .bootstrap import SOURCE_NAME
    write_json(source/'classifier_manifest.json',{'format':'router_source_v1','files':{
        SOURCE_NAME:sha256(source/SOURCE_NAME),'region_logreg.npz':sha256(target)},'model':spec})
    # Evaluate the exact safe numeric source representation without touching
    # runtime/. Optional conversion is a separate, transactional operation.
    from .conversion import load_source_classifier, score_numeric
    values,restored_spec,_=load_source_classifier(base)
    all_p=score_numeric(values,restored_spec,features)
    reference=model.predict_proba(subset(features,representation))[:,1]
    parity=float(np.max(np.abs(all_p-reference)))
    if parity>1e-10:raise AssertionError('Numeric model differs from trained logistic regression')
    holdout=metrics(y[held],all_p[held]);holdout['n_studies']=len(set(groups[held]))
    study_correct=[bool(np.all((all_p[held][groups[held]==g]>=.5).astype(int)==y[held][groups[held]==g])) for g in sorted(set(groups[held]))]
    holdout['studies_all_images_correct']=sum(study_correct)
    from scipy.stats import beta
    image_successes=int(round(holdout['accuracy']*len(held)))
    study_successes=sum(study_correct)
    holdout['exact_95pct_lower_bound_images_ignoring_clustering']=float(beta.ppf(.025,image_successes,len(held)-image_successes+1)) if image_successes else 0.0
    holdout['exact_95pct_lower_bound_all_correct_studies']=float(beta.ppf(.025,study_successes,len(study_correct)-study_successes+1)) if study_successes else 0.0
    report={'protocol':split['protocol'],'grouping':'study_id; patient identity unavailable after anonymization',
        'n_images':len(rows),'development_images':len(dev),'development_studies':len(set(groups[dev])),
        'holdout_images':len(held),'holdout_studies':len(set(groups[held])),
        'selected':{k:selection[k] for k in ['representation','C','selected_without_holdout']},
        'development':selected['metrics'],'development_candidates':comparisons,'holdout':holdout,
        'training_recipe_sha256':recipe_fingerprint(),'numeric_conversion_max_error':parity,'holdout_errors':[ids[i] for i in held if int(all_p[i]>=.5)!=y[i]],
        'prohibited_features':['filename','path','DICOM fields','native dimensions/aspect ratio','laterality','quality labels'],
        'fit_image_ids':spec['train_image_ids'],'held_out_image_ids':spec['holdout_image_ids'],
        'checkpoint_dir':str(base),'seconds':time.perf_counter()-started,'converted':False,
        'limitations':['Same acquisition source; no independent institution validation','Study-disjoint only; patient independence cannot be established','Only two known anatomy classes; confidence is not an OOD detector','Perfect finite holdout accuracy is not a guarantee on future images']}
    write_json(output/'metrics.json',report)
    write_json(output/'holdout_predictions.json',[{'image_id':ids[i],'study_id':groups[i],'true_region':'lumbar_spine' if y[i] else 'hip','predicted_region':'lumbar_spine' if all_p[i]>=.5 else 'hip','p_lumbar_spine':float(all_p[i]),'confidence':float(max(all_p[i],1-all_p[i]))} for i in held])
    write_json(base/'training_report.json',report)
    progress.update('holdout',f"Accuracy={holdout['accuracy']:.6f}, F1={holdout['macro_f1']:.6f}",len(held),len(held))
    if convert:
        from .conversion import convert_all
        report['conversion']=convert_all(base,data=data,verbose=verbose,ci=ci)
        report['converted']=True
        write_json(base/'training_report.json',report)
        write_json(output/'metrics.json',report)
    return report
