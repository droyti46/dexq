import assert from 'node:assert/strict';
import test from 'node:test';
import { axesEqual, resultWithSavedAxis, resultWithDraftAxis, reportTable, validateSubmission } from './manualAxis.ts';
import { reportCsv } from './projects.ts';
import { excelReportData } from './report.ts';
import type { Axis, WorkspaceItem } from './projects.ts';
import type { AnalysisResult, CheckResult } from './types.ts';

const axis: Axis = { top: [50, 10], bottom: [50, 90] };
const check = (id: string, violation: boolean): CheckResult => ({
  check_id: id, title: id, status: violation ? 'failed' : 'passed', violation,
  summary: 'Автоматическая оценка', confidence: 0.9, details: {}, method: 'model', model_status: 'ready',
});
const result: AnalysisResult = {
  analysis_id: 'test', filename: 'image.dcm', study_uid: null, image_uid: null,
  anatomical_region: 'lumbar_spine', region_source: 'model', projection: 'unknown', projection_source: 'not_determined',
  needs_review: false, quality_class: 1, violation_types: ['spine_axis'], processing_status: 'Success',
  time_of_processing: 1, checks: [check('spine_axis', true), check('spine_coverage', false)],
  preview_data_url: '', annotated_data_url: null, error: null,
  geometry: { image_width: 100, image_height: 100, coordinate_system: 'native_pixels_x_right_y_down',
    axis_line: { top_xy: [30, 10], bottom_xy: [50, 90], angle_deg: 14 }, vertebral_candidates: [], gap_lines: [], foreground_bbox: null },
};
const item: WorkspaceItem = { id: 'a', filename: result.filename, size: 1, position: 1, status: 'ready', progress: 100, result };

test('axis comparison detects draft changes, reset and return to saved coordinates', () => {
  assert.equal(axesEqual(null, null), true);
  assert.equal(axesEqual(axis, { top: [...axis.top], bottom: [...axis.bottom] }), true);
  assert.equal(axesEqual(axis, null), false);
  assert.equal(axesEqual(axis, { ...axis, top: [51, 10] }), false);
});

test('only saved axes replace the spine verdict and preserve the automatic result', () => {
  assert.equal(resultWithSavedAxis(item), result);
  const saved: WorkspaceItem = { ...item, savedAxis: axis };
  const corrected = resultWithSavedAxis(saved)!;
  assert.equal(corrected.quality_class, 0);
  assert.deepEqual(corrected.violation_types, []);
  assert.equal(corrected.checks[0].violation, false);
  assert.equal(corrected.checks[0].confidence, null);
  assert.equal(corrected.needs_review, true);
  assert.match(corrected.checks[0].summary, /Эвристика/);
  assert.equal(corrected.geometry?.axis_line?.angle_deg, 0);
  assert.deepEqual(corrected.geometry?.axis_line?.top_xy, axis.top);
  assert.equal(result.quality_class, 1);
  assert.equal(result.checks[0].violation, true);
  assert.equal(result.geometry?.axis_line?.angle_deg, 14);
});

test('saved axis preserves other violations and can introduce a new spine violation', () => {
  const saved: WorkspaceItem = { ...item, savedAxis: axis, result: { ...result, checks: [check('spine_axis', true), check('spine_coverage', true)], violation_types: ['spine_axis', 'spine_coverage'] } };
  assert.deepEqual(resultWithSavedAxis(saved)?.violation_types, ['spine_coverage']);
  assert.equal(resultWithSavedAxis(saved)?.quality_class, 1);
  const tilted: WorkspaceItem = { ...item, savedAxis: { top: [20, 10], bottom: [50, 90] },
    result: { ...result, quality_class: 0, checks: [check('spine_axis', false), check('spine_coverage', false)], violation_types: [] } };
  assert.equal(resultWithSavedAxis(tilted)?.checks[0].status, 'failed');
  assert.equal(resultWithSavedAxis(tilted)?.quality_class, 1);
  assert.deepEqual(resultWithSavedAxis(tilted)?.violation_types, ['spine_axis']);
});

test('manual axes do not turn failed or unsupported studies into successful analyses', () => {
  const failed: WorkspaceItem = { ...item, status: 'error', savedAxis: axis, result: { ...result, quality_class: null, processing_status: 'Failure' } };
  assert.equal(resultWithSavedAxis(failed), failed.result);
  const hip: WorkspaceItem = { ...item, savedAxis: axis, result: { ...result, anatomical_region: 'proximal_femur' } };
  assert.equal(resultWithSavedAxis(hip), hip.result);
  assert.throws(() => resultWithSavedAxis({ ...item, savedAxis: { top: [0, 90], bottom: [0, 10] } }));
});

test('draft axis updates axis and combined verdict without changing artifacts or saved export', () => {
  const original = { ...result, checks: [...result.checks, check('spine_artifacts', false)] };
  const source = { ...item, result: original };
  const draft = resultWithDraftAxis(source, axis)!;
  assert.equal(draft.quality_class, 0);
  assert.equal(draft.checks.find((c) => c.check_id === 'spine_axis')?.violation, false);
  assert.equal(draft.checks.find((c) => c.check_id === 'spine_artifacts'), original.checks[2]);
  assert.equal(draft.checks.find((c) => c.check_id === 'spine_axis')?.confidence, null);
  assert.equal(reportTable([source], 'submission').rows[0][4], 1);
  const artifactFailed = { ...source, result: { ...original, checks: [...original.checks.slice(0, 2), check('spine_artifacts', true)], violation_types: ['spine_artifacts', 'spine_axis'] } };
  assert.equal(resultWithDraftAxis(artifactFailed, axis)?.quality_class, 1);
  assert.deepEqual(resultWithDraftAxis(artifactFailed, axis)?.violation_types, ['spine_artifacts']);
  assert.equal(resultWithDraftAxis(source, null), original);
});

test('submission export keeps exactly eight automatic fields, DICOM identifiers and ZIP paths', () => {
  const corrected: WorkspaceItem = { ...item, savedAxis: axis, inputPath: 'study-2/scan.dcm',
    result: { ...result, study_uid: '1.2.3', image_uid: '4.5.6' } };
  const submission = reportTable([corrected], 'submission');
  assert.equal(submission.headers.length, 8);
  assert.deepEqual(submission.rows[0], ['study-2/scan.dcm', '1.2.3', '4.5.6', 'lumbar_spine', 1, 'spine_axis', 'Success', 1]);
  assert.equal(excelReportData([corrected], 'submission')[0].length, 8);
  const csv = reportCsv([corrected], 'submission');
  assert.match(csv, /"study-2\/scan.dcm","1.2.3","4.5.6","lumbar_spine","1","spine_axis","Success","1"/);
  assert.equal(csv.split('\r\n')[0].split(',').length, 8);
  const sameBasename: WorkspaceItem = { ...corrected, id: 'b', inputPath: 'study-3/scan.dcm',
    result: { ...corrected.result!, study_uid: '7.8.9', image_uid: '10.11.12' } };
  assert.deepEqual(reportTable([corrected, sameBasename], 'submission').rows.map((row) => row[0]),
    ['study-2/scan.dcm', 'study-3/scan.dcm']);
  const failed: WorkspaceItem = { ...item, id: 'failed', status: 'error', result: undefined, filename: 'broken.dcm' };
  assert.deepEqual(reportTable([failed], 'submission').rows[0], ['broken.dcm', '', '', 'unknown', '', '', 'Failure', 0]);
});

test('submission validation catches queued rows, missing DICOM identifiers and duplicate paths', () => {
  const dicom: WorkspaceItem = { ...item, inputPath: 'study_a/scan.dcm',
    result: { ...result, study_uid: '1.2.3', image_uid: '4.5.6' } };
  assert.deepEqual(validateSubmission([dicom]), []);
  assert.deepEqual(validateSubmission([dicom, { ...dicom, id: 'b', inputPath: 'study_b/scan.dcm',
    result: { ...dicom.result!, image_uid: '7.8.9' } }]), []);
  assert.ok(validateSubmission([{ ...dicom, result: { ...dicom.result!, study_uid: null } }]).some((warning) => warning.includes('UID')));
  assert.ok(validateSubmission([dicom, { ...dicom, id: 'b' }]).some((warning) => warning.includes('пути')));
  assert.ok(validateSubmission([dicom, { ...dicom, id: 'q', status: 'queued', result: undefined }]).some((warning) => warning.includes('завершена')));
  assert.ok(validateSubmission([{ ...dicom, id: 'z', filename: 'archive.zip', inputPath: undefined, status: 'error', result: undefined }])
    .some((warning) => warning.includes('архива')));
});

test('CSV and Excel export saved verdict, coordinates and heuristic provenance', () => {
  const saved: WorkspaceItem = { ...item, savedAxis: axis };
  const csv = reportCsv([saved, { ...item, id: 'b' }]);
  const rows = csv.split('\r\n');
  assert.match(rows[0], /assessment_source,axis_top_x,axis_top_y,axis_bottom_x,axis_bottom_y,axis_angle_deg,axis_rule/);
  assert.match(rows[1], /"0","","Success","1","manual_axis_heuristic","50","10","50","90","0","angle_gt_5_deg"/);
  assert.match(rows[2], /"1","spine_axis","Success","1","automatic"/);
  const data = excelReportData([saved, { ...item, id: 'b' }]);
  assert.equal(data[1][4].value, 0);
  assert.equal(data[1][8].value, 'manual_axis_heuristic');
  assert.deepEqual(data[1].slice(9, 14).map((cell) => cell.value), [50, 10, 50, 90, 0]);
  assert.equal(data[2][4].value, 1);
  assert.equal(excelReportData([item])[0].length, 15);
  assert.equal(reportTable([{ ...item, status: 'queued', result: undefined }], 'clinical').headers.length, 15);
  assert.equal(excelReportData([item], 'submission')[0].length, 8);
});
