export type AnatomicalRegion = 'auto' | 'lumbar_spine' | 'proximal_femur' | 'unknown';
export type CheckStatus = 'passed' | 'failed' | 'not_evaluated' | 'error';

export interface CheckResult {
  check_id: string;
  title: string;
  status: CheckStatus;
  violation: boolean | null;
  summary: string;
  confidence: number | null;
  details: Record<string, unknown>;
  method: string;
  model_status: string;
}

export interface AnalysisResult {
  analysis_id: string;
  filename: string;
  study_uid: string | null;
  image_uid: string | null;
  anatomical_region: AnatomicalRegion;
  region_source: string;
  quality_class: 0 | 1 | null;
  violation_types: string[];
  processing_status: 'Success' | 'Failure';
  time_of_processing: number;
  checks: CheckResult[];
  preview_data_url: string;
}

export interface BatchResult {
  items: Array<{ filename: string; result: AnalysisResult | null; error: string | null }>;
  successful: number;
  failed: number;
}

