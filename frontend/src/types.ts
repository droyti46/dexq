export type AnatomicalRegion = 'auto' | 'lumbar_spine' | 'proximal_femur' | 'unknown';
export type CheckStatus = 'passed' | 'failed' | 'not_evaluated' | 'error';
export type Point = [number, number];

export interface ImageGeometry {
  image_width: number;
  image_height: number;
  coordinate_system: 'native_pixels_x_right_y_down';
  axis_line: { top_xy: Point; bottom_xy: Point; angle_deg: number | null } | null;
  vertebral_candidates: Array<{ center_xy: Point }>;
  gap_lines: Array<{ endpoints_xy: [Point, Point] }>;
  foreground_bbox: [number, number, number, number] | null;
}

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
  projection: 'AP' | 'PA' | 'unknown';
  projection_source: 'dicom_view_position' | 'not_determined';
  needs_review: boolean;
  quality_class: 0 | 1 | null;
  violation_types: string[];
  processing_status: 'Success' | 'Failure';
  time_of_processing: number;
  checks: CheckResult[];
  preview_data_url: string;
  annotated_data_url: string | null;
  geometry: ImageGeometry | null;
  error: string | null;
}

export interface BatchResult {
  items: Array<{ filename: string; input_position: number | null; result: AnalysisResult | null; error: string | null }>;
  successful: number;
  failed: number;
}

