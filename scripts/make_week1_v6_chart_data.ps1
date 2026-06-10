param(
    [string]$V6ResultsRoot = "$env:USERPROFILE\Downloads\v6-results",
    [string]$OutputDir = "outputs\PRESENTATION MATERIAL\week1_v6_chart_data"
)

$ErrorActionPreference = "Stop"

$v6 = Get-ChildItem -LiteralPath $V6ResultsRoot -Directory | Sort-Object Name | Select-Object -Last 1
if (-not $v6) {
    throw "No V6 result directories found under $V6ResultsRoot"
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$timing = Import-Csv -LiteralPath (Join-Path $v6.FullName "timing_by_track_count.csv")
$area = Import-Csv -LiteralPath (Join-Path $v6.FullName "area_reliability.csv")
$model = Import-Csv -LiteralPath (Join-Path $v6.FullName "model_comparison.csv")

function Get-VersionLabel([string]$mode) {
    switch ($mode) {
        "v4-serial-baseline" { return "V4 serial baseline" }
        "v5-shared" { return "V5 shared wrapper" }
        "v6-gated-sot-memory" { return "V6 gated SOT memory" }
        default { return $mode }
    }
}

function Get-ConfigLabel([string]$config) {
    switch ($config) {
        "B-224" { return "ViT-B / 224" }
        "L-224" { return "ViT-L / 224" }
        "g-224" { return "ViT-g / 224" }
        default { return $config }
    }
}

function To-NullableDouble($value) {
    if ($null -eq $value -or "$value" -eq "") {
        return $null
    }
    return [double]$value
}

$timingChart = $timing |
    Sort-Object lorat_config, execution_mode, { [int]$_.target_tracks } |
    ForEach-Object {
        [pscustomobject][ordered]@{
            sequence = $_.sequence
            config = $_.lorat_config
            config_label = Get-ConfigLabel $_.lorat_config
            version = $_.execution_mode
            version_label = Get-VersionLabel $_.execution_mode
            target_objects = [int]$_.target_tracks
            actual_objects = [int]$_.actual_tracks
            frames = [int]$_.frames
            update_frames = [int]$_.update_frames
            boxes_tracking = [int]$_.boxes_tracking
            tracking_fps = [double]$_.fps_tracking
            tracking_ms_per_box = [double]$_.tracking_ms_per_bbox
            mean_iou = [double]$_.mean_iou
            iou_at_0_50 = [double]$_.iou50
            peak_gpu_reserved_mb = [double]$_.gpu_memory_peak_reserved_mb
            max_evaluator_batch = [int]$_.max_evaluator_batch
            model_forward_items_per_update_frame = [double]$_.model_forward_items_per_update_frame
            model_forward_items_per_box = [double]$_.model_forward_items_per_bbox
            fps_sustains_25 = $_.fps_sustains_25
        }
    }
$timingChart | Export-Csv -NoTypeInformation -LiteralPath (Join-Path $OutputDir "week1_v6_timing_all_versions_chart.csv")

$speedRows = foreach ($config in ($timing | Select-Object -ExpandProperty lorat_config -Unique)) {
    $counts = $timing |
        Where-Object { $_.lorat_config -eq $config } |
        Select-Object -ExpandProperty target_tracks -Unique |
        Sort-Object { [int]$_ }
    foreach ($n in $counts) {
        $base = $timing | Where-Object { $_.lorat_config -eq $config -and $_.target_tracks -eq $n -and $_.execution_mode -eq "v4-serial-baseline" } | Select-Object -First 1
        $v5 = $timing | Where-Object { $_.lorat_config -eq $config -and $_.target_tracks -eq $n -and $_.execution_mode -eq "v5-shared" } | Select-Object -First 1
        $v6r = $timing | Where-Object { $_.lorat_config -eq $config -and $_.target_tracks -eq $n -and $_.execution_mode -eq "v6-gated-sot-memory" } | Select-Object -First 1
        if ($base) {
            $serialFps = [double]$base.fps_tracking
            [pscustomobject][ordered]@{
                sequence = $base.sequence
                config = $config
                config_label = Get-ConfigLabel $config
                target_objects = [int]$n
                serial_fps = $serialFps
                v5_fps = if ($v5) { [double]$v5.fps_tracking } else { $null }
                v6_fps = if ($v6r) { [double]$v6r.fps_tracking } else { $null }
                v5_speedup_vs_serial = if ($v5 -and $serialFps -ne 0) { [double]$v5.fps_tracking / $serialFps } else { $null }
                v6_speedup_vs_serial = if ($v6r -and $serialFps -ne 0) { [double]$v6r.fps_tracking / $serialFps } else { $null }
                serial_ms_per_box = [double]$base.tracking_ms_per_bbox
                v5_ms_per_box = if ($v5) { [double]$v5.tracking_ms_per_bbox } else { $null }
                v6_ms_per_box = if ($v6r) { [double]$v6r.tracking_ms_per_bbox } else { $null }
            }
        }
    }
}
$speedRows | Export-Csv -NoTypeInformation -LiteralPath (Join-Path $OutputDir "week1_v6_speedup_vs_serial_chart.csv")

$areaChart = $area |
    Sort-Object lorat_config, execution_mode, { [int]$_.target_tracks }, min_area_px |
    ForEach-Object {
        [pscustomobject][ordered]@{
            sequence = $_.sequence
            config = $_.lorat_config
            config_label = Get-ConfigLabel $_.lorat_config
            version = $_.execution_mode
            version_label = Get-VersionLabel $_.execution_mode
            target_objects = [int]$_.target_tracks
            actual_objects = [int]$_.actual_tracks
            area_bin = $_.area_bin
            min_area_px = To-NullableDouble $_.min_area_px
            max_area_px = To-NullableDouble $_.max_area_px
            samples = [int]$_.samples
            mean_area_px = [double]$_.mean_area_px
            mean_iou = [double]$_.mean_iou
            iou_at_0_50 = [double]$_.iou50
            unreliable_rate = [double]$_.unreliable_rate
            reliable = $_.reliable
        }
    }
$areaChart | Export-Csv -NoTypeInformation -LiteralPath (Join-Path $OutputDir "week1_v6_small_object_reliability_10frame_chart.csv")

$nearThreshold = $areaChart |
    Where-Object { $_.samples -ge 5 -and ($_.iou_at_0_50 -ge 0.5 -or $_.mean_iou -ge 0.45) } |
    Sort-Object @{ Expression = "iou_at_0_50"; Descending = $true }, @{ Expression = "mean_iou"; Descending = $true }
$nearThreshold | Export-Csv -NoTypeInformation -LiteralPath (Join-Path $OutputDir "week1_v6_small_object_near_threshold_bins.csv")

$modelChart = $model |
    Sort-Object lorat_config, execution_mode, { [int]$_.target_tracks } |
    ForEach-Object {
        [pscustomobject][ordered]@{
            sequence = $_.sequence
            config = $_.lorat_config
            config_label = Get-ConfigLabel $_.lorat_config
            version = $_.execution_mode
            version_label = Get-VersionLabel $_.execution_mode
            backbone = $_.backbone
            input_size = [int]$_.input_size
            target_objects = [int]$_.target_tracks
            actual_objects = [int]$_.actual_tracks
            tracking_fps = [double]$_.fps_tracking
            tracking_ms_per_box = [double]$_.tracking_ms_per_bbox
            peak_gpu_reserved_mb = [double]$_.gpu_memory_peak_reserved_mb
            mean_iou = [double]$_.mean_iou
            iou_at_0_50 = [double]$_.iou50
            smallest_reliable_area_px = $_.smallest_reliable_area_px
        }
    }
$modelChart | Export-Csv -NoTypeInformation -LiteralPath (Join-Path $OutputDir "week1_v6_model_comparison_chart.csv")

$summaryLines = @(
    "# Week 1 V6 Chart Data",
    "",
    "Source folder: ``$($v6.FullName)``",
    "",
    "Files:",
    "- ``week1_v6_timing_all_versions_chart.csv``: FPS, ms/box, IoU, and GPU memory for V4/V5/V6 across B/L/g and object count.",
    "- ``week1_v6_speedup_vs_serial_chart.csv``: V5 and V6 FPS speedup versus V4 serial baseline.",
    "- ``week1_v6_small_object_reliability_10frame_chart.csv``: every-10-frame area-bin reliability benchmark.",
    "- ``week1_v6_small_object_near_threshold_bins.csv``: bins that got closest to reliability thresholds.",
    "- ``week1_v6_model_comparison_chart.csv``: compact model-size comparison table.",
    "",
    "Benchmark definitions:",
    "- ``tracking_fps = update_frames / tracking_seconds``.",
    "- ``tracking_ms_per_box = tracking_seconds * 1000 / boxes_tracking``.",
    "- ``IoU = intersection(predicted_box, ground_truth_box) / union(predicted_box, ground_truth_box)``.",
    "- Small-object reliability samples each track every 10 frames and compares to the same object's current-frame ground truth.",
    "- Reliable area-bin rule: ``IoU@0.50 >= 0.80``, ``mean_iou >= 0.50``, and ``samples >= 10``.",
    "",
    "Chart suggestions:",
    "- Line chart: ``target_objects`` vs ``tracking_fps``, colored by ``version_label``, faceted or filtered by ``config_label``.",
    "- Line chart: ``target_objects`` vs ``tracking_ms_per_box``, colored by ``config_label``.",
    "- Bar chart: ``v6_speedup_vs_serial`` by ``target_objects``, grouped by ``config_label``.",
    "- Line/bar chart: ``area_bin`` vs ``iou_at_0_50``, filtered to V6 or by model config.",
    "- Scatter chart: ``peak_gpu_reserved_mb`` vs ``tracking_fps``, colored by config/version."
)
$summaryLines | Set-Content -LiteralPath (Join-Path $OutputDir "README_week1_v6_chart_data.md") -Encoding UTF8

Get-ChildItem -LiteralPath $OutputDir | Select-Object Name, Length, LastWriteTime
