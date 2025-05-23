from threading import Thread
from exporter.metrics_exporter import start_exporter, report_vehicle
from detection.vehicle_tracker import run_vehicle_tracking


def run_exporter():
    start_exporter(port=8000)  # Exposes /metrics


# TODO: Move this callback elsewhere
def on_vehicle_detected(zone_in, zone_out, intersection, duration_sec):
    report_vehicle(zone_in, zone_out, intersection, duration_sec)


# Entry point to run detection + exporter
if __name__ == "__main__":
    # Start metrics exporter in background
    Thread(target=run_exporter, daemon=True).start()

    # Start vehicle tracking (pass in callback to report metrics)
    run_vehicle_tracking(
        source_weights_path="data/yolo11_vehicles_weights.pt",
        intersection_name="donohoe_101_nb_offramp",
        source_video_path="data/DJI_0009_keep_clear.MP4",
        target_video_path="data/DJI_0009_keep_clear_result.MP4",
        confidence_threshold=0.3,
        iou_threshold=0.5,
    )
