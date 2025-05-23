from prometheus_client import start_http_server, Histogram

# Metrics
vehicle_travel_duration = Histogram(
    "vehicle_travel_duration_seconds",
    "Duration vehicles take to travel between zones",
    ["zone_in", "zone_out", "intersection"],
    buckets=[1, 2, 3, 5, 10, 15, 30, 60, 120],
)


def start_exporter(port=8000):
    start_http_server(port)


def report_vehicle(zone_in, zone_out, intersection, duration_sec):
    zone_in, zone_out = str(zone_in), str(zone_out)
    vehicle_travel_duration.labels(
        zone_in=zone_in, zone_out=zone_out, intersection=intersection
    ).observe(duration_sec)
