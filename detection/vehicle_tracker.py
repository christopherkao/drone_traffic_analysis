import argparse
from typing import Dict, Iterable, List, Set

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

import supervision as sv

from exporter.metrics_exporter import report_vehicle

COLORS = sv.ColorPalette.from_hex(["#E6194B", "#3CB44B", "#FFE119", "#3C76D1"])
KEEP_CLEAR_COLOR = sv.Color.from_hex("#FF00FF")

FRAMES_PER_SECOND = 30.0

# Donohoe and 101 NB Offramp
ZONE_IN_POLYGONS = [
    np.array([[1328, 521], [1614, 522], [1602, 194], [1370, 183]]),
    np.array([[2504, 905], [2511, 536], [3216, 542], [3260, 895]]),
    np.array([[1622, 1463], [2442, 1495], [2315, 2142], [1691, 2140]]),
    np.array([[1385, 1059], [1360, 1321], [563, 1316], [560, 1071]]),
]

ZONE_OUT_POLYGONS = [
    np.array([[1746, 506], [2149, 508], [2065, 161], [1804, 146]]),
    np.array([[2466, 1500], [2492, 978], [3119, 1014], [3116, 1391]]),
    np.array([[0, 0], [0, 0], [0, 0], [0, 0]]),
    np.array([[1360, 1045], [1359, 553], [654, 552], [602, 1044]]),
]

KEEP_CLEAR_POLYGONS = [
    np.array([[1372, 1288], [1375, 564], [2370, 548], [2359, 1484], [1627, 1458]])
]


class DetectionsManager:
    def __init__(self, intersection_name: str) -> None:
        self.intersection_name = intersection_name
        # TODO: Augment to include frame number when vehicle was first seen in the zone
        # self.tracker_id_to_zone_in_id: Dict[int, int] = {}
        self.tracker_id_to_zone_in_id: Dict[int, tuple[int, int]] = {}
        # {zone_out_id: {zone_in_id: {tracker_id}}}
        self.counts: Dict[int, Dict[int, Set[int]]] = {}

        # map from tracker_id to frame number when first seen
        self.tracker_id_to_keep_clear_first_frame: Dict[int, int] = {}

    # Updates the class_id for detections based on the zones they are in
    # Also writes out metrics when a vehicle with a zone in reaches a zone out
    def update(
        self,
        detections_all: sv.Detections,
        detections_in_zones: List[sv.Detections],
        detections_out_zones: List[sv.Detections],
        detections_keep_clear_zones: List[sv.Detections],
        frame_index: int,
    ) -> sv.Detections:
        for zone_in_id, detections_in_zone in enumerate(detections_in_zones):
            for tracker_id in detections_in_zone.tracker_id:
                # TODO: Track entry time
                if tracker_id not in self.tracker_id_to_zone_in_id:
                    self.tracker_id_to_zone_in_id[tracker_id] = (
                        zone_in_id,
                        frame_index,
                    )
                # self.tracker_id_to_zone_in_id.setdefault(tracker_id, zone_in_id)

        for zone_out_id, detections_out_zone in enumerate(detections_out_zones):
            for tracker_id in detections_out_zone.tracker_id:
                if tracker_id in self.tracker_id_to_zone_in_id:
                    # TODO: Track exit time and calculate durations to get from zone in to zone out
                    zone_in_id, zone_in_frame_index = self.tracker_id_to_zone_in_id[
                        tracker_id
                    ]
                    self.counts.setdefault(zone_out_id, {})
                    self.counts[zone_out_id].setdefault(zone_in_id, set())
                    # TODO: If an entry is added for first time, record metric.
                    # This means that vehicle just arrived in zone out from zone in.
                    if tracker_id not in self.counts[zone_out_id][zone_in_id]:
                        duration_sec = (
                            frame_index - zone_in_frame_index
                        ) / FRAMES_PER_SECOND
                        print(
                            f"Vehicle {tracker_id} arrived in zone_out {zone_out_id} at frame {frame_index} and zone_in {zone_in_id} at frame {zone_in_frame_index}"
                        )
                        print(f"Total time in seconds: {duration_sec}")
                        # Log metric here
                        report_vehicle(
                            zone_in=zone_in_id,
                            zone_out=zone_out_id,
                            intersection=self.intersection_name,
                            duration_sec=duration_sec,
                        )
                    self.counts[zone_out_id][zone_in_id].add(tracker_id)

        # Uses detections_keep_clear_zones to track the first frame when a vehicle was seen in the keep clear zone
        for tracker_id in detections_keep_clear_zones.tracker_id:
            self.tracker_id_to_keep_clear_first_frame.setdefault(
                tracker_id, frame_index
            )

        # if len(detections_all) > 0:
        #     # class_id is set to the zone in id for each detection
        #     detections_all.class_id = np.vectorize(
        #         lambda x: self.tracker_id_to_zone_in_id.get(x, -1)
        #     )(detections_all.tracker_id)
        # else:
        #     detections_all.class_id = np.array([], dtype=int)
        if len(detections_all) > 0:
            class_ids = []
            for tracker_id in detections_all.tracker_id:
                if tracker_id in self.tracker_id_to_zone_in_id:
                    zone_in_id, _ = self.tracker_id_to_zone_in_id[tracker_id]
                    class_ids.append(zone_in_id)
                else:
                    class_ids.append(-1)
            detections_all.class_id = np.array(class_ids, dtype=int)
        else:
            detections_all.class_id = np.array([], dtype=int)
        return detections_all[detections_all.class_id != -1]


def initiate_polygon_zones(
    polygons: List[np.ndarray],
    triggering_anchors: Iterable[sv.Position] = [sv.Position.CENTER],
) -> List[sv.PolygonZone]:
    return [
        sv.PolygonZone(
            polygon=polygon,
            triggering_anchors=triggering_anchors,
        )
        for polygon in polygons
    ]


class VideoProcessor:
    def __init__(
        self,
        source_weights_path: str,
        intersection_name: str,
        source_video_path: str,
        target_video_path: str = None,
        confidence_threshold: float = 0.3,
        iou_threshold: float = 0.7,
    ) -> None:
        self.conf_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.source_video_path = source_video_path
        self.target_video_path = target_video_path

        self.model = YOLO(source_weights_path)
        self.tracker = sv.ByteTrack()

        self.video_info = sv.VideoInfo.from_video_path(source_video_path)
        self.zones_in = initiate_polygon_zones(ZONE_IN_POLYGONS, [sv.Position.CENTER])
        self.zones_out = initiate_polygon_zones(ZONE_OUT_POLYGONS, [sv.Position.CENTER])
        self.zones_keep_clear = initiate_polygon_zones(
            KEEP_CLEAR_POLYGONS, [sv.Position.CENTER]
        )

        self.bounding_box_annotator = sv.BoundingBoxAnnotator(color=COLORS)
        self.label_annotator = sv.LabelAnnotator(
            color=COLORS, text_color=sv.Color.BLACK, text_position=sv.Position.TOP_LEFT
        )
        self.confidence_label_annotator = sv.LabelAnnotator(
            color=COLORS,
            text_color=sv.Color.WHITE,
            text_position=sv.Position.TOP_RIGHT,
        )
        self.trace_annotator = sv.TraceAnnotator(
            color=COLORS, position=sv.Position.CENTER, trace_length=100, thickness=2
        )
        self.keep_clear_label_annotator = sv.LabelAnnotator(
            color=COLORS,
            text_color=KEEP_CLEAR_COLOR,
            text_position=sv.Position.BOTTOM_LEFT,
        )
        self.zone_in_frame_index_label_annotator = sv.LabelAnnotator(
            color=COLORS,
            text_color=sv.Color.WHITE,
            text_position=sv.Position.BOTTOM_LEFT,
        )

        self.detections_manager = DetectionsManager(intersection_name)

    def process_video(self):
        frame_generator = sv.get_video_frames_generator(
            source_path=self.source_video_path
        )

        if self.target_video_path:
            with sv.VideoSink(self.target_video_path, self.video_info) as sink:
                # for frame in tqdm(frame_generator, total=self.video_info.total_frames):
                for idx, frame in enumerate(
                    tqdm(frame_generator, total=self.video_info.total_frames)
                ):
                    annotated_frame = self.process_frame(frame, idx)
                    sink.write_frame(annotated_frame)
                #     annotated_frame = self.process_frame(frame)
                #     sink.write_frame(annotated_frame)
        else:
            # This was giving the error: qt.qpa.xcb: could not connect to display
            # qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "/home/christopherkao/.pyenv/versions/3.10.12/envs/traffic-analysis-env/lib/python3.10/site-packages/cv2/qt/plugins" even though it was found.
            # This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem.
            # for frame in tqdm(frame_generator, total=self.video_info.total_frames):
            for idx, frame in enumerate(
                tqdm(frame_generator, total=self.video_info.total_frames)
            ):
                annotated_frame = self.process_frame(frame, idx)
                cv2.imshow("Processed Video", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            cv2.destroyAllWindows()

    def annotate_frame(
        self, frame: np.ndarray, detections: sv.Detections
    ) -> np.ndarray:
        annotated_frame = frame.copy()
        for i, (zone_in, zone_out) in enumerate(zip(self.zones_in, self.zones_out)):
            # Annotate zone in polygons
            annotated_frame = sv.draw_polygon(
                annotated_frame, zone_in.polygon, COLORS.colors[i]
            )
            # Annotate zone out polygons
            annotated_frame = sv.draw_polygon(
                annotated_frame, zone_out.polygon, COLORS.colors[i]
            )

        # Annotate keep clear polygons
        for i, zone_keep_clear in enumerate(self.zones_keep_clear):
            annotated_frame = sv.draw_polygon(
                annotated_frame, zone_keep_clear.polygon, KEEP_CLEAR_COLOR
            )

        labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]
        confidence_labels = [f"{confidence}" for confidence in detections.confidence]
        # Draws path of detections
        annotated_frame = self.trace_annotator.annotate(annotated_frame, detections)
        # Draws bounding boxes around detections
        annotated_frame = self.bounding_box_annotator.annotate(
            annotated_frame, detections
        )
        annotated_frame = self.label_annotator.annotate(
            annotated_frame, detections, labels
        )
        # annotated_frame = self.confidence_label_annotator.annotate(
        #     annotated_frame, detections, confidence_labels
        # )

        # Draws the number of frames since the vehicle was first seen in the keep clear zone
        # annotated_frame = self.keep_clear_label_annotator.annotate(
        #     annotated_frame,
        #     detections,
        #     [
        #         (
        #             str(
        #                 self.detections_manager.tracker_id_to_keep_clear_first_frame[
        #                     tracker_id
        #                 ]
        #             )
        #             if tracker_id
        #             in self.detections_manager.tracker_id_to_keep_clear_first_frame
        #             else ""
        #         )
        #         for tracker_id in detections.tracker_id
        #     ],
        #     # TODO: Get the frame_index and subtract the first frame when the vehicle was seen in the keep clear zone
        #     # [
        #     #     str(
        #     #         frame_index
        #     #         - self.detections_manager.tracker_id_to_keep_clear_first_frame[
        #     #             tracker_id
        #     #         ]
        #     #     )
        #     #     for tracker_id in detections.tracker_id
        #     # ],
        # )

        # Draws the frame number when vehicle was first seen in zone in.
        annotated_frame = self.zone_in_frame_index_label_annotator.annotate(
            annotated_frame,
            detections,
            [
                (
                    str(self.detections_manager.tracker_id_to_zone_in_id[tracker_id][1])
                    if tracker_id
                    in self.detections_manager.tracker_id_to_keep_clear_first_frame
                    else ""
                )
                for tracker_id in detections.tracker_id
            ],
        )

        # Annotate the counts for each zone out
        for zone_out_id, zone_out in enumerate(self.zones_out):
            zone_center = sv.get_polygon_center(polygon=zone_out.polygon)
            if zone_out_id in self.detections_manager.counts:
                # Gets the counts for the current zone out
                counts = self.detections_manager.counts[zone_out_id]
                for i, zone_in_id in enumerate(counts):
                    # Gets the counts to the zone out from the current zone in
                    count = len(self.detections_manager.counts[zone_out_id][zone_in_id])
                    # Offsets to be below the display count from previous zone in
                    text_anchor = sv.Point(x=zone_center.x, y=zone_center.y + 40 * i)
                    annotated_frame = sv.draw_text(
                        scene=annotated_frame,
                        text=str(count),
                        text_anchor=text_anchor,
                        background_color=COLORS.colors[zone_in_id],
                    )

        return annotated_frame

    # This function is called for each frame in the video
    # It processes the frame and returns the annotated frame
    def process_frame(self, frame: np.ndarray, frame_index: int) -> np.ndarray:
        results = self.model(
            frame, verbose=False, conf=self.conf_threshold, iou=self.iou_threshold
        )[0]
        # This is really the main Detections call
        detections = sv.Detections.from_ultralytics(results)
        detections.class_id = np.zeros(len(detections))
        # ByteTrack update
        detections = self.tracker.update_with_detections(detections)

        detections_in_zones = []
        detections_out_zones = []

        for zone_in, zone_out in zip(self.zones_in, self.zones_out):
            # trigger returns boolean array if each detection is within the polygon zone
            # detections[] __getitem__ returns detections that are within the polygon zone. __getitem__ is an overloaded function.
            detections_in_zone = detections[zone_in.trigger(detections=detections)]
            detections_in_zones.append(detections_in_zone)
            detections_out_zone = detections[zone_out.trigger(detections=detections)]
            detections_out_zones.append(detections_out_zone)

        # Look for cars in the keep clear zone
        detections_keep_clear_zones = []
        for zone_keep_clear in self.zones_keep_clear:
            detections_keep_clear_zone = detections[
                zone_keep_clear.trigger(detections=detections)
            ]
            detections_keep_clear_zones.append(detections_keep_clear_zone)

        # not sure why we need to feed it through this update() function?
        detections = self.detections_manager.update(
            detections,
            detections_in_zones,
            detections_out_zones,
            detections_keep_clear_zone,
            frame_index,
        )

        return self.annotate_frame(frame, detections)


# This is the main entry point for the script
def run_vehicle_tracking(
    source_weights_path: str,
    intersection_name: str,
    source_video_path: str,
    target_video_path: str = None,
    confidence_threshold: float = 0.3,
    iou_threshold: float = 0.7,
):
    processor = VideoProcessor(
        source_weights_path=source_weights_path,
        intersection_name=intersection_name,
        source_video_path=source_video_path,
        target_video_path=target_video_path,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
    )
    processor.process_video()
