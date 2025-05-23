import argparse
from typing import Dict, Iterable, List, Set

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

import supervision as sv

COLORS = sv.ColorPalette.from_hex(["#E6194B", "#3CB44B", "#FFE119", "#3C76D1"])
KEEP_CLEAR_COLOR = sv.Color.from_hex("#FF00FF")

# AM BELOW

# Cooley and Bell
# ZONE_IN_POLYGONS = [
#     np.array([[1891, 706], [1884, 380], [2138, 377], [2126, 714]]),
#     np.array([[2131, 1337], [2137, 1868], [2329, 1855], [2306, 1335]]),
#     np.array([[1735, 1178], [1729, 1448], [1173, 1370], [1175, 1136]]),
# ]

# ZONE_OUT_POLYGONS = [
#     np.array([[2139, 722], [2140, 253], [2295, 248], [2308, 720]]),
#     np.array([[2124, 1341], [1876, 1344], [1886, 1896], [2131, 1899]]),
#     np.array([[1735, 1159], [1741, 929], [1233, 961], [1223, 1119]]),
# ]

# Donohoe and E Bayshore
# ZONE_IN_POLYGONS = [
#     np.array([[1200, 657], [1743, 518], [1602, 15], [1010, 12]]),
#     np.array([[2546, 1015], [2508, 645], [3258, 605], [3281, 952]]),
#     np.array([[2058, 1705], [2323, 1707], [2293, 2145], [2009, 2142]]),
# ]

# ZONE_OUT_POLYGONS = [
#     np.array([[1916, 480], [2184, 372], [2057, 23], [1794, 24]]),
#     np.array([[2575, 1397], [2535, 1131], [3117, 1069], [3157, 1337]]),
#     np.array([[1684, 1700], [1280, 1678], [1257, 2137], [1673, 2150]]),
# ]

# Donohoe and Cooley
# ZONE_IN_POLYGONS = [
#     np.array([[1356, 317], [1794, 398], [1755, 10], [1502, 3]]),
#     np.array([[2160, 902], [2165, 612], [2746, 774], [2697, 1043]]),
#     np.array([[2429, 1902], [1776, 1672], [1803, 2017], [2361, 2075]]),
#     np.array([[1564, 1478], [1370, 952], [349, 1023], [352, 1556]])
# ]

# ZONE_OUT_POLYGONS = [
#     np.array([[1793, 400], [1752, 9], [1970, 3], [2042, 459]]),
#     np.array([[2557, 1765], [2519, 1214], [3205, 1455], [3052, 1992]]),
#     np.array([[0, 0], [0, 0], [0, 0], [0, 0]]),
#     np.array([[1349, 884], [1198, 455], [487, 559], [513, 961]])
# ]

# Donohoe and Euclid
# ZONE_IN_POLYGONS = [
#     np.array([[1779, 1056], [2039, 1074], [2055, 520], [1881, 531]]),
#     np.array([[2450, 1496], [2446, 1311], [3077, 1283], [3111, 1430]]),
#     np.array([[1611, 1708], [1685, 1455], [1158, 1312], [1045, 1532]])
# ]

# ZONE_OUT_POLYGONS = [
#     np.array([[2025, 1094], [2289, 1092], [2292, 406], [2032, 397]]),
# np.array([[2445, 1512], [3113, 1548], [3152, 1808], [2424, 1821]]),
# np.array([[1671, 1428], [1705, 1205], [1183, 1122], [1088, 1279]])
# ]

# Donohoe and Highway 101 NB Off Ramp
# ZONE_IN_POLYGONS = [
#     np.array([[1151, 549], [1561, 540], [1476, 143], [1210, 124]]),
#     np.array([[2537, 965], [2533, 566], [3136, 567], [3174, 978]]),
#     np.array([[2395, 1794], [1567, 1749], [1580, 2137], [2265, 2141]]),
#     np.array([[1219, 1424], [1220, 1133], [483, 1137], [476, 1434]]),
# ]

# ZONE_OUT_POLYGONS = [
#     np.array([[1689, 511], [2098, 510], [2037, 161], [1734, 151]]),
#     np.array([[2532, 1648], [2545, 1067], [3296, 1116], [3294, 1550]]),
#     np.array([[0, 0], [0, 0], [0, 0], [0, 0]]),
#     np.array([[1229, 1133], [1242, 552], [443, 559], [420, 1131]]),
# ]

# University and Donohoe
# ZONE_IN_POLYGONS = [
#     np.array([[1513, 645], [1898, 621], [1893, 87], [1506, 62]]),
#     np.array([[2310, 723], [2487, 1138], [3264, 1136], [3201, 730]]),
#     np.array([[1878, 1666], [2017, 2142], [2631, 2143], [2571, 1646]]),
#     np.array([[1402, 1374], [1397, 1003], [709, 997], [691, 1278]])
# ]

# ZONE_OUT_POLYGONS = [
#     np.array([[1910, 615], [2193, 587], [2187, 24], [1893, 18]]),
#     np.array([[2656, 1533], [2495, 1142], [3083, 1135], [3075, 1344]]),
#     np.array([[1824, 1674], [1605, 1669], [1724, 2138], [1944, 2146]]),
#     np.array([[1403, 988], [1403, 768], [788, 762], [772, 985]])
# ]

# University and 101 SB Onramp
# ZONE_IN_POLYGONS = [
#     np.array([[1913, 602], [1482, 629], [1402, 13], [1843, 11]]),
#     np.array([[2409, 951], [2445, 380], [2995, 214], [3038, 656]]),
#     np.array([[1906, 1743], [2457, 1757], [2407, 2143], [1889, 2147]])
# ]

# ZONE_OUT_POLYGONS = [
#     np.array([[1924, 584], [2396, 383], [2158, 12], [1857, 15]]),
#     np.array([[2424, 1130], [2543, 1525], [3093, 989], [2903, 836]]),
#     np.array([[1735, 1706], [1486, 1691], [1438, 2131], [1754, 2136]])
# ]

# University and Woodland
# ZONE_IN_POLYGONS = [
#     np.array([[1544, 566], [2096, 512], [2130, 8], [1563, 6]]),
#     np.array([[2696, 1129], [2782, 948], [3445, 923], [3467, 1091]]),
#     np.array([[1920, 1663], [2403, 1522], [2350, 2140], [1903, 2134]]),
#     np.array([[1048, 1609], [960, 1178], [41, 1370], [20, 1827]])
# ]

# ZONE_OUT_POLYGONS = [
#     np.array([[2154, 495], [2175, 13], [2536, 24], [2506, 465]]),
#     np.array([[2590, 1354], [2684, 1138], [3145, 1123], [3143, 1283]]),
#     np.array([[1434, 1803], [1909, 1667], [1885, 2146], [1477, 2146]]),
#     np.array([[956, 1167], [927, 987], [125, 1086], [169, 1337]])
# ]


# PM BELOW
# University and Donohoe
# ZONE_IN_POLYGONS = [
#     np.array([[1400, 593], [1841, 556], [1848, 35], [1417, 42]]),
#     np.array([[2503, 1164], [2287, 677], [3314, 703], [3353, 1158]]),
#     np.array([[1795, 1755], [2602, 1736], [2599, 2148], [1894, 2150]]),
#     np.array([[1267, 1406], [1142, 996], [561, 987], [541, 1311]]),
# ]

# ZONE_OUT_POLYGONS = [
#     np.array([[1843, 555], [2174, 546], [2168, 17], [1839, 10]]),
#     np.array([[2502, 1174], [2667, 1588], [3344, 1328], [3323, 1169]]),
#     np.array([[1483, 1745], [1730, 1744], [1826, 2145], [1570, 2146]]),
#     np.array([[1259, 985], [1260, 745], [744, 742], [730, 978]]),
# ]

# Donohoe and Cooley
# ZONE_IN_POLYGONS = [
#     np.array([[1971, 480], [1570, 366], [1699, 21], [1929, 13]]),
#     np.array([[2274, 1027], [2284, 688], [2972, 919], [2903, 1173]]),
#     np.array([[1913, 1725], [2539, 1978], [2481, 2148], [1937, 2102]]),
#     np.array([[1729, 1549], [1543, 1006], [807, 1026], [792, 1561]])
# ]

# ZONE_OUT_POLYGONS = [
#     np.array([[1945, 466], [1940, 14], [2169, 14], [2238, 542]]),
#     np.array([[2665, 1303], [2636, 1872], [3071, 2043], [3222, 1505]]),
#     np.array([[0, 0], [0, 0], [0, 0], [0, 0]]),
#     np.array([[1528, 929], [1383, 480], [500, 567], [494, 981]])
# ]

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

# University and Bay
# ZONE_IN_POLYGONS = [
#     np.array([[1470, 663], [2000, 686], [2034, 111], [1540, 86]]),
#     np.array([[2515, 1114], [2547, 592], [3321, 684], [3315, 1156]]),
#     np.array([[1808, 1673], [2347, 1682], [2168, 2146], [1678, 2142]]),
#     np.array([[1162, 1610], [1239, 1097], [482, 1148], [430, 1642]])
# ]

# ZONE_OUT_POLYGONS = [
#     np.array([[2069, 691], [2463, 594], [2461, 73], [2102, 52]]),
#     np.array([[2469, 1637], [2514, 1184], [3110, 1214], [3127, 1615]]),
#     np.array([[1755, 1732], [1302, 1729], [1234, 2141], [1622, 2139]]),
#     np.array([[1225, 1083], [1274, 814], [635, 869], [619, 1135]])
# ]

# University and Runnymede
# ZONE_IN_POLYGONS = [
#     np.array([[1685, 772], [2153, 743], [2350, 320], [1859, 276]]),
#     np.array([[2668, 1259], [2755, 1045], [3201, 1056], [3151, 1276]]),
#     np.array([[1869, 1631], [2375, 1667], [2313, 2140], [1853, 2144]]),
#     np.array([[1268, 1459], [1321, 1212], [507, 1172], [487, 1412]])
# ]

# ZONE_OUT_POLYGONS = [
#     np.array([[2223, 751], [2471, 140], [2843, 141], [2567, 740]]),
#     np.array([[2559, 1496], [2656, 1286], [3373, 1300], [3354, 1524]]),
#     np.array([[1815, 1634], [1363, 1589], [1367, 2140], [1793, 2146]]),
#     np.array([[1297, 1234], [774, 1201], [790, 1042], [1333, 1052]])
# ]

# East Bayshore and Embarcadero
# ZONE_IN_POLYGONS = [
#     np.array([[1829, 1449], [1581, 1431], [1730, 1134], [1907, 1156]]),
#     np.array([[2066, 1574], [2292, 1526], [2254, 1429], [2064, 1430]]),
#     np.array([[1836, 1771], [1870, 2082], [2096, 2066], [2045, 1784]]),
#     np.array([[1509, 1627], [1582, 1540], [1264, 1507], [1215, 1583]])
# ]

# ZONE_OUT_POLYGONS = [
#     np.array([[1870, 1444], [2045, 1453], [2005, 1224], [1906, 1216]]),
#     np.array([[2071, 1576], [2138, 1667], [2331, 1600], [2281, 1529]]),
#     np.array([[1801, 1726], [1641, 1746], [1673, 1962], [1817, 1954]]),
#     np.array([[1580, 1533], [1622, 1456], [1391, 1458], [1375, 1513]])
# ]

# East Bayshore and Laura
# ZONE_IN_POLYGONS = [
#     np.array([[2222, 1356], [2128, 1449], [1857, 1196], [1940, 1146]]),
#     np.array([[2576, 1294], [2807, 1072], [2711, 1028], [2462, 1256]]),
#     np.array([[2488, 1621], [2142, 2107], [1944, 2013], [2315, 1561]])
# ]

# ZONE_OUT_POLYGONS = [
#     np.array([[2237, 1352], [2332, 1270], [2173, 1172], [2067, 1225]]),
#     np.array([[2804, 1090], [2911, 1133], [2749, 1333], [2593, 1284]]),
#     np.array([[2180, 1509], [1922, 1799], [2049, 1871], [2311, 1563]])
# ]

# Bay and Ralmar
# ZONE_IN_POLYGONS = [
#     np.array([[1947, 1026], [2035, 947], [1715, 766], [1627, 860]]),
#     np.array([[2431, 880], [2310, 838], [2469, 609], [2577, 641]]),
#     np.array([[2305, 1613], [2234, 1893], [2449, 1937], [2480, 1581]]),
#     np.array([[1777, 1532], [1587, 1658], [1687, 1802], [1911, 1677]]),
# ]

# ZONE_OUT_POLYGONS = [
#     np.array([[2196, 873], [2045, 931], [1806, 801], [1856, 763]]),
#     np.array([[2482, 880], [2635, 685], [2795, 726], [2662, 929]]),
#     np.array([[2063, 1747], [1964, 2013], [2176, 2004], [2287, 1626]]),
#     np.array([[1664, 1395], [1483, 1482], [1560, 1608], [1768, 1514]]),
# ]


class DetectionsManager:
    def __init__(self) -> None:
        self.tracker_id_to_zone_id: Dict[int, int] = {}
        # {zone_out_id: {zone_in_id: {tracker_id}}}
        self.counts: Dict[int, Dict[int, Set[int]]] = {}

        # map from tracker_id to frame number when first seen
        self.tracker_id_to_keep_clear_first_frame: Dict[int, int] = {}

    # Updates the class_id for detections based on the zones they are in
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
                self.tracker_id_to_zone_id.setdefault(tracker_id, zone_in_id)

        for zone_out_id, detections_out_zone in enumerate(detections_out_zones):
            for tracker_id in detections_out_zone.tracker_id:
                if tracker_id in self.tracker_id_to_zone_id:
                    # TODO: Track exit time and calculate durations to get from zone in to zone out
                    zone_in_id = self.tracker_id_to_zone_id[tracker_id]
                    self.counts.setdefault(zone_out_id, {})
                    self.counts[zone_out_id].setdefault(zone_in_id, set())
                    self.counts[zone_out_id][zone_in_id].add(tracker_id)

        # Uses detections_keep_clear_zones to track the first frame when a vehicle was seen in the keep clear zone
        for tracker_id in detections_keep_clear_zones.tracker_id:
            self.tracker_id_to_keep_clear_first_frame.setdefault(
                tracker_id, frame_index
            )

        if len(detections_all) > 0:
            # class_id is set to the zone in id for each detection
            detections_all.class_id = np.vectorize(
                lambda x: self.tracker_id_to_zone_id.get(x, -1)
            )(detections_all.tracker_id)
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

        self.detections_manager = DetectionsManager()

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
        annotated_frame = self.confidence_label_annotator.annotate(
            annotated_frame, detections, confidence_labels
        )

        # Draws the number of frames since the vehicle was first seen in the keep clear zone
        annotated_frame = self.keep_clear_label_annotator.annotate(
            annotated_frame,
            detections,
            [
                (
                    str(
                        self.detections_manager.tracker_id_to_keep_clear_first_frame[
                            tracker_id
                        ]
                    )
                    if tracker_id
                    in self.detections_manager.tracker_id_to_keep_clear_first_frame
                    else ""
                )
                for tracker_id in detections.tracker_id
            ],
            # TODO: Get the frame_index and subtract the first frame when the vehicle was seen in the keep clear zone
            # [
            #     str(
            #         frame_index
            #         - self.detections_manager.tracker_id_to_keep_clear_first_frame[
            #             tracker_id
            #         ]
            #     )
            #     for tracker_id in detections.tracker_id
            # ],
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
        # At this point, all of the metrics should be ready to upload to whatever metric ingestion:
        #   - zone_in_id
        #   - zone_out_id
        #   - tracker_id
        #   - class_id
        #   - counts to each zone out from each zone in
        return self.annotate_frame(frame, detections)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Traffic Flow Analysis with YOLO and ByteTrack"
    )

    parser.add_argument(
        "--source_weights_path",
        required=True,
        help="Path to the source weights file",
        type=str,
    )
    parser.add_argument(
        "--source_video_path",
        required=True,
        help="Path to the source video file",
        type=str,
    )
    parser.add_argument(
        "--target_video_path",
        default=None,
        help="Path to the target video file (output)",
        type=str,
    )
    parser.add_argument(
        "--confidence_threshold",
        default=0.3,
        help="Confidence threshold for the model",
        type=float,
    )
    parser.add_argument(
        "--iou_threshold", default=0.7, help="IOU threshold for the model", type=float
    )

    args = parser.parse_args()
    processor = VideoProcessor(
        source_weights_path=args.source_weights_path,
        source_video_path=args.source_video_path,
        target_video_path=args.target_video_path,
        confidence_threshold=args.confidence_threshold,
        iou_threshold=args.iou_threshold,
    )
    processor.process_video()
