from dataclasses import dataclass, field
import math

@dataclass
class Track:
    track_id: int
    box: tuple
    center: tuple
    last_seen: float
    first_seen: float
    zone: str = "SAFE"
    water_since: float | None = None
    was_in_water: bool = False
    trail: list = field(default_factory=list)

class CentroidTracker:
    """Small dependency-free tracker; suitable for a fixed camera and moderate crowd."""
    def __init__(self, max_distance=90, max_lost=2.0):
        self.max_distance = max_distance
        self.max_lost = max_lost
        self.tracks = {}
        self.next_id = 1

    @staticmethod
    def center(box):
        x1,y1,x2,y2 = box
        return ((x1+x2)//2, (y1+y2)//2)

    def update(self, detections, now):
        unmatched = set(self.tracks)
        assignments = []
        # Greedy nearest-centroid association, prioritizing short distances.
        candidates = []
        for tid, tr in self.tracks.items():
            for i, det in enumerate(detections):
                c = self.center(det)
                d = math.hypot(c[0]-tr.center[0], c[1]-tr.center[1])
                if d <= self.max_distance:
                    candidates.append((d, tid, i))
        used_dets = set()
        for _, tid, i in sorted(candidates):
            if tid not in unmatched or i in used_dets:
                continue
            unmatched.remove(tid)
            used_dets.add(i)
            assignments.append((tid, i))

        for tid, i in assignments:
            tr = self.tracks[tid]
            tr.box = detections[i]
            tr.center = self.center(detections[i])
            tr.last_seen = now
            tr.trail.append(tr.center)
            tr.trail = tr.trail[-30:]

        for i, box in enumerate(detections):
            if i in used_dets:
                continue
            c = self.center(box)
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = Track(tid, box, c, now, now, trail=[c])

        lost = []
        for tid in list(unmatched):
            tr = self.tracks[tid]
            if now - tr.last_seen > self.max_lost:
                lost.append(self.tracks.pop(tid))
        return list(self.tracks.values()), lost
