# Adapted from https://github.com/joonson/syncnet_python/blob/master/run_pipeline.py

import os, pdb, subprocess, glob, cv2
import numpy as np
from shutil import rmtree
import torch
import shutil

from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector

from scipy.interpolate import interp1d
from scipy.io import wavfile
from scipy import signal

from .detectors import S3FD


class SyncNetDetector:
    def __init__(self, ckpt_path, device, detect_results_dir="detect_results"):
        self.s3f_detector = S3FD(ckpt_path=ckpt_path, device=device)
        self.detect_results_dir = detect_results_dir
        # print(f" 1: {self.detect_results_dir}")

    def __call__(self, video_path: str, min_track=50, scale=False):
        crop_dir = os.path.join(self.detect_results_dir, "crop")
        video_dir = os.path.join(self.detect_results_dir, "video")
        frames_dir = os.path.join(self.detect_results_dir, "frames")
        temp_dir = os.path.join(self.detect_results_dir, "temp")

        # ========== DELETE EXISTING DIRECTORIES ==========
        if os.path.exists(crop_dir):
            rmtree(crop_dir)

        if os.path.exists(video_dir):
            rmtree(video_dir)

        if os.path.exists(frames_dir):
            # rmtree(frames_dir)
            shutil.rmtree(frames_dir, ignore_errors=True)

        if os.path.exists(temp_dir):
            rmtree(temp_dir)

        # ========== MAKE NEW DIRECTORIES ==========

        os.makedirs(crop_dir)
        os.makedirs(video_dir)
        os.makedirs(frames_dir)
        os.makedirs(temp_dir)

        # ========== CONVERT VIDEO AND EXTRACT FRAMES ==========

        if scale:
            scaled_video_path = os.path.join(video_dir, "scaled.mp4")
            command = f"ffmpeg -loglevel error -y -nostdin -i {video_path} -vf scale='224:224' {scaled_video_path}"
            subprocess.run(command, shell=True, check=True)
            video_path = scaled_video_path

        command = f"ffmpeg -y -nostdin -loglevel error -i {video_path} -qscale:v 2 -async 1 -r 25 {os.path.join(video_dir, 'video.mp4')}"
        subprocess.run(command, shell=True, stdout=None, check=True)

        command = f"ffmpeg -y -nostdin -loglevel error -i {os.path.join(video_dir, 'video.mp4')} -qscale:v 2 -f image2 {os.path.join(frames_dir, '%06d.jpg')}"
        subprocess.run(command, shell=True, stdout=None, check=True)

        command = f"ffmpeg -y -nostdin -loglevel error -i {os.path.join(video_dir, 'video.mp4')} -ac 1 -vn -acodec pcm_s16le -ar 16000 {os.path.join(video_dir, 'audio.wav')}"
        subprocess.run(command, shell=True, stdout=None, check=True)

        faces = self.detect_face(frames_dir)

        scene = self.scene_detect(video_dir)

        # Face tracking
        alltracks = []

        for shot in scene:
            if shot[1].frame_num - shot[0].frame_num >= min_track:
                alltracks.extend(self.track_face(faces[shot[0].frame_num : shot[1].frame_num], min_track=min_track))

        # Face crop
        for ii, track in enumerate(alltracks):
            self.crop_video(track, os.path.join(crop_dir, "%05d" % ii), frames_dir, 25, temp_dir, video_dir)

        rmtree(temp_dir)

    def scene_detect(self, video_dir):
        video_manager = VideoManager([os.path.join(video_dir, "video.mp4")])
        stats_manager = StatsManager()
        scene_manager = SceneManager(stats_manager)
        # Add ContentDetector algorithm (constructor takes detector options like threshold).
        scene_manager.add_detector(ContentDetector())
        base_timecode = video_manager.get_base_timecode()

        video_manager.set_downscale_factor()

        video_manager.start()

        scene_manager.detect_scenes(frame_source=video_manager)

        scene_list = scene_manager.get_scene_list(base_timecode)

        if scene_list == []:
            scene_list = [(video_manager.get_base_timecode(), video_manager.get_current_timecode())]

        return scene_list

    def track_face(self, scenefaces, num_failed_det=25, min_track=50, min_face_size=100):
        iouThres = 0.5  # Minimum IOU between consecutive face detections
        tracks = []

        while True:
            track = []
            for framefaces in scenefaces:
                for face in framefaces:
                    if track == []:
                        track.append(face)
                        framefaces.remove(face)
                    elif face["frame"] - track[-1]["frame"] <= num_failed_det:
                        iou = bounding_box_iou(face["bbox"], track[-1]["bbox"])
                        if iou > iouThres:
                            track.append(face)
                            framefaces.remove(face)
                            continue
                    else:
                        break

            if track == []:
                break
            elif len(track) > min_track:
                framenum = np.array([f["frame"] for f in track])
                bboxes = np.array([np.array(f["bbox"]) for f in track])

                frame_i = np.arange(framenum[0], framenum[-1] + 1)

                bboxes_i = []
                for ij in range(0, 4):
                    interpfn = interp1d(framenum, bboxes[:, ij])
                    bboxes_i.append(interpfn(frame_i))
                bboxes_i = np.stack(bboxes_i, axis=1)

                if (
                    max(np.mean(bboxes_i[:, 2] - bboxes_i[:, 0]), np.mean(bboxes_i[:, 3] - bboxes_i[:, 1]))
                    > min_face_size
                ):
                    tracks.append({"frame": frame_i, "bbox": bboxes_i})

        return tracks

    def detect_face(self, frames_dir, facedet_scale=0.25):
        flist = glob.glob(os.path.join(frames_dir, "*.jpg"))
        flist.sort()

        dets = []

        for fidx, fname in enumerate(flist):
            image = cv2.imread(fname)

            image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            bboxes = self.s3f_detector.detect_faces(image_np, conf_th=0.9, scales=[facedet_scale])

            dets.append([])
            for bbox in bboxes:
                dets[-1].append({"frame": fidx, "bbox": (bbox[:-1]).tolist(), "conf": bbox[-1]})

        return dets

    def crop_video(self, track, cropfile, frames_dir, frame_rate, temp_dir, video_dir, crop_scale=0.4):
        flist = glob.glob(os.path.join(frames_dir, "*.jpg"))
        flist.sort()

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vOut = cv2.VideoWriter(cropfile + "t.mp4", fourcc, frame_rate, (224, 224))

        dets = {"x": [], "y": [], "s": []}

        for det in track["bbox"]:
            dets["s"].append(max((det[3] - det[1]), (det[2] - det[0])) / 2)
            dets["y"].append((det[1] + det[3]) / 2)  # crop center x
            dets["x"].append((det[0] + det[2]) / 2)  # crop center y

        # Smooth detections
        dets["s"] = signal.medfilt(dets["s"], kernel_size=13)
        dets["x"] = signal.medfilt(dets["x"], kernel_size=13)
        dets["y"] = signal.medfilt(dets["y"], kernel_size=13)

        for fidx, frame in enumerate(track["frame"]):
            cs = crop_scale

            bs = dets["s"][fidx]  # Detection box size
            bsi = int(bs * (1 + 2 * cs))  # Pad videos by this amount

            image = cv2.imread(flist[frame])

            frame = np.pad(image, ((bsi, bsi), (bsi, bsi), (0, 0)), "constant", constant_values=(110, 110))
            my = dets["y"][fidx] + bsi  # BBox center Y
            mx = dets["x"][fidx] + bsi  # BBox center X

            face = frame[int(my - bs) : int(my + bs * (1 + 2 * cs)), int(mx - bs * (1 + cs)) : int(mx + bs * (1 + cs))]

            vOut.write(cv2.resize(face, (224, 224)))

        audiotmp = os.path.join(temp_dir, "audio.wav")
        audiostart = (track["frame"][0]) / frame_rate
        audioend = (track["frame"][-1] + 1) / frame_rate

        vOut.release()

        # ========== CROP AUDIO FILE ==========

        command = "ffmpeg -y -nostdin -loglevel error -i %s -ss %.3f -to %.3f %s" % (
            os.path.join(video_dir, "audio.wav"),
            audiostart,
            audioend,
            audiotmp,
        )
        output = subprocess.run(command, shell=True, stdout=None, check=True)

        sample_rate, audio = wavfile.read(audiotmp)

        # ========== COMBINE AUDIO AND VIDEO FILES ==========

        command = "ffmpeg -y -nostdin -loglevel error -i %st.mp4 -i %s -c:v copy -c:a aac %s.mp4" % (
            cropfile,
            audiotmp,
            cropfile,
        )
        output = subprocess.run(command, shell=True, stdout=None, check=True)

        os.remove(cropfile + "t.mp4")

        return {"track": track, "proc_track": dets}


def bounding_box_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou
