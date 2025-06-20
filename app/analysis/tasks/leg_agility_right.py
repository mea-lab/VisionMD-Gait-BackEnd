import math
import cv2
import mediapipe as mp
import numpy as np
import os, uuid, time, json, traceback
from multiprocessing import Pool, cpu_count
from django.core.files.storage import FileSystemStorage
from django.conf import settings

from .base_task import BaseTask
from app.analysis.detectors.mp_poseheavy_detector import PoseHeavyDetector
from app.analysis.signal_analyzers.peakfinder_signal_analyzer import PeakfinderSignalAnalyzer

class LegAgilityRightTask(BaseTask):
    """
    Leg Agility task for the right leg.
    Tracks shoulder midpoint, right knee, and hip midpoint.
    Signal is the vertical difference (shoulder y-coordinate minus knee y-coordinate).
    Normalization factor is the average distance between shoulder midpoint and hip midpoint.
    """
    LANDMARKS = {
        "NOSE": 0,
        "LEFT_EYE_INNER": 1,
        "LEFT_EYE": 2,
        "LEFT_EYE_OUTER": 3,
        "RIGHT_EYE_INNER": 4,
        "RIGHT_EYE": 5,
        "RIGHT_EYE_OUTER": 6,
        "LEFT_EAR": 7,
        "RIGHT_EAR": 8,
        "MOUTH_LEFT": 9,
        "MOUTH_RIGHT": 10,
        "LEFT_SHOULDER": 11,
        "RIGHT_SHOULDER": 12,
        "LEFT_ELBOW": 13,
        "RIGHT_ELBOW": 14,
        "LEFT_WRIST": 15,
        "RIGHT_WRIST": 16,
        "LEFT_HIP": 23,
        "RIGHT_HIP": 24,
        "LEFT_KNEE": 25,
        "RIGHT_KNEE": 26,
        "LEFT_ANKLE": 27,
        "RIGHT_ANKLE": 28,
        "LEFT_HEEL": 29,
        "RIGHT_HEEL": 30,
        "LEFT_FOOT_INDEX": 31,
        "RIGHT_FOOT_INDEX": 32
    }

    # Abstract properties required by BaseTask.
    video_id = None
    file_name = None
    task_name = None
    video = None
    fps = None
    start_time = None
    start_frame_idx = None
    end_time = None
    end_frame_idx = None
    file_path = None
    original_bounding_box = None
    enlarged_bounding_box = None

    def api_response(self, request):
        try:
            # 1) Process video and define all abstract class parameters
            self.prepare_video_parameters(request)

            # 3) Get analyzer
            signal_analyzer = self.get_signal_analyzer()

            # 4) Extract landmarks using the defined detector
            with Pool(processes=max(1, cpu_count() // 2)) as pool:
                result = pool.apply(
                    LegAgilityRightTask.extract_landmarks,
                    args=(self.file_path, self.start_frame_idx, self.end_frame_idx, self.fps, self.enlarged_bounding_box, self.LANDMARKS)
                )
                essential_landmarks, all_landmarks = result

            normalization_factor = self.calculate_normalization_factor(essential_landmarks)
            raw_signal = self.calculate_signal(essential_landmarks)


            # Get output from the signal analyzer
            results = signal_analyzer.analyze(
                normalization_factor=normalization_factor,
                raw_signal=raw_signal,
                start_time=self.start_time,
                end_time=self.end_time
            )
            
            # Structure output
            output = {}
            output['File name'] = self.file_name
            output['Task name'] = self.task_name
            output = output | results
            output["landMarks"] = essential_landmarks
            output["allLandMarks"] = all_landmarks
            output["normalization_factor"] = normalization_factor

        except Exception as e:
            raise Exception(str(e))

        finally:
            if self.video and self.video.isOpened():
                self.video.release()

        return output

    def prepare_video_parameters(self, request):
        """
        Parses POST data, saves the video file, computes bounding boxes and frame indices.
        """
        # Get all variables set up and check if folder and file paths exist
        video_id = request.GET.get('id', None)
        if not video_id:
            raise Exception("Video project id not provided.")
        
        try:
            json_data = json.loads(request.POST['json_data'])
        except (KeyError, json.JSONDecodeError):
            raise Exception("Invalid or missing 'json_data' in POST data")

        folder_path = os.path.join(settings.MEDIA_ROOT, "video_uploads")
        project_folder_path = os.path.join(folder_path, video_id)
        if not os.path.isdir(project_folder_path):
            raise Exception("Video project folder does not exist.")
        
        subfolder_path = os.path.join(folder_path, video_id)
        metadata = {}
        if os.path.isdir(subfolder_path):
            json_path = os.path.join(subfolder_path, "metadata.json")
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            except (IOError, json.JSONDecodeError):
                raise Exception("Warning: Video project data cannot be decoded.")                    
    
        file_name = metadata["metadata"]["video_name"]
        file_path = os.path.join(settings.MEDIA_ROOT, "video_uploads", video_id, file_name)
        task_name = f"{json_data['task_name']}_{json_data['id']}"
    
        video = cv2.VideoCapture(file_path)
        video_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        original_bounding_box = json_data['boundingBox']
        start_time = json_data['start_time']
        end_time = json_data['end_time']
        fps = video.get(cv2.CAP_PROP_FPS)
        start_frame_idx = math.floor(fps * start_time)
        end_frame_idx   = math.floor(fps * end_time)

        new_x = int(max(0, original_bounding_box['x'] - original_bounding_box['width'] * 0.125))
        new_y = int(max(0, original_bounding_box['y'] - original_bounding_box['height'] * 0.125))
        new_width = int(min(video_width - new_x, original_bounding_box['width'] * 1.25))
        new_height = int(min(video_height - new_y, original_bounding_box['height'] * 1.25))
        enlarged_bounding_box = {
            'x': new_x,
            'y': new_y,
            'width': new_width,
            'height': new_height
        }
        
        self.video = video
        self.video_id = video_id
        self.file_name = file_name
        self.task_name = task_name
        self.file_path = file_path
        self.original_bounding_box = original_bounding_box
        self.enlarged_bounding_box = enlarged_bounding_box
        self.start_time = start_time
        self.end_time = end_time
        self.start_frame_idx = start_frame_idx
        self.end_frame_idx = end_frame_idx
        self.fps = fps

        return {
            "video": video,
            "video_id": video_id,
            "file_name": file_name,
            "task_name": task_name,
            "file_path": file_path,
            "original_bounding_box": original_bounding_box,
            "enlarged_bounding_box": enlarged_bounding_box,
            "start_time": start_time,
            "end_time": end_time,
            "start_frame_idx": start_frame_idx,
            "end_frame_idx": end_frame_idx,
            "fps": fps,
        }
    
    def get_detector(self):
        return PoseHeavyDetector().get_detector()

    def get_signal_analyzer(self):
        return PeakfinderSignalAnalyzer()

    @staticmethod
    def extract_landmarks(video_path, start_frame_idx, end_frame_idx, fps, enlarged_bounding_box, LANDMARKS) -> tuple:
        detector = PoseHeavyDetector().get_detector()

        enlarged_bb = enlarged_bounding_box
        x1 = enlarged_bb['x']
        y1 = enlarged_bb['y']
        x2 = x1 + enlarged_bb['width']
        y2 = y1 + enlarged_bb['height']
        enlarged_coords = (x1, y1, x2, y2)

        essential_landmarks = []
        all_landmarks = []

        video = cv2.VideoCapture(video_path)
        video.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)
        current_frame_idx = start_frame_idx

        while current_frame_idx < end_frame_idx:
            success, frame = video.read()
            if not success:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cropped_frame = rgb_frame[y1:y2, x1:x2, :].astype(np.uint8)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cropped_frame)

            timestamp = int((current_frame_idx / fps) * 1000)
            detection_result = detector.detect_for_video(image, timestamp)

            if not detection_result.pose_landmarks:
                essential_landmarks.append([])
                all_landmarks.append([])
            else:
                landmarks = detection_result.pose_landmarks[0]
                # Use the right knee since this is the right leg task.
                knee_idx = LANDMARKS["RIGHT_KNEE"]
                knee = [
                    landmarks[knee_idx].x * (x2 - x1),
                    landmarks[knee_idx].y * (y2 - y1)
                ]
                left_shoulder = landmarks[LANDMARKS["LEFT_SHOULDER"]]
                right_shoulder = landmarks[LANDMARKS["RIGHT_SHOULDER"]]
                shoulder_mid = [
                    ((left_shoulder.x + right_shoulder.x) / 2) * (x2 - x1),
                    ((left_shoulder.y + right_shoulder.y) / 2) * (y2 - y1)
                ]
                left_hip = landmarks[LANDMARKS["LEFT_HIP"]]
                right_hip = landmarks[LANDMARKS["RIGHT_HIP"]]
                hip_mid = [
                    ((left_hip.x + right_hip.x) / 2) * (x2 - x1),
                    ((left_hip.y + right_hip.y) / 2) * (y2 - y1)
                ]
                essential = [shoulder_mid, knee, hip_mid]
                all_lms = BaseTask.get_all_landmarks_coord(landmarks, enlarged_coords)
                essential_landmarks.append(essential)
                all_landmarks.append(all_lms)
            current_frame_idx += 1

        return essential_landmarks, all_landmarks

    def calculate_signal(self, essential_landmarks):
        signal = []
        prev_signal = 0
        for frame_lms in essential_landmarks:
            if len(frame_lms) < 3:
                signal.append(prev_signal)
                continue
            shoulder_mid, knee, _ = frame_lms
            diff = shoulder_mid[1] - knee[1]
            diff = diff if diff >= 0 else 0
            prev_signal = diff
            signal.append(diff)
        return signal

    def calculate_normalization_factor(self, essential_landmarks):
        distances = []
        for frame_lms in essential_landmarks:
            if len(frame_lms) < 3:
                continue
            shoulder_mid, _, hip_mid = frame_lms
            d = math.dist(shoulder_mid, hip_mid)
            distances.append(d)
        return float(np.mean(distances)) if distances else 1.0
