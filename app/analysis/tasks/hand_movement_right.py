# tasks/hand_movement_right.py

import json
import math
import cv2
import mediapipe as mp
import numpy as np
from rest_framework.decorators import api_view
from rest_framework.response import Response
from multiprocessing import Pool, cpu_count
from django.core.files.storage import FileSystemStorage
import os, uuid, time, json, traceback
from django.conf import settings


from .base_task import BaseTask
from app.analysis.detectors.mp_hand_detector import HandDetector
from app.analysis.signal_analyzers.peakfinder_signal_analyzer import PeakfinderSignalAnalyzer

class HandMovementRightTask(BaseTask):
    """
    For hand movement right task:
      - We detect the full hand (index/middle/ring fingertips + wrist).
      - The signal is average fingertip-to-wrist distance.
      - The normalization factor is the max of (middle_finger,wrist) distance across frames.
    """

    # ------------------------------------------------------------------
    # --- START: Abstract properties definitions
    # ------------------------------------------------------------------
    LANDMARKS = {
        "WRIST": 0,
        "THUMB_CMC": 1,
        "THUMB_MCP": 2,
        "THUMB_IP": 3,
        "THUMB_TIP": 4,
        "INDEX_FINGER_MCP": 5,
        "INDEX_FINGER_PIP": 6,
        "INDEX_FINGER_DIP": 7,
        "INDEX_FINGER_TIP": 8,
        "MIDDLE_FINGER_MCP": 9,
        "MIDDLE_FINGER_PIP": 10,
        "MIDDLE_FINGER_DIP": 11,
        "MIDDLE_FINGER_TIP": 12,
        "RING_FINGER_MCP": 13,
        "RING_FINGER_PIP": 14,
        "RING_FINGER_DIP": 15,
        "RING_FINGER_TIP": 16,
        "PINKY_MCP": 17,
        "PINKY_PIP": 18,
        "PINKY_DIP": 19,
        "PINKY_TIP": 20
    }

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
    # ----------------------------------------------------------------
    # --- END: Abstract properties definitions
    # ----------------------------------------------------------------





    # -------------------------------------------------------------
    # --- START: Abstract methods definitions
    # -------------------------------------------------------------
    def api_response(self, request):
        try:
            # 1) Process video and define all abstract class parameters
            self.prepare_video_parameters(request)


            # 3) Get analyzer
            signal_analyzer = self.get_signal_analyzer()

            # 4) Extract landmarks using the defined detector
            with Pool(processes=max(1, cpu_count() // 2)) as pool:
                result = pool.apply(
                    HandMovementRightTask.extract_landmarks,
                    args=(self.file_path, self.start_frame_idx, self.end_frame_idx, self.fps, self.enlarged_bounding_box, self.LANDMARKS)
                )
                essential_landmarks, all_landmarks = result

            # 5) Calculate the signal using the land marks
            normalization_factor = self.calculate_normalization_factor(all_landmarks)

            # 6) Calculate the  normalization factor using the land marks
            raw_signal = self.calculate_signal(essential_landmarks)
            
            # 7) Get output from the signal analyzer
            results = signal_analyzer.analyze(
                normalization_factor=normalization_factor,
                raw_signal=raw_signal,
                start_time=self.start_time,
                end_time=self.end_time
            )
            
            # 6) Structure output
            output = {}
            output['File name'] = self.file_name
            output['Task name'] = self.task_name
            output = output | results
            output["landMarks"] = essential_landmarks
            output["allLandMarks"] = all_landmarks
            output["normalization_factor"] = normalization_factor

        except Exception as e:
            raise Exception(str(e))

        if self.video:
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
    
        #Get all necessary class attributes
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

        #Set all necessary class attributes
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
        """
        Getter for the detector used for the task.

        Returns:
            - An object of the detector class
        """
        return HandDetector().get_detector()
    


    def get_signal_analyzer(self):
        """
        Getter for the signal analyzer used for the task.

        Returns:
            - An object of the analyzer class
        """
        return PeakfinderSignalAnalyzer()
    

    
    @staticmethod
    def extract_landmarks(video_path, start_frame_idx, end_frame_idx, fps, enlarged_bounding_box, LANDMARKS) -> tuple:
        """
        Process video frames between start_frame and end_frame and extract hand landmarks 
        for the right hand from each frame.
        """

        # Set up the necessary variables
        detector = HandDetector().get_detector()
        essential_landmarks = []
        all_landmarks = []
        enlarged_coords = (
            enlarged_bounding_box['x'],
            enlarged_bounding_box['y'],
            enlarged_bounding_box['x'] + enlarged_bounding_box['width'],
            enlarged_bounding_box['y'] + enlarged_bounding_box['height']
        )
        
        # Start at the given frame index
        video = cv2.VideoCapture(video_path)
        video.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)
        current_frame_idx = start_frame_idx

        while current_frame_idx < end_frame_idx:
            success, frame = video.read()
            if not success:
                break

            # Step 1: Convert to RGB and crop the frame using the bounding box.
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            x1, y1, x2, y2 = enlarged_coords
            image_data = rgb_frame[y1:y2, x1:x2, :].astype(np.uint8)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_data)

            # Step 2: Run the detector using a timestamp computed from the frame index.
            timestamp = int(current_frame_idx / fps * 1000)
            detection_result = detector.detect_for_video(image, timestamp)

            # Step 3: Look for the right hand landmarks.
            hand_index = -1
            handedness = detection_result.handedness
            for idx in range(0, len(handedness)):
                if handedness[idx][0].category_name == "Right":
                    hand_index = idx
                
            if hand_index == -1 or not detection_result.hand_landmarks[hand_index]:
                essential_landmarks.append([])
                all_landmarks.append([])
            else:
                hand_landmarks = detection_result.hand_landmarks[hand_index]

                # Extract the landmark coordinates for key points.
                index_finger = BaseTask.get_landmark_coords(hand_landmarks[LANDMARKS["INDEX_FINGER_TIP"]], enlarged_coords)
                middle_finger = BaseTask.get_landmark_coords(hand_landmarks[LANDMARKS["MIDDLE_FINGER_TIP"]], enlarged_coords)
                ring_finger = BaseTask.get_landmark_coords(hand_landmarks[LANDMARKS["RING_FINGER_TIP"]], enlarged_coords)
                wrist = BaseTask.get_landmark_coords(hand_landmarks[LANDMARKS["WRIST"]], enlarged_coords)
                essential = [index_finger, middle_finger, ring_finger, wrist]

                # Retrieve all landmarks from the detection.
                all_lms = BaseTask.get_all_landmarks_coord(hand_landmarks, enlarged_coords)
                
                essential_landmarks.append(essential)
                all_landmarks.append(all_lms)
            
            # Step 4: Move on to next frame
            current_frame_idx += 1

        return essential_landmarks, all_landmarks

    
    def calculate_signal(self, essential_landmarks):
        """
        For each frame, measure average distance of (index, middle, ring) to wrist. If missing landmarks, fallback to previous distance.

        Returns:
            - Array of the distance for each frame.
        """
        signal = []
        prev_dist = 0
        for frame_lms in essential_landmarks:
            if len(frame_lms) < 4:
                signal.append(prev_dist)
                continue

            index_finger, middle_finger, ring_finger, wrist = frame_lms
            distance = (
                math.dist(index_finger, wrist) +
                math.dist(middle_finger, wrist) +
                math.dist(ring_finger, wrist)
            ) / 3.0

            prev_dist = distance
            signal.append(distance)

        return signal

    def calculate_normalization_factor(self, essential_landmarks):
        """
        Calculates the max distance from middle_finger to the wrist across all frames.
        
        Returns:
            - The max
        """
        distances = []
        for frame_lms in essential_landmarks:
            if len(frame_lms) < 4:
                continue
            middle_finger = frame_lms[1]
            wrist = frame_lms[3]
            d = math.dist(middle_finger, wrist)
            distances.append(d)

        return max(distances) if distances else 1.0
    # -------------------------------------------------------------
    # --- END: Abstract methods definitions
    # -------------------------------------------------------------