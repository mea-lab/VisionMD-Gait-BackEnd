# tasks/base_task.py

import cv2
import mediapipe as mp
import numpy as np
import math
import os, uuid, time, json, traceback
from django.core.files.storage import FileSystemStorage
from abc import ABC, abstractmethod
from .base_task import BaseTask
from django.conf import settings
from rest_framework.response import Response

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO
import tensorflow as tf
import tensorflow_hub as tfhub
from tensorflow.python.eager.context import context
import torch
from scipy.signal import butter, filtfilt
from scipy.signal import periodogram
import numpy as np
import tensorrt as trt
import os

class TremorTask(BaseTask):
    """
    Base class for all tasks (hand movement, finger tap, leg agility, toe tapping, etc.)
    Each concrete subclass must implement these abstract methods for retrieving
    & processing landmarks.
    """

    # ------------------------------------------------------------------
    # --- START: Abstract properties to be implemented by subclasses ---
    # ------------------------------------------------------------------
    LANDMARKS = {
        "THUMB": 0,
        "INDEX": 1,
        "MIDDLE": 2,
        "RING": 3,
        "PINKY": 4,
    }

    video_id = None
    file_path = None
    file_name = None
    task_name = None

    fps = None
    start_time = None
    start_frame_idx = None
    end_time = None
    end_frame_idx = None

    focal_length = None
    height_cm = None

    original_bounding_box = None
    enlarged_bounding_box = None
    subject_bounding_boxes = None
    # ----------------------------------------------------------------
    # --- END: Abstract properties to be implemented by subclasses ---
    # ----------------------------------------------------------------





    # ---------------------------------------------------------------
    # --- START: Abstract methods to be implemented by subclasses ---
    # ---------------------------------------------------------------
    def api_response(self, request):
        """
        Function that handles the api response for each task
        """
        try:
            # 1) Getting video parameters from request
            self.prepare_video_parameters(request)

            # 2) Getting detector and using detector to get landmarks
            with tf.device('/CPU:0'):
                detector = self.get_detector()
            landmarks, landmarks_mirrored = self.extract_landmarks(detector)    
            tf.keras.backend.clear_session()
            context().clear_kernel_cache()

            # 2) Getting signals
            phases, strides, signals = self.calculate_signal(landmarks['poses3d'], self.height_cm * 10)
            phases_mirrored, strides_mirrored, signals_mirrored = self.calculate_signal(landmarks_mirrored['poses3d'], self.height_cm * 10)

            # 3) Get signal analyzer to use it to get feature results
            signal_analyzer = self.get_signal_analyzer()
            print("Analyzing original signal...")
            results, gait_event_dic = signal_analyzer.analyze(phases, strides, landmarks['poses3d'], self.fps)
            print("Analyzing mirrored signal...")
            results_mirrored, gait_event_dic_mirrored = signal_analyzer.analyze(phases_mirrored, strides_mirrored, landmarks_mirrored['poses3d'], self.fps)
            avg_results = self.calculate_average_features(results, results_mirrored)

            del GaitTask._gait_phase_transformer, signal_analyzer
            GaitTask._gait_phase_transformer, signal_analyzer = None, None
            tf.keras.backend.clear_session()
            context().clear_kernel_cache()

            # 5) Get landmark colors
            landmark_colors = self.calculate_landmark_colors(landmarks['poses3d'], gait_event_dic, self.fps)

            # 4) Build up response to API call
            response = {}
            response['File name'] = self.file_name
            response['Task name'] = self.task_name
            response = response | avg_results
            response['signals'] = signals_mirrored
            response['landMarks'] = landmarks['poses2d'].tolist()
            response['landMarks_3D'] = landmarks['poses3d'].tolist()
            response['gait_event_dic'] = {
                k: v.tolist()
                for k, v in gait_event_dic.items()
            }
            response['landmark_colors'] = landmark_colors.tolist()
        except Exception as e:
            return Response(f"Error with gait analysis: {str(e)}", status=500)
        finally:
            # 5) Clean up memory
            if hasattr(self, "video") and self.video is not None:
                self.video.release()
            tf.keras.backend.clear_session()
            context().clear_kernel_cache()

        return response

    
    @abstractmethod
    def prepare_video_parameters(self, request):
        """
        Prepares video parameters from the HTTP request:
         - Parses JSON for bounding box and time codes.
         - Saves the uploaded video file.
         - Computes the expanded bounding box.
         - Determines FPS and start/end frame indices.
        Returns a dictionary of parameters. 
        MUST DEFINE ALL ABSTRACT PROPERTIES. 
        """
        pass


    @abstractmethod 
    def get_detector(self) -> object:
        """
        Getter for the detector used by the task.

        Returns an instance of the detector using the detectors classes
        """
        pass


    @abstractmethod
    def get_signal_analyzer(self) -> object:
        """
        Getter for the signal analyzer used by the task

        Returns an instance of the signal analyze using the analyzer classes
        """
        pass


    @abstractmethod
    def calculate_signal(self, essential_landmarks) -> list:
        """
        Given a set of display landmarks (one list per frame), return the raw 1D
        signal array.
        """
        pass

    @abstractmethod
    def extract_landmarks(self, detector) -> tuple:
        """
        Process video frames between start_frame and end_frame and extract hand landmarks 
        for the left hand from each frame.
        
        Returns:
            tuple: (essential_landmarks, all_landmarks)
            - essential_landmarks: a list of lists where each inner list contains the key landmark coordinates for that frame.
            - all_landmarks: a list of lists containing all the landmark coordinates for that frame.
        """
        pass


    @abstractmethod
    def calculate_normalization_factor(self, essential_landmarks) -> float:
        """
        Return a caluclated scalar factor used to normalize the raw 1D signal.
        """
        pass

    def bandpass_filter(signal, lowcut=2, highcut=10, fs=60):
        # Design Butterworth bandpass filter
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(N=2, Wn=[low, high], btype='band', analog=False)
        # Remove NaNs if any
        signal = np.asarray(signal)
        if np.isnan(signal).any():
            signal = np.nan_to_num(signal)
        return filtfilt(b, a, signal)

    # Function to estimate the principal frequency of a signal
    def principal_frequency(signal, fs):
        f, Pxx = periodogram(signal, fs=fs)
        idx = np.argmax(Pxx[1:]) + 1  # skip DC component
        return f[idx], np.sqrt(Pxx[idx])  # amplitude is sqrt of power at peak
    
    # -------------------------------------------------------------
    # --- END: Abstract methods to be implemented by subclasses ---
    # -------------------------------------------------------------





    # --------------------------------------------------
    # --- START: Utility functions as static methods ---
    # --------------------------------------------------
    @staticmethod
    def get_landmark_coords(landmark, enlarged_coords, original_coords):
        """
        Computes the (x, y) coordinates of a given landmark relative to the provided bounds.
        """
        x1, y1, x2, y2 = enlarged_coords
        ox1, oy1, ox2, oy2 = original_coords
        return [
            landmark.x * (x2 - x1) +  x1 - ox1,
            landmark.y * (y2 - y1) +  y1 - oy1,
        ]

    @staticmethod
    def get_all_landmarks_coord(landmarks, enlarged_coords, original_coords):
        """
        Processes a list of landmarks and returns their (x, y, z) coordinates relative
        to the provided bounds.
        """
        x1, y1, x2, y2 = enlarged_coords
        ox1, oy1, ox2, oy2 = original_coords
        coords = []
        for lm in landmarks:
            coords.append([
                lm.x * (x2 - x1) + x1 - ox1,
                lm.y * (y2 - y1) + y1 - oy1,
                lm.z
            ])
        return coords
    # ------------------------------------------------
    # --- END: Utility functions as static methods ---
    # ------------------------------------------------