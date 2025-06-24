import os
import math
import json
import uuid
import numpy as np
import traceback
from django.core.files.storage import FileSystemStorage
import gc
import tensorflow as tf
from tensorflow.python.eager.context import context
import numpy as np
import cv2
import numpy as np
from tqdm import tqdm
import tensorflow_hub as hub
import gc
from PIL import Image
from .base_task import BaseTask
from django.conf import settings
from rest_framework.response import Response

from app.analysis.signal_analyzers.gait_signal_analyzer import GaitSignalAnalyzer
from app.analysis.models.gait_transformer.gait_phase_transformer_old import load_default_model, get_gait_phase_stride_transformer, gait_phase_stride_inference
from app.analysis.models.gait_transformer.gait_phase_kalman import gait_kalman_smoother, compute_phases, get_event_times



class GaitTask(BaseTask):

    # ------------------------------------------------------------------
    # --- START: Abstract properties definitions
    # ------------------------------------------------------------------
    LANDMARKS = np.array(['htop', 'neck', 'rsho', 'relb', 'rwri', 'lsho',
                            'lelb', 'lwri', 'rhip', 'rkne', 'rank', 'lhip', 
                            'lkne', 'lank', 'pelv', 'spin', 'head'])
        
    # Properties are set via prepare_video_parameters.
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

    _metrabs_detector = None
    _gait_phase_transformer = None
    skeleton = 'mpi_inf_3dhp_17'
    _metrabs_joint_order = np.array(['htop', 'neck', 'rsho', 'relb', 'rwri', 'lsho',
                            'lelb', 'lwri', 'rhip', 'rkne', 'rank', 'lhip', 
                            'lkne', 'lank', 'pelv', 'spin', 'head'])
    _gait_phase_joint_order = ['pelv', 'rhip', 'rkne', 'rank', 'lhip', 'lkne', 
                            'lank', 'spin', 'neck', 'head', 'htop', 'lsho', 
                            'lelb', 'lwri', 'rsho', 'relb', 'rwri']
    _gait_phase_order_idx = None
    # ------------------------------------------------------------------
    # --- END: Abstract properties definitions
    # ------------------------------------------------------------------
    




    # -------------------------------------------------------------
    # --- START: Abstract methods definitions
    # -------------------------------------------------------------
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
                print("Warning: Video project data cannot be decoded.")
                return Response({}, status=404)
                    
        #Get all necessary class attributes
        file_name = metadata["metadata"]["video_name"]
        file_path = os.path.join(settings.MEDIA_ROOT, "video_uploads", video_id, file_name)
        task_name = f"{json_data['task_name']}_{json_data['id']}"
    
        video = cv2.VideoCapture(file_path)
        video_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fps = video.get(cv2.CAP_PROP_FPS)
        start_time = json_data['start_time']
        end_time = json_data['end_time']
        start_frame_idx = math.floor(fps * start_time)
        end_frame_idx   = math.ceil(fps * end_time)

        original_bounding_box = json_data['boundingBox']
        subject_bounding_boxes = [box for box in json_data['subject_bounding_boxes'] if start_frame_idx <= box['frameNumber'] <= end_frame_idx]

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
    
        focal_length = int(json_data.get('focal_length')) if json_data.get('focal_length') else -1
        height_cm = int(json_data.get('height')) if json_data.get('height') else None
        if (len(subject_bounding_boxes) != end_frame_idx - start_frame_idx + 1):
            print("Number of frames", end_frame_idx - start_frame_idx)
            print("Len of subject bounding boxes", len(subject_bounding_boxes))
            raise Exception("Number of subject bounding boxes does not match number of frames. Potentially chosen subject in some frames of the chosen task clip.")
        if (focal_length < 5 or focal_length > 35):
            focal_length = -1
        if(height_cm == None):
            raise Exception("Invalid or missing height in POST data")


        #Set all necessary class attributes
        self.video = video
        self.file_path = file_path
        self.file_name = file_name
        self.original_bounding_box = original_bounding_box
        self.enlarged_bounding_box = enlarged_bounding_box
        self.subject_bounding_boxes = subject_bounding_boxes
        self.start_time = start_time
        self.end_time = end_time
        self.fps = fps
        self.start_frame_idx = start_frame_idx
        self.end_frame_idx = end_frame_idx
        self.focal_length = focal_length
        self.height_cm = height_cm
        self.task_name = task_name

        return {
            "video": video,
            "file_name": file_name,
            "file_path": file_path,
            "original_bounding_box": original_bounding_box,
            "enlarged_bounding_box": enlarged_bounding_box,
            "start_time": start_time,
            "end_time": end_time,
            "start_frame_idx": start_frame_idx,
            "end_frame_idx": end_frame_idx,
            "focal_length": focal_length,
            "height_cm": height_cm,
        }



    def get_detector(self) -> object:
        """
        Getter for the detector used by the task.

        Returns an instance of the detector using the detectors classes
        """
        with tf.device('/CPU:0'):
            if GaitTask._metrabs_detector is None:
                print("Grabbing metrabs models")
                model_path = os.path.join(settings.BASE_DIR, 'app', 'analysis', 'models', 'metrabs_local_s' )
                

                if os.path.isdir(model_path):
                    GaitTask._metrabs_detector = hub.load(model_path)
                    print("Model loaded from ./metrabs_local_s")
                else:
                    model = hub.load('https://bit.ly/metrabs_s')
                    tf.saved_model.save(model, model_path)
                    GaitTask._metrabs_detector = hub.load(model_path)
                    print("Model saved to ./metrabs_local_s")

            return GaitTask._metrabs_detector



    def get_signal_analyzer(self) -> object:
        """
        Getter for the signal analyzer used by the task

        Returns an instance of the signal analyze using the analyzer classes
        """
        return GaitSignalAnalyzer()

    

    def calculate_signal(self, poses3D, height_mm, L=60, pos_divider=2) -> dict:
        """
        Processes 3D keypoints using the gait transformer model and returns phases, strides.

        Parameters:
            output_directory (str): Directory to save the resulting JSON file.
            height (float): Subject height in mm
            L (int): Window length for inference
            pos_divider (int): Positional divider used in model loading
        """

        if GaitTask._gait_phase_transformer is None:
            GaitTask._gait_phase_transformer = load_default_model(pos_divider=2)
        GaitTask._gait_phase_order_idx = np.array(
            [self._metrabs_joint_order.tolist().index(j) for j in GaitTask._gait_phase_joint_order]
        )

        keypoints = poses3D.copy()[:, GaitTask._gait_phase_order_idx]
        keypoints = keypoints / 1000.0
        keypoints = keypoints - np.mean(keypoints, axis=1, keepdims=True)
        keypoints = keypoints[:, :, [0, 2, 1]]
        keypoints[:, :, 2] *= -1


        # Run inference
        height_arr = np.array(height_mm, dtype=float)
        phases, strides = gait_phase_stride_inference(keypoints, height_arr, GaitTask._gait_phase_transformer, L * pos_divider)
            
        signals = {}

        # Foot position
        signals["Foot right angle"] = list(strides[:, 0])
        signals["Foot left angle"] = list(strides[:, 1])

        # Foot velocity
        signals["Foot right velocity"] = list(strides[:, 3])
        signals["Foot left velocity"] = list(strides[:, 4])

        # Pelvis velocity
        signals["Pelvis velocity"] = list(strides[:, 2])

        # Hip angles
        signals["Hip right angle"] = list(strides[:, 5])
        signals["Hip left angle"] = list(strides[:, 6])

        # Knee angles
        signals["Knee right angle"] = list(strides[:, 7])
        signals["Knee left angle"] = list(strides[:, 8])

        # Phases
        signals["Phase 0"] = list(phases[:, 0])
        signals["Phase 1"] = list(phases[:, 1])
        signals["Phase 2"] = list(phases[:, 2])
        signals["Phase 3"] = list(phases[:, 3])

        signals = {key: [float(v) for v in value] for key, value in signals.items()}
        return phases, strides, signals




    def extract_landmarks(self, detector=None) -> tuple:
        """
        Process video frames between start_frame and end_frame and extract hand landmarks 
        for the left hand from each frame — both normal and horizontally mirrored versions.

        Returns:
            tuple: (all_preds, mirrored_all_preds)
                - all_preds: dict with keys "poses2d", "poses3d", "boxes" for the normal frames
                - mirrored_all_preds: dict with keys "poses2d", "poses3d", "boxes" for the mirrored frames
        """
        file_path = self.file_path
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        start_frame = self.start_frame_idx
        end_frame = self.end_frame_idx
        fps = self.fps
        focal_length_equivalent = self.focal_length
        height_cm = self.height_cm
        enlarged_bounding_box = self.enlarged_bounding_box

        poses2d_lists, poses3d_lists, boxes_lists = [], [], []
        poses2d_lists_mirr, poses3d_lists_mirr, boxes_lists_mirr = [], [], []
        missing_mask = []
        multiple_people_detected = False

        # camera intrinsics compute
        cap = cv2.VideoCapture(file_path)
        orig_width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        orig_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        cap.release()
        diag_35x24mm = (36**2 + 24**2) ** 0.5
        diag_px = (orig_width**2 + orig_height**2)**0.5
        fx = fy = focal_length_equivalent * (diag_px / diag_35x24mm)
        cx, cy = orig_width/2.0, orig_height/2.0
        K_full = np.array([[fx, 0, cx],
                        [0, fy, cy],
                        [0,  0,  1]], dtype=np.float32)
        K_tensor = tf.convert_to_tensor(K_full, dtype=tf.float32)

        # reader
        batch_size = 16
        vid = self.video_reader(file_path, batch_size, start_frame)
        raw_frame_idx = start_frame
        stop = False

        for frame_batch in tqdm(vid, desc=f"Processing {file_name}"):
            # collect just the frames in the desired [start_frame, end_frame] range
            frames = []
            for frame in frame_batch:
                if raw_frame_idx < start_frame:
                    raw_frame_idx += 1
                    continue
                if raw_frame_idx > end_frame:
                    stop = True
                    break
                frames.append(frame)
                raw_frame_idx += 1
            if stop and not frames:
                break
            if not frames:
                continue

            # --- Prepare tensors for BOTH original and mirrored batches ---
            batch_np = np.stack(frames)
            batch_tensor = tf.convert_to_tensor(batch_np, dtype=tf.uint8)
            # mirror via TensorFlow
            batch_tensor_mirr = tf.image.flip_left_right(batch_tensor)
            n = batch_tensor.shape[0]
            K_batch = tf.tile(tf.expand_dims(K_tensor, 0), [n,1,1])
            frame_idx_list = list(range(raw_frame_idx - len(frames), raw_frame_idx))

            # --- Create bounding boxes for original frames ---
            boxes_list = []
            for frame_num in frame_idx_list:
                bbox_entry = next((item for item in self.subject_bounding_boxes if item["frameNumber"] == frame_num), None)
                if bbox_entry and bbox_entry["data"]:
                    subject_box = next(box for box in bbox_entry["data"])
                    if subject_box:
                        x, y = subject_box["x"], subject_box["y"]
                        w, h = subject_box["width"], subject_box["height"]
                        boxes_list.append([[float(x), float(y), float(w), float(h)]])
                    else:
                        raise Exception(f"Subject bounding box not found for frame idx {frame_num}")
                else:
                    raise Exception(f"Subject bounding box not found for frame idx {frame_num}")
            boxes = tf.ragged.constant(boxes_list, ragged_rank=1, inner_shape=(4,), dtype=tf.float32)
            
            # --- Create bounding boxes for mirrored frames ---
            mirrored_boxes_list = []
            for box_per_frame in boxes_list:
                original_box = box_per_frame[0]
                x, y, w, h = original_box
                mirrored_x = orig_width - (x + w)
                mirrored_boxes_list.append([[mirrored_x, y, w, h]])
            boxes_mirrored = tf.ragged.constant(mirrored_boxes_list, ragged_rank=1, inner_shape=(4,), dtype=tf.float32)

            print("boxes.shape:", boxes.shape)
            print("boxes.dtype:", boxes.dtype)
            print("boxes_mirrored.shape:", boxes_mirrored.shape)
            print("boxes_mirrored.dtype:", boxes_mirrored.dtype)

            # run detector with instrinc matrix
            if focal_length_equivalent != -1:
                pred = GaitTask._metrabs_detector.estimate_poses_batched(
                    images=batch_tensor,
                    intrinsic_matrix=K_batch,
                    boxes=boxes,
                    skeleton=self.skeleton,
                )
                pred_mirr = GaitTask._metrabs_detector.estimate_poses_batched(
                    images=batch_tensor_mirr,
                    intrinsic_matrix=K_batch,
                    boxes=boxes_mirrored,
                    skeleton=self.skeleton,
                )
            # run detector without instrinc matrix
            else:
                pred = GaitTask._metrabs_detector.estimate_poses_batched(
                    images=batch_tensor,
                    boxes=boxes,
                    skeleton=self.skeleton,
                )
                pred_mirr = GaitTask._metrabs_detector.estimate_poses_batched(
                    images=batch_tensor_mirr,
                    boxes=boxes_mirrored,
                    skeleton=self.skeleton,
                )

            # --- Accumulate both original and mirrored detections ---
            for j in range(n):
                # ORIGINAL
                if pred["poses2d"][j].shape[0] > 0:
                    poses2d_lists.append(pred["poses2d"][j:j+1, 0:1].numpy())
                    poses3d_lists.append(pred["poses3d"][j:j+1, 0:1].numpy())
                    missing = False
                else:
                    # fill NaNs
                    poses2d_lists.append(np.full([1,1,17,2], np.nan, np.float16))
                    poses3d_lists.append(np.full([1,1,17,3], np.nan, np.float16))
                    missing = True
                missing_mask.append(missing)
                if pred["poses2d"][j].shape[0] > 1:
                    multiple_people_detected = True

                # MIRRORED
                if pred_mirr["poses2d"][j].shape[0] > 0:
                    poses2d_lists_mirr.append(pred_mirr["poses2d"][j:j+1, 0:1].numpy())
                    poses3d_lists_mirr.append(pred_mirr["poses3d"][j:j+1, 0:1].numpy())
                else:
                    poses2d_lists_mirr.append(np.full([1,1,17,2], np.nan, np.float16))
                    poses3d_lists_mirr.append(np.full([1,1,17,3], np.nan, np.float16))

            del pred, pred_mirr, batch_tensor, batch_tensor_mirr

        # --- Post‐processing for ORIGINAL ---
        all_poses2d = np.concatenate(poses2d_lists, axis=0)[:,0,:,:]
        ox1 = self.original_bounding_box['x']
        oy1 = self.original_bounding_box['y']
        all_poses2d[..., 0] -= ox1
        all_poses2d[..., 1] -= oy1
        all_poses3d = np.concatenate(poses3d_lists, axis=0)[:,0,:,:]
        missing_mask = np.array(missing_mask)
        interp2d = self.interpolate_missing_poses(all_poses2d, missing_mask)
        interp3d = self.interpolate_missing_poses(all_poses3d, missing_mask)
        corr3d   = self.correct_left_right_swapping(interp3d)
        all_preds = {"poses2d": interp2d, "poses3d": corr3d}

        # --- Post‐processing for MIRRORED (same pipeline) ---
        mir_poses2d = np.concatenate(poses2d_lists_mirr, axis=0)[:,0,:,:]
        mir_poses2d[...,0] -= ox1
        mir_poses2d[...,1] -= oy1
        mir_poses3d = np.concatenate(poses3d_lists_mirr, axis=0)[:,0,:,:]
        mir_interp2d = self.interpolate_missing_poses(mir_poses2d, missing_mask)
        mir_interp3d = self.interpolate_missing_poses(mir_poses3d, missing_mask)
        mir_corr3d   = self.correct_left_right_swapping(mir_interp3d)
        mirrored_all_preds = {
            "poses2d": mir_interp2d,
            "poses3d": mir_corr3d,
        }

        # --- warnings & return ---
        if multiple_people_detected:
            print(f"Warning: {file_name} had multiple people in some frames.")
        if missing_mask.sum() > 0:
            print(f"Warning: {missing_mask.sum()} frames found no person, saved under undetected dir.")
        print(f"Completed processing {file_name}")

        return all_preds, mirrored_all_preds



    def calculate_normalization_factor(self, essential_landmarks) -> float:
        """
        Return a caluclated scalar factor used to normalize the raw 1D signal.
        """
        return None
    


    def calculate_landmark_colors(self, poses_3D, gait_event_dic, fps=None):
        """
        Colour the left/right ankles green during stance and blue during swing.
        """
        import numpy as np

        def to_idx(arr):
            """Floor to int and keep inside [0, n_frames-1]."""
            return np.clip(np.floor(arr).astype(int), 0, n_frames - 1)

        def stance_mask(downs, ups):
            """Build stance Boolean mask with no ordering assumptions."""
            ev = [(int(t), 'd') for t in downs] + [(int(t), 'u') for t in ups]
            ev.sort(key=lambda x: x[0])

            mask = np.zeros(n_frames, dtype=bool)
            in_stance = bool(ev and ev[0][1] == 'u')
            last_t = 0
            for t, kind in ev:
                t = np.clip(t, 0, n_frames)
                if in_stance:
                    mask[last_t:t] = True
                in_stance = (kind == 'd')
                last_t = t
            if in_stance:
                mask[last_t:] = True
            return mask

        n_frames, n_joints = poses_3D.shape[:2]
        landmark_colors = np.full((n_frames, n_joints, 3), [255, 0, 0], dtype=np.uint8)

        # joint indices in your Metrabs order
        L_ANK = int(np.where(self._metrabs_joint_order == "lank")[0][0])
        R_ANK = int(np.where(self._metrabs_joint_order == "rank")[0][0])

        # convert lists → int frame indices
        ld = to_idx(gait_event_dic.get('left_down',  []))
        lu = to_idx(gait_event_dic.get('left_up',    []))
        rd = to_idx(gait_event_dic.get('right_down', []))
        ru = to_idx(gait_event_dic.get('right_up',   []))

        # stance masks
        left_stance  = stance_mask(ld, lu)
        right_stance = stance_mask(rd, ru)

        # colour ankles
        landmark_colors[left_stance,  L_ANK] = [0, 255,   0]   # stance green
        landmark_colors[~left_stance, L_ANK] = [0,   0, 255]   # swing blue
        landmark_colors[right_stance, R_ANK] = [0, 255,   0]
        landmark_colors[~right_stance, R_ANK] = [0,   0, 255]

        return landmark_colors
    # -------------------------------------------------------------
    # --- END: Abstract methods definitions
    # -------------------------------------------------------------





    # -------------------------------------------------------------
    # --- START: Custom helper methods definitions
    # -------------------------------------------------------------
    # ----- Function for calculating the averages features of original and mirrored videos
    def calculate_average_features(self, original_features, mirrored_features):
        average = {
            "Average stance time": (original_features["Average stance time"] + mirrored_features["Average stance time"]) / 2.0,
            "Average swing time": (original_features["Average swing time"] + mirrored_features["Average swing time"]) / 2.0,
            "Average double support time": (original_features["Average double support time"] + mirrored_features["Average double support time"]) / 2.0,
            "Average step time": (original_features["Average step time"] + mirrored_features["Average step time"]) / 2.0,
            "Average step length": (original_features["Average step length"] + mirrored_features["Average step length"]) / 2.0,
            "Average velocity": (original_features["Average velocity"] + mirrored_features["Average velocity"]) / 2.0,
            "Average cadence": (original_features["Average cadence"] + mirrored_features["Average cadence"]) / 2.0,
            "Average stance time left": (original_features["Average stance time left"] + mirrored_features["Average stance time right"]) / 2.0,
            "Average stance time right": (original_features["Average stance time right"] + mirrored_features["Average stance time left"]) / 2.0,
            "Average swing time left": (original_features["Average swing time left"] + mirrored_features["Average swing time right"]) / 2.0,
            "Average swing time right": (original_features["Average swing time right"] + mirrored_features["Average swing time left"]) / 2.0,
            "Average step time left": (original_features["Average step time left"] + mirrored_features["Average step time right"]) / 2.0,
            "Average step time right": (original_features["Average step time right"] + mirrored_features["Average step time left"]) / 2.0,
            "Average step length left": (original_features["Average step length left"] + mirrored_features["Average step length right"]) / 2.0,
            "Average step length right": (original_features["Average step length right"] + mirrored_features["Average step length left"]) / 2.0,
            "Arm swing correlation": (original_features["Arm swing correlation"] + mirrored_features["Arm swing correlation"]) / 2.0,
        }
                
        return average
    
    ### ----- Function for interpolating missing poses -----
    def interpolate_missing_poses(self, poses: np.ndarray, missing_mask: np.ndarray) -> np.ndarray:
        """
        Fill missing frames in `poses` by linear interpolation between the closest
        non-missing frames on each side.  If only one neighbor exists, does a constant
        fill.  Raises if all frames are missing or if any missing frame has no neighbors.
        
        Args:
            poses        (T, J, D) array of keypoints with NaNs for missing frames
            missing_mask (T,)  boolean array, True where frame is entirely missing
        
        Returns:
            poses_interp (T, J, D) array with missing frames filled
        """
        poses_interp = poses.copy()
        T, J, D = poses.shape
        missing_mask = missing_mask.astype(bool)
        
        # At least one frame must be valid
        if missing_mask.all():
            raise ValueError("Cannot interpolate when all frames are missing.")
        
        # Compute nearest valid index on the left for each t
        idx = np.arange(T)
        left_valid = np.where(~missing_mask, idx, -1)
        left_neighbors = np.maximum.accumulate(left_valid)
        
        # Compute nearest valid index on the right for each t
        rev_idx = idx[::-1]
        right_valid_rev = np.where(~missing_mask[::-1], rev_idx, T)
        right_neighbors_rev = np.minimum.accumulate(right_valid_rev)
        right_neighbors = right_neighbors_rev[::-1]
        
        # Indices of frames to fill
        miss_idx = np.nonzero(missing_mask)[0]
        
        # Check every missing frame has at least one neighbor
        no_left  = left_neighbors[miss_idx]  == -1
        no_right = right_neighbors[miss_idx] == T
        if np.any(no_left & no_right):
            bad = miss_idx[no_left & no_right]
            raise ValueError(f"Frame(s) {bad.tolist()} have no neighbors with real values.")
        
        # 1) Frames with both neighbors → linear interp
        both = ~(no_left | no_right)
        mi_both  = miss_idx[both]
        li, ri  = left_neighbors[mi_both], right_neighbors[mi_both]
        dt = (ri - li).astype(float)
        w_right = (mi_both - li) / dt    # weight for the 'right' point
        w_left  = (ri - mi_both) / dt    # weight for the 'left'  point
        
        P_left  = poses[li]              # (N, J, D)
        P_right = poses[ri]              # (N, J, D)
        interp  = (P_left * w_left[:,None,None]
                + P_right * w_right[:,None,None])
        poses_interp[mi_both] = interp
        
        # 2) Frames with only left neighbor → constant fill
        left_only_idx = miss_idx[no_right & ~no_left]
        poses_interp[left_only_idx] = poses[left_neighbors[left_only_idx]]
        
        # 3) Frames with only right neighbor → constant fill
        right_only_idx = miss_idx[no_left & ~no_right]
        poses_interp[right_only_idx] = poses[right_neighbors[right_only_idx]]
        
        return poses_interp
    


    def correct_left_right_swapping(self, poses, window_size=3, margin=100):
        """
        For each frame f, directly compare that frame's left/right
        joint positions to each of the previous `window_size` frames.
        If a majority say 'swap is more consistent', then swap at f.
        Margin is 100mm
        """
        metrabs_joint_order = np.array([
            'htop','neck','rsho','relb','rwri','lsho','lelb','lwri',
            'rhip','rkne','rank','lhip','lkne','lank','pelv','spin','head'
        ])
        IDX = {name: i for i, name in enumerate(metrabs_joint_order)}
        PAIRS = [
            ('lwri','rwri'),
            ('lelb','relb'),
            ('lsho','rsho'),
            ('lank','rank'),
            ('lkne','rkne'),
            ('lhip','rhip'),
        ]

        F, J, _ = poses.shape
        fixed_poses = poses.copy()
        swapped = False

        for f in range(1, F):
            start = max(0, f - window_size)
            prev_idxs = range(start, f)
            for left_name, right_name in PAIRS:
                Li, Ri = IDX[left_name], IDX[right_name]
                curL, curR = fixed_poses[f, Li], fixed_poses[f, Ri]

                swap_votes = 0
                for p in prev_idxs:
                    prevL, prevR = fixed_poses[p, Li], fixed_poses[p, Ri]
                    dd_same = np.linalg.norm(curL - prevL) + np.linalg.norm(curR - prevR)
                    dd_swap = np.linalg.norm(curL - prevR) + np.linalg.norm(curR - prevL)

                    if dd_swap + margin < dd_same:
                        swap_votes += 1

                if swap_votes > len(prev_idxs) / 2.0:
                    swapped = True
                    fixed_poses[f, [Li, Ri]] = fixed_poses[f, [Ri, Li]]
        if swapped: print("Warning: Swap was performed on frames")
        return fixed_poses
    


    # Video Reader helper function
    def video_reader(self, filepath, batch_size=4, start_frame=0):
        if not os.path.isfile(filepath):
            print("Error: File path is not a video")
            return None, None
        
        cap = cv2.VideoCapture(filepath)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frames = []
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                if frames:
                    yield np.stack(frames)
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            if len(frames) == batch_size:
                yield np.stack(frames)
                frames = []
        cap.release()

        
    # -------------------------------------------------------------
    # --- END: Custom helper methods definitions
    # -------------------------------------------------------------