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

from app.analysis.signal_analyzers.gait_signal_analyzer import GaitSignalAnalyzer


class GaitTask(BaseTask):

    # ------------------------------------------------------------------
    # --- START: Abstract properties definitions
    # ------------------------------------------------------------------
    LANDMARKS = np.array(['htop', 'neck', 'rsho', 'relb', 'rwri', 'lsho',
                            'lelb', 'lwri', 'rhip', 'rkne', 'rank', 'lhip', 
                            'lkne', 'lank', 'pelv', 'spin', 'head'])

    gait_phase_joint_order = ['pelv', 'rhip', 'rkne', 'rank', 'lhip', 'lkne', 
                    'lank', 'spin', 'neck', 'head', 'htop', 'lsho', 
                    'lelb', 'lwri', 'rsho', 'relb', 'rwri']
        
    # Properties are set via prepare_video_parameters.
    original_bounding_box = None
    enlarged_bounding_box = None
    video = None
    fps = None
    start_time = None
    start_frame_idx = None
    end_time = None
    end_frame_idx = None
    file_path = None
    focal_length = None
    height_cm = None

    skeleton = 'mpi_inf_3dhp_17'
    _metrabs_detector = None
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
        print("Request:",request,"\n\n")

        # Getting landmarks from metrabs model and then clearing memory for next model
        self.prepare_video_parameters(request)
        with tf.device('/CPU:0'):
            detector = self.get_detector()

        landmarks = self.extract_landmarks(detector)    
        # del GaitTask._metrabs_detector
        # GaitTask._metrabs_detector = None
        tf.keras.backend.clear_session()
        context().clear_kernel_cache()

        # Getting signal analyzer from metrabs model and then clearing memory for next model
        signal_analyzer = self.get_signal_analyzer()
        results, phases, strides, landmark_colors = signal_analyzer.analyze(landmarks['poses3d'])
        signals = self.calculate_signal(phases, strides)
        signals = {key: [float(v) for v in value] for key, value in signals.items()}

        del GaitSignalAnalyzer._GaitPhaseTransformer, signal_analyzer
        GaitSignalAnalyzer._GaitPhaseTransformer, signal_analyzer = None, None
        tf.keras.backend.clear_session()
        context().clear_kernel_cache()

        results['signals'] = signals
        results['landMarks'] = landmarks['poses2d'].tolist()
        results['poses3D'] = landmarks['poses3d'].tolist()
        results['landmark_colors'] = landmark_colors.tolist()

        #Clean up memory
        if self.video:
            self.video.release()
        if os.path.exists(self.file_path):
            os.remove(self.file_path)
        tf.keras.backend.clear_session()
        context().clear_kernel_cache()

        return results

    
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
        APP_ROOT = os.path.dirname(os.path.abspath(__file__))
        try:
            json_data = json.loads(request.POST['json_data'])
            if 'video' not in request.FILES or len(request.FILES) == 0:
                raise Exception("'video' field missing or no files uploaded")
        except (KeyError, json.JSONDecodeError):
            raise Exception("Invalid or missing 'json_data' in POST data")
    
        file_name = f"{uuid.uuid4().hex[:15].upper()}.mp4"
        folder_path = os.path.join(APP_ROOT, '../video_uploads')
        file_path = os.path.join(folder_path, file_name)
        FileSystemStorage(folder_path).save(file_name, request.FILES['video'])
    
        video = cv2.VideoCapture(file_path)
        video_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
        #Get all necessary class attributes
        original_bounding_box = json_data['boundingBox']
        start_time = json_data['start_time']
        end_time = json_data['end_time']
        focal_length = int(json_data.get('focal_length')) if json_data.get('focal_length') else None
        height_cm = int(json_data.get('height')) if json_data.get('height') else None


        print("start_time", start_time)
        print("end_time", end_time)
        fps = video.get(cv2.CAP_PROP_FPS)
        start_frame_idx = math.floor(fps * start_time)
        end_frame_idx   = math.ceil(fps * end_time)
        print("start_time", start_frame_idx)
        print("end_time", end_frame_idx)
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
        self.file_path = file_path
        self.original_bounding_box = original_bounding_box
        self.enlarged_bounding_box = enlarged_bounding_box
        self.start_time = start_time
        self.end_time = end_time
        self.fps = fps
        self.start_frame_idx = start_frame_idx
        self.end_frame_idx = end_frame_idx
        self.focal_length = focal_length
        self.height_cm = height_cm

        return {
            "video": video,
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
        # return None
        return GaitSignalAnalyzer()

    
    def calculate_signal(self, phases, strides, walking_prob=None) -> dict:
        """
        Given a set of strides and phases a dictionary containing raw 1D signal arrays.
        """
            
        traces = {}

        # Foot position
        traces["Foot right angle"] = list(strides[:, 0])
        traces["Foot left angle"] = list(strides[:, 1])

        # Foot velocity
        traces["Foot right velocity"] = list(strides[:, 3])
        traces["Foot left velocity"] = list(strides[:, 4])

        # Pelvis velocity
        traces["Pelvis velocity"] = list(strides[:, 2])

        # Hip angles
        traces["Hip right angle"] = list(strides[:, 5])
        traces["Hip left angle"] = list(strides[:, 6])

        # Knee angles
        traces["Knee right angle"] = list(strides[:, 7])
        traces["Knee left angle"] = list(strides[:, 8])

        # Phases
        traces["Phase 0"] = list(phases[:, 0])
        traces["Phase 1"] = list(phases[:, 1])
        traces["Phase 2"] = list(phases[:, 2])
        traces["Phase 3"] = list(phases[:, 3])
        # traces["Phase left 0"] = list(phases[:, 3])

        # Walking probability
        if walking_prob is not None:
            traces["walking_prob"] = list(walking_prob[:])

        return traces

    def extract_landmarks(self, detector=None) -> tuple:
        """
        Process video frames between start_frame and end_frame and extract hand landmarks 
        for the left hand from each frame.
        
        Returns:
            tuple: (essential_landmarks, all_landmarks)
            - essential_landmarks: a list of lists where each inner list contains the key landmark coordinates for that frame.
            - all_landmarks: a list of lists containing all the landmark coordinates for that frame.
        """
        # Setting video related variables
        file_path = self.file_path
        if not os.path.isfile(file_path): raise Exception("Error: File path is not a video")

        file_name = os.path.splitext(os.path.basename(file_path))[0]
        start_frame = self.start_frame_idx
        end_frame = self.end_frame_idx
        fps = self.fps
        focal_length_equivalent = self.focal_length
        height_cm = self.height_cm
        enlarged_bounding_box = self.enlarged_bounding_box

        # Setting variables for loop that processes frames
        poses2d_lists = []
        poses3d_lists = []
        boxes_lists = []
        missing_mask = []
        multiple_people_detected = False
        raw_frame_idx = start_frame
        stop = False
        batch_size = 16

        #Setting variables for cropping and resizing 
        cap = cv2.VideoCapture(file_path)
        orig_width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        orig_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        cap.release()

        diag_35x24mm = (36**2 + 24**2) ** 0.5
        w_px, h_px = orig_width, orig_height
        diag_px   = (w_px**2 + h_px**2) ** 0.5

        fx = focal_length_equivalent * (diag_px / diag_35x24mm)
        fy = focal_length_equivalent * (diag_px / diag_35x24mm)
        cx = orig_width  / 2.0
        cy = orig_height / 2.0
        K_full = np.array([[fx,  0, cx],
                           [ 0, fy, cy],
                           [ 0,  0,  1]], dtype=np.float32)
        x1 = enlarged_bounding_box['x']
        y1 = enlarged_bounding_box['y']
        K_tensor = tf.convert_to_tensor(K_full,dtype=tf.float32)

        # Start reading in batches for the video and processing them
        vid = self.video_reader(file_path, batch_size, start_frame)
        for frame_batch in tqdm(vid, desc=f"Processing {file_name}"):
            resized_cropped_frame_batch = []
            for frame in frame_batch:
                if raw_frame_idx < start_frame:
                    raw_frame_idx += 1
                    continue
                if raw_frame_idx > end_frame:
                    stop = True
                    break
                resized_cropped_frame_batch.append(frame)
                raw_frame_idx += 1

            
            if not resized_cropped_frame_batch:
                # After end frame idx
                if stop:
                    break
                # Before start frame idx
                else:
                    continue

            # Prepare batch tensor for model
            batch_tensor = tf.convert_to_tensor(np.stack(resized_cropped_frame_batch), dtype=tf.uint8)
            n = tf.shape(batch_tensor)[0]
            K_batch = tf.tile(tf.expand_dims(K_tensor, 0), [n,1,1])

            pred = GaitTask._metrabs_detector.detect_poses_batched(
                images=batch_tensor,
                intrinsic_matrix=K_batch,
                skeleton=self.skeleton,
                detector_flip_aug=True,
                detector_threshold=0.2,
            )

            for j in range(len(resized_cropped_frame_batch)):
                if pred["poses2d"][j].shape[0] > 0:
                    missing = False
                    poses2d_lists.append(pred["poses2d"][j:j+1, 0:1, ...].numpy())
                    poses3d_lists.append(pred["poses3d"][j:j+1, 0:1, ...].numpy())
                    boxes_lists.append(pred["boxes"][j:j+1, 0:1, ...].numpy())
                else:
                    missing = True
                    poses2d_lists.append(np.full([1, 1, 17, 2], np.nan, dtype=np.float16))
                    poses3d_lists.append(np.full([1, 1, 17, 3], np.nan, dtype=np.float16))
                    boxes_lists.append(np.full([1, 1, 5], np.nan, dtype=np.float16))

                missing_mask.append(missing)
                if pred["poses2d"][j].shape[0] > 1:
                    multiple_people_detected = True

            del pred, batch_tensor

        # Complete loop, interpolate missing poses and fix incorrectly swapped poses
        all_poses2d = np.concatenate(poses2d_lists, axis=0)[:,0,:,:]
        all_poses2d[..., 0] -= x1
        all_poses2d[..., 1] -= y1

        all_poses3d = np.concatenate(poses3d_lists, axis=0)[:,0,:,:]
        all_boxes = np.concatenate(boxes_lists, axis=0)
        missing_mask = np.array(missing_mask)

        interpolated_poses3D = self.interpolate_missing_poses(all_poses3d, missing_mask)
        interpolated_poses2D = self.interpolate_missing_poses(all_poses2d, missing_mask)
        corrected_poses3D = self.correct_left_right_swapping(interpolated_poses3D)

        all_preds = {            
            "poses2d": interpolated_poses2D,
            "poses3d": corrected_poses3D,
            "boxes"  : all_boxes,
        }

        # Post‐processing warnings
        if multiple_people_detected:
            print(f"Warning: {file_name} had multiple people in some frames.")
        if sum(missing_mask) > 0:
            print(f"Warning: {sum(missing_mask)} frames found no person, saved under undetected dir.")
        print(f"Completed processing {file_name}")
        return all_preds


    def calculate_normalization_factor(self, essential_landmarks) -> float:
        """
        Return a caluclated scalar factor used to normalize the raw 1D signal.
        """
        return None
    # -------------------------------------------------------------
    # --- END: Abstract methods definitions
    # -------------------------------------------------------------





    # -------------------------------------------------------------
    # --- START: Custom helper methods definitions
    # -------------------------------------------------------------
    
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