import numpy as np
import scipy.signal as signal
import scipy.signal as signal
import scipy.interpolate as interpolate
import tensorflow as tf
import tensorflow_hub as hub
from app.analysis.models.gait_transformer.gait_phase_transformer_old import load_default_model, get_gait_phase_stride_transformer, gait_phase_stride_inference
from app.analysis.models.gait_transformer.gait_phase_kalman import gait_kalman_smoother, compute_phases, get_event_times
import os
from django.conf import settings
from app.analysis.signal_analyzers.base_signal_analyzer import BaseSignalAnalyzer


class GaitSignalAnalyzer(BaseSignalAnalyzer):
    """
    Base signal analyzer for all signals (hand movement, finger tap, leg agility, toe tapping, etc.)
    Each  subclass must implement these abstract methods for analyzing a signal
    """

    # ------------------------------------------------------------------
    # --- START: Abstract properties definitions
    # ------------------------------------------------------------------

    _GaitPhaseTransformer = None
    _metrabs_joint_order = np.array(['htop', 'neck', 'rsho', 'relb', 'rwri', 'lsho',
                            'lelb', 'lwri', 'rhip', 'rkne', 'rank', 'lhip', 
                            'lkne', 'lank', 'pelv', 'spin', 'head'])
    
    _gait_phase_joint_order = ['pelv', 'rhip', 'rkne', 'rank', 'lhip', 'lkne', 
                            'lank', 'spin', 'neck', 'head', 'htop', 'lsho', 
                            'lelb', 'lwri', 'rsho', 'relb', 'rwri']
    
    _gait_phase_order_idx = None

    # ------------------------------------------------------------------
    # --- End: Abstract properties definitions
    # ------------------------------------------------------------------





    # ------------------------------------------------------------------
    # --- START: Abstract methods ---
    # ------------------------------------------------------------------
    def analyze(self, corrected_poses3D, fps = 30, height_mm = 1500) -> dict:
        GaitSignalAnalyzer._gait_phase_order_idx = np.array(
            [self._metrabs_joint_order.tolist().index(j) for j in GaitSignalAnalyzer._gait_phase_joint_order]
        )
        if GaitSignalAnalyzer._GaitPhaseTransformer is None:
            GaitSignalAnalyzer._GaitPhaseTransformer = load_default_model(pos_divider=2)

        phases_ordered, strides, gait_event_dic = self.process_gait_keypoints(corrected_poses3D, height_mm)
        strides = strides
        phases = np.reshape(phases_ordered, [-1, 4, 2])[:, :, 0]
        results = self.analyze_gait_video_features(gait_event_dic, corrected_poses3D, GaitSignalAnalyzer._gait_phase_order_idx, fps)

        # Generate land mark colors
        n_frames = corrected_poses3D.shape[0]
        n_joints = corrected_poses3D.shape[1]
        landmark_colors = np.zeros((n_frames, n_joints, 3), dtype=np.uint8)

        # indices of the ankle joints in self._gait_phase_joint_order
        LEFT_ANKLE_IDX  = self._gait_phase_joint_order.index('lank')
        RIGHT_ANKLE_IDX = self._gait_phase_joint_order.index('rank')

        # stance vs swing flags for each frame
        left_stance  = np.zeros(n_frames, dtype=bool)
        right_stance = np.zeros(n_frames, dtype=bool)

        for d, u in zip(gait_event_dic['left_down'], gait_event_dic['left_up']):
            left_stance[int(round(d)): int(round(u)) + 1] = True
        for d, u in zip(gait_event_dic['right_down'], gait_event_dic['right_up']):
            right_stance[int(round(d)): int(round(u)) + 1] = True

        # apply colors – green = stance, blue = swing
        landmark_colors[left_stance,  LEFT_ANKLE_IDX]  = [0, 255, 0]
        landmark_colors[~left_stance, LEFT_ANKLE_IDX]  = [0,   0, 255]
        landmark_colors[right_stance, RIGHT_ANKLE_IDX] = [0, 255, 0]
        landmark_colors[~right_stance,RIGHT_ANKLE_IDX] = [0,   0, 255]


        return results, phases, strides, landmark_colors


    # ------------------------------------------------------------------
    # --- END: Abstract methods ---
    # ------------------------------------------------------------------





    # ------------------------------------------------------------------
    # --- START: Helper methods ---
    # ------------------------------------------------------------------

    ### ----- Function that returns dictionary of gait events from 3D points -----
    def process_gait_keypoints(
        self,
        poses3D,
        height_mm,
        L=60,
        pos_divider=2
    ):
        """
        Processes a 3D keypoints using the gait transformer model and returns phases, strides and gait events.

        Parameters:
            output_directory (str): Directory to save the resulting JSON file.
            height (float): Subject height in mm
            L (int): Window length for inference
            pos_divider (int): Positional divider used in model loading
        """
        # Reorder, normalize and transform keypoints
        keypoints = poses3D.copy()[:, GaitSignalAnalyzer._gait_phase_order_idx]
        keypoints = keypoints / 1000.0          # mm → m
        keypoints = keypoints - np.mean(keypoints, axis=1, keepdims=True)
        keypoints = keypoints[:, :, [0, 2, 1]]  # reordering axes
        keypoints[:, :, 2] *= -1                # flip z

        # Run inference
        height_arr = np.array(height_mm, dtype=float)
        phase, stride = gait_phase_stride_inference(keypoints, height_arr, GaitSignalAnalyzer._GaitPhaseTransformer, L * pos_divider)

        # Kalman smoothing
        phase_ordered = np.take(phase, [0, 4, 1, 5, 2, 6, 3, 7], axis=-1)
        state, _, _ = gait_kalman_smoother(phase_ordered)
        timestamps = np.arange(state.shape[0])
        gait_event_dic = get_event_times(state, timestamps)

        # Return result as dictionary
        print(f"Processed 3D points")
        return phase_ordered, stride, gait_event_dic
    
    
    def analyze_gait_video_features(
        self,
        gait_event_dic: dict,
        keypoints_3D: np.ndarray,
        gait_phase_order_idx: list,
        fps,
    ) -> dict:
        """
        Analyze spatiotemporal gait features from gait event timings and 3D keypoints.

        Parameters:
            gait_event_dic (dict): Must contain four keys—
                'left_down', 'left_up', 'right_down', 'right_up'—
                each mapping to a 1D numpy array of event-frame floats.
            keypoints_3D (np.ndarray): Array of shape (n_frames, n_keypoints, 3)
                containing raw 3D keypoint coordinates.
            gait_phase_order_idx (list): Permutation index to reorder the keypoints.
            fps (int): Frames per second of the video.

        Returns:
            dict: A dictionary of lists (length 1) of gait features.
        """

        # --- 1) Extract raw event arrays ---
        lhs = np.asarray(gait_event_dic['left_down'],   dtype=float)
        ltf = np.asarray(gait_event_dic['left_up'],     dtype=float)
        rhs = np.asarray(gait_event_dic['right_down'],  dtype=float)
        rtf = np.asarray(gait_event_dic['right_up'],    dtype=float)

        # --- 2) Temporal phases (in frames) ---
        
        # Calculate swing times
        if ltf[0] < lhs[0]:
            num_left_swings = min(len(ltf), len(lhs))
            L_swing  = lhs[0:num_left_swings] - ltf[0:num_left_swings]
        else:
            num_left_swings = min(len(ltf), len(lhs) - 1)
            L_swing  = lhs[1:num_left_swings + 1] - ltf[0:num_left_swings]
            
        if rtf[0] < rhs[0]:
            num_right_swings = min(len(rtf), len(rhs))
            R_swing  = rhs[0:num_right_swings] - rtf[0:num_right_swings]
        else:
            num_right_swings = min(len(rtf), len(rhs) - 1)
            R_swing  = rhs[1:num_right_swings + 1] - rtf[0:num_right_swings]

        # Calculate stance times
        if lhs[0] < ltf[0]:
            num_left_stances = min(len(lhs), len(ltf))
            L_stance  = ltf[0:num_left_stances] - lhs[0:num_left_stances]
        else:
            num_left_stances = min(len(lhs), len(ltf) - 1)
            L_stance  = ltf[1:num_left_stances + 1] - lhs[0:num_left_stances]

        if rhs[0] < rtf[0]:
            num_right_stances = min(len(rhs), len(rtf))
            R_stance  = rtf[0:num_right_stances] - rhs[0:num_right_stances]
        else:
            num_right_stances = min(len(rhs), len(rtf) - 1)
            R_stance  = rtf[1:num_right_stances + 1] - rhs[0:num_right_stances]

        # Calculate step times
        if rhs[0] < lhs[0]:
            num_left_steps = min(len(rhs), len(lhs))
            L_steptime = lhs[:num_left_steps] - rhs[:num_left_steps]
        else:
            num_left_steps = min(len(rhs), len(lhs) - 1)
            L_steptime = lhs[1:num_left_steps + 1] - rhs[:num_left_steps] 

        if lhs[0] < rhs[0]:
            num_right_steps = min(len(lhs), len(rhs))
            R_steptime = rhs[:num_right_steps] - lhs[:num_right_steps]
        else:
            num_right_steps = min(len(lhs), len(rhs) - 1)
            R_steptime = rhs[1:num_right_steps + 1] - lhs[:num_right_steps]

        # Calculate double support times
        if lhs[0] < rtf[0]:
            num_d1 = min(len(lhs), len(rtf))
            d1 = rtf[:num_d1] - lhs[:num_d1]
        else:
            num_d1 = min(len(lhs), len(rtf) - 1)
            d1 = rtf[1:1 + num_d1] - lhs[:num_d1]

        if rhs[0] < ltf[0]:
            num_d2 = min(len(rhs), len(ltf))
            d2 = ltf[:num_d2] - rhs[:num_d2]
        else:
            num_d2 = min(len(rhs), len(ltf) - 1)
            d2 = ltf[1:1 + num_d2] - rhs[:num_d2]
        d1_trimmed = d1[:min(len(d1), len(d2))]
        d2_trimmed = d2[:min(len(d1), len(d2))]

        # Combine for overall
        all_swings    = np.concatenate([L_swing, R_swing])
        all_stances   = np.concatenate([L_stance, R_stance])
        all_steptimes = np.concatenate([L_steptime, R_steptime])
        all_double_support = (d1_trimmed + d2_trimmed)[~np.isnan(d1_trimmed + d2_trimmed)]

        # --- 3) Convert to seconds & compute temporal averages ---
        avg_swing_left   = L_swing.mean()    / fps
        avg_swing_right  = R_swing.mean()    / fps
        avg_stance_left  = L_stance.mean()   / fps
        avg_stance_right = R_stance.mean()   / fps
        avg_steptime_left  = L_steptime.mean() / fps
        avg_steptime_right = R_steptime.mean() / fps

        avg_swing    = all_swings.mean()    / fps
        avg_stance   = all_stances.mean()   / fps
        avg_steptime = all_steptimes.mean() / fps
        avg_double   = all_double_support.mean() / fps
        cadence      = 60.0 / avg_steptime

        # --- 4) Spatial metrics from hip Z (reorder, scale to meters, flip Y) ---
        kp = keypoints_3D[:, gait_phase_order_idx] / 1000.0 
        kp[:, :, 1] *= -1.0
        z_hip = kp[:, 0, 2]

        lhs_idx = np.round(lhs).astype(int)
        rhs_idx = np.round(rhs).astype(int)

        # step lengths per side
        if lhs[0] < rhs[0]:
            m = min(len(lhs_idx) - 1, len(rhs_idx))
            sl_left  = np.abs(z_hip[lhs_idx[1:m+1]] - z_hip[rhs_idx[:m]])
            sl_right = np.abs(z_hip[rhs_idx[:m]]    - z_hip[lhs_idx[:m]])
        else:
            m = min(len(lhs_idx), len(rhs_idx) - 1)
            sl_left  = np.abs(z_hip[lhs_idx[:m]]      - z_hip[rhs_idx[:m]])
            sl_right = np.abs(z_hip[rhs_idx[1:m+1]]   - z_hip[lhs_idx[:m]])


        all_step_lengths = np.concatenate([sl_left, sl_right])
        strikes   = np.sort(np.concatenate([lhs_idx, rhs_idx]))
        avg_velocity = np.abs((z_hip[strikes[-1]] - z_hip[strikes[0]]) / (strikes[-1] - strikes[0]) * fps)

        # arm-swing correlation: elbows (keypoint idx 15 vs 12)
        kp_ctr   = kp - kp[:, 0:1]
        arm_corr = np.corrcoef(kp_ctr[:, 15, 2], kp_ctr[:, 12, 2])[0, 1]

        results = {
            "Trial name":                    [""],
            "Average stance time":           float(avg_stance),
            "Average swingtime":             float(avg_swing),
            "Average double support time":   float(avg_double),
            "Average step time":             float(avg_steptime),
            "Average step length":           float(all_step_lengths.mean()),
            "Average velocity":              float(avg_velocity),
            "Average cadence":               float(cadence),
            "Average stance time left":      float(avg_stance_left),
            "Average stance time right":     float(avg_stance_right),
            "Average swing time left":       float(avg_swing_left),
            "Average swing time right":      float(avg_swing_right),
            "Average step time left":        float(avg_steptime_left),
            "Average step time right":       float(avg_steptime_right),
            "Average step length left":      float(sl_left.mean()),
            "Average step length right":     float(sl_right.mean()),
            "Arm swing correlation":         float(arm_corr),
        }
        
        return results
    # ------------------------------------------------------------------
    # --- END: Helper methods ---
    # ------------------------------------------------------------------