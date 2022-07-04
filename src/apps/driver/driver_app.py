import os
import pickle
import tempfile
from pathlib import Path

import streamlit as st

st.set_page_config(layout="wide")
import shutil

import matplotlib.pyplot as plt
import numpy as np

from src.agents.linear_agent import LinearAgent as Agent
from src.constants import DRIVER_PRECOMPUTED_POLICIES_PATH
from src.envs.driver import get_driver_target_velocity
from src.linear.driver_config import (
    ALGORITHM,
    CANDIDATE_POLICY_UPDATE_RATE,
    DIMENSIONALITY,
    IDRL,
    N_PROCESSES,
    NUM_CANDIDATE_POLICIES,
    NUM_QUERY,
    PRIOR_VARIANCE_SCALE,
    QUERY_LOGGING_RATE,
    SEEDS,
    SIMULATION_STEPS,
    THETA_NORM,
    TRAJECTORY_QUERY,
    X_MAX,
    X_MIN,
)
from src.reward_models.logistic_reward_models import (
    LinearLogisticRewardModel,
    LogisticRewardModel,
)
from src.utils import save_video

FILE_PATH = Path(os.path.abspath(__file__))
APP_DIR_PATH = FILE_PATH.parent
CSS_PATH = APP_DIR_PATH / "style.css"
tmp_path = None


def generate_query():
    query_best, true_label, utility, queried_states = st.session_state[
        "agent"
    ].optimize_query(algorithm=algorithm, n_jobs=4)
    st.session_state["queries"] = (queried_states[0], queried_states[1])
    st.session_state["query_best"] = query_best
    st.session_state["true_label"] = true_label

    st.session_state["env"].reset()
    if trajectory_query:
        trajectory_frames_1 = st.session_state["env"].get_trajectory_frames(
            queried_states[0]
        )
        trajectory_frames_2 = st.session_state["env"].get_trajectory_frames(
            queried_states[1]
        )
        save_video(trajectory_frames_1, str(tmp_path / "trajectory_1.mp4"))
        save_video(trajectory_frames_2, str(tmp_path / "trajectory_2.mp4"))
        video_bytes_1 = open(str(tmp_path / "trajectory_1.mp4"), "rb").read()
        video_bytes_2 = open(str(tmp_path / "trajectory_2.mp4"), "rb").read()
        return video_bytes_1, video_bytes_2

    else:
        fig_queries = st.session_state["env"].plot_query_states_pair(
            queried_states[0], queried_states[1], label
        )
    return fig_queries


def get_current_optimal_policy_video():
    theta_hat = st.session_state["agent"].get_parameters_estimate().squeeze()
    policy = st.session_state["env"].get_optimal_policy(theta=theta_hat)
    st.session_state["optimal_policy"] = policy
    frames = st.session_state["env"].get_policy_frames(policy)
    save_video(frames, str(tmp_path / "optimal_policy.mp4"))
    video_bytes = open(str(tmp_path / "optimal_policy.mp4"), "rb").read()
    return video_bytes


def update_agent(label):
    st.session_state["agent"].update_belief(st.session_state["query_best"], label)
    st.session_state["query_count"] += 1
    st.session_state["labeling_disagreement"] += 1 - int(
        label == st.session_state["true_label"]
    )


def get_true_optimal_policy_video():
    if st.session_state["true_optimal_policy_video"] is None:
        policy = st.session_state["env"].get_optimal_policy()
        frames = st.session_state["env"].get_policy_frames(policy)
        save_video(frames, str(tmp_path / "true_optimal_policy.mp4"))
        video_bytes = open(str(tmp_path / "true_optimal_policy.mp4"), "rb").read()
        st.session_state["true_optimal_policy_video"] = video_bytes
    return st.session_state["true_optimal_policy_video"]


def run_app(
    algorithm: str,
    dimensionality: int,
    theta_norm: float,
    x_min: float,
    x_max: float,
    prior_variance_scale: float,
    simulation_steps: int,
    num_candidate_policies: int,
    candidate_policy_update_rate: int,
    query_logging_rate: int,
    num_query: int,
    idrl: bool,
    trajectory_query: bool,
):
    # set the style
    with open(str(CSS_PATH)) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # Set all app states
    if "env" not in st.session_state:
        st.session_state["env"] = get_driver_target_velocity()
        st.info("Environment Initialized")

    if "reward_model" not in st.session_state:
        st.session_state["reward_model"] = LinearLogisticRewardModel(
            dim=dimensionality,
            prior_variance=prior_variance_scale * (theta_norm) ** 2 / 2,
            param_norm=theta_norm,
            x_min=x_min,
            x_max=x_max,
        )
        st.info("Reward Model Initialized")

    if "agent" not in st.session_state:
        st.session_state["agent"] = Agent(
            query_expert=st.session_state["env"].get_comparison_from_feature_diff,
            state_to_features=st.session_state["env"].get_reward_features,
            get_optimal_policy=st.session_state["env"].get_optimal_policy,
            env_step=st.session_state["env"].step,
            env_reset=st.session_state["env"].reset,
            precomputed_policy_path=DRIVER_PRECOMPUTED_POLICIES_PATH / "policies.pkl",
            reward_model=st.session_state["reward_model"],
            num_candidate_policies=num_candidate_policies,
            idrl=idrl,
            candidate_policy_update_rate=candidate_policy_update_rate,
            state_space_dim=dimensionality,
            use_trajectories=trajectory_query,
            num_query=num_query,
        )
        st.info("Agent Initialized")

    if "policies" not in st.session_state:
        with open(f"{str(DRIVER_PRECOMPUTED_POLICIES_PATH)}/policies.pkl", "rb") as f:
            st.session_state["policies"] = pickle.load(f)
        st.info("Precomputed Policies Loaded")
    if "step" not in st.session_state:
        st.session_state["step"] = 0
    if "queries" not in st.session_state:
        st.session_state["queries"] = None
    if "query_best" not in st.session_state:
        st.session_state["query_best"] = None
    if "optimal_policy" not in st.session_state:
        st.session_state["optimal_policy"] = None
    if "query_count" not in st.session_state:
        st.session_state["query_count"] = 0
    if "true_label" not in st.session_state:
        st.session_state["true_label"] = None
    if "labeling_disagreement" not in st.session_state:
        st.session_state["labeling_disagreement"] = 0
    if "true_optimal_policy_video" not in st.session_state:
        st.session_state["true_optimal_policy_video"] = None

    with st.form(key="trajectory labeling"):

        st.header("Trajectory Labeling")
        c_generate_query = st.columns(1)[0]
        done_generate_query = st.columns(1)[0]
        c_trajectory_1, c_trajectory_2 = st.columns(2)
        label = st.radio("Better Trajectory", ["Left", "Right"])
        submit_query = st.form_submit_button("Submit Feedback")

        st.header("Current Optimal Policy")
        done_show_optimal_policy = st.columns(1)[0]
        c_optimal_policy, c_true_optimal_policy = st.columns(2)

        if st.session_state["query_count"] == 0 or submit_query:
            with st.spinner("Generating Query and Optimal Policy..."):

                c_optimal_policy.subheader("Estimated Policy")
                c_true_optimal_policy.subheader("True Optimal Policy")

                current_opt_policy = get_current_optimal_policy_video()
                true_opt_policy = get_true_optimal_policy_video()

                video_1, video_2 = generate_query()

                c_trajectory_1.video(video_1)
                c_trajectory_2.video(video_2)

                c_optimal_policy.video(current_opt_policy)
                c_true_optimal_policy.video(true_opt_policy)

                done_generate_query.success("Done!")
                done_show_optimal_policy.success("Done!")

        if submit_query:
            if label == "Left":
                update_agent(1)
            elif label == "Right":
                update_agent(0)
            else:
                raise ValueError()
            with st.sidebar:
                st.metric("Number of Queries", st.session_state["query_count"])
                st.metric(
                    "Labeling Disagreement", st.session_state["labeling_disagreement"]
                )


if __name__ == "__main__":
    temp_dir = tempfile.TemporaryDirectory()
    tmp_path = Path(temp_dir.name)

    st.title("Information Directed Preference Learning")
    try:
        run_app(
            algorithm=ALGORITHM,
            dimensionality=DIMENSIONALITY,
            theta_norm=THETA_NORM,
            x_min=X_MIN,
            x_max=X_MAX,
            prior_variance_scale=PRIOR_VARIANCE_SCALE,
            simulation_steps=SIMULATION_STEPS,
            candidate_policy_update_rate=CANDIDATE_POLICY_UPDATE_RATE,
            num_candidate_policies=NUM_CANDIDATE_POLICIES,
            query_logging_rate=QUERY_LOGGING_RATE,
            num_query=NUM_QUERY,
            idrl=IDRL,
            trajectory_query=TRAJECTORY_QUERY,
        )
    except:
        temp_dir.cleanup()
