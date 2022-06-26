import os
import pickle
from pathlib import Path

import streamlit as st

st.set_page_config(layout="wide")
import shutil

import matplotlib.pyplot as plt
import numpy as np

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
from src.linear.experiments_driver import Agent
from src.reward_models.logistic_reward_models import (
    LinearLogisticRewardModel,
    LogisticRewardModel,
)
from src.utils import save_video

FILE_PATH = Path(os.path.abspath(__file__))
APP_DIR_PATH = FILE_PATH.parent
TMP_PATH = APP_DIR_PATH / "tmp"
CSS_PATH = APP_DIR_PATH / "style.css"


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
    if "reward_model" not in st.session_state:
        st.session_state["reward_model"] = LinearLogisticRewardModel(
            dim=dimensionality,
            prior_variance=prior_variance_scale * (theta_norm) ** 2 / 2,
            param_norm=theta_norm,
            x_min=x_min,
            x_max=x_max,
        )
    if "agent" not in st.session_state:
        st.session_state["agent"] = Agent(
            query_expert=st.session_state["env"].get_hard_comparison_from_feature_diff,
            state_to_features=st.session_state["env"].get_query_features,
            reward_model=st.session_state["reward_model"],
            state_space_dim=dimensionality,
        )
    if "policies" not in st.session_state:
        with open(f"{str(DRIVER_PRECOMPUTED_POLICIES_PATH)}/policies.pkl", "rb") as f:
            st.session_state["policies"] = pickle.load(f)
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

    def generate_query():
        if st.session_state["step"] % candidate_policy_update_rate == 0 and idrl:
            print("Computing Candidate Policies")
            print(f"Estimated time: {5.1*num_candidate_policies/60} minutes")

            # sample parameters
            assert num_candidate_policies > 1, "idrl requires more than 1 policy"
            sampled_params = st.session_state["agent"].sample_parameters(
                n_samples=num_candidate_policies
            )

            # get optimal policy wrt to each sampled parameter
            sampled_optimal_policies = []
            for theta in sampled_params:
                policy, *_ = st.session_state["env"].get_optimal_policy(theta=theta)
                sampled_optimal_policies.append(policy)

            # get the mean state visitation difference between policies
            svf_diff_mean, state_support = st.session_state[
                "env"
            ].estimate_pairwise_svf_mean(sampled_optimal_policies)
            features = [
                st.session_state["env"].get_query_features(x.squeeze().tolist())
                for x in state_support
            ]
            features = np.array(features)
            v = features.T @ svf_diff_mean
        else:
            v = None

        if trajectory_query:
            # sample the precomputed policies
            if num_query > len(st.session_state["policies"]):
                raise ValueError(
                    "The number of queries cannot be met. Increase the number of precomputed policies"
                )
            idx = np.random.choice(
                len(st.session_state["policies"]), size=num_query, replace=False
            )
            _policies = [st.session_state["policies"][i] for i in idx]
            rollout_queries = st.session_state["env"].get_queries_from_policies(
                _policies, return_trajectories=True
            )
        else:
            # sample the precomputed policies
            idx = np.random.choice(
                len(st.session_state["policies"]),
                size=num_query // st.session_state["env"].episode_length,
                replace=False,
            )
            _policies = [st.session_state["policies"][i] for i in idx]
            rollout_queries = st.session_state["env"].get_queries_from_policies(
                _policies, return_trajectories=False
            )

        query_best, true_label, utility, queried_states = st.session_state[
            "agent"
        ].optimize_query(
            algorithm=algorithm,
            rollout_queries=rollout_queries,
            v=v,
            trajectories=trajectory_query,
        )
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
            save_video(trajectory_frames_1, str(TMP_PATH / "trajectory_1.mp4"))
            save_video(trajectory_frames_2, str(TMP_PATH / "trajectory_2.mp4"))
            video_bytes_1 = open(str(TMP_PATH / "trajectory_1.mp4"), "rb").read()
            video_bytes_2 = open(str(TMP_PATH / "trajectory_2.mp4"), "rb").read()
            return video_bytes_1, video_bytes_2

        else:
            fig_queries = st.session_state["env"].plot_query_states_pair(
                queried_states[0], queried_states[1], label
            )
        return fig_queries

    def get_current_optimal_policy_video():
        theta_hat = st.session_state["agent"].get_parameters_estimate().squeeze()
        policy, *_ = st.session_state["env"].get_optimal_policy(theta=theta_hat)
        st.session_state["optimal_policy"] = policy
        frames = st.session_state["env"].get_policy_frames(policy)
        save_video(frames, str(TMP_PATH / "optimal_policy.mp4"))
        video_bytes = open(str(TMP_PATH / "optimal_policy.mp4"), "rb").read()
        return video_bytes

    def update_agent(label):
        st.session_state["agent"].update_belief(st.session_state["query_best"], label)
        st.session_state["query_count"] += 1
        st.session_state["labeling_disagreement"] += 1 - int(
            label == st.session_state["true_label"]
        )

    with st.form(key="trajectory labeling"):

        st.header("Trajectory Labeling")
        c_generate_query = st.columns(1)[0]
        done_generate_query = st.columns(1)[0]
        c_trajectory_1, c_trajectory_2 = st.columns((1, 1))
        label = st.radio("Better Trajectory", ["Left", "Right"])
        submit_query = st.form_submit_button("Submit Query")

        st.header("Current Optimal Policy")
        c_show_optimal_policy = st.columns(1)[0]
        done_show_optimal_policy = st.columns(1)[0]
        c_optimal_policy = st.columns(1)[0]

        if st.session_state["query_count"] == 0 or submit_query:
            with st.spinner("Generating Query and Optimal Policy..."):
                c_optimal_policy.video(get_current_optimal_policy_video())
                done_show_optimal_policy.success("Done!")
                video_1, video_2 = generate_query()
                c_trajectory_1.video(video_1)
                c_trajectory_2.video(video_2)
                done_generate_query.success("Done!")
        if submit_query:
            if label == "Left":
                update_agent(1)
            elif label == "Right":
                update_agent(0)
            else:
                raise ValueError()
            with st.sidebar:
                st.metric("Number of Queries", st.session_state["query_count"])
                st.metric("Labeling Disagreement", st.session_state["labeling_disagreement"])


if __name__ == "__main__":
    os.makedirs(TMP_PATH, exist_ok=True)
    st.title("Driver Environment Trajectory Comparisons")
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
        shutil.rmtree(TMP_PATH)
