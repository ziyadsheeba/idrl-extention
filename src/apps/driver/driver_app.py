import pickle
import os
from pathlib import Path
import streamlit as st
import matplotlib.pyplot as plt
from src.constants import DRIVER_PRECOMPUTED_POLICIES_PATH
from src.envs.driver import get_driver_target_velocity
import numpy as np
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

TMP_PATH = Path("./tmp")


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
            query_expert=st.session_state["env"].get_comparison_from_feature_diff,
            state_to_features=st.session_state["env"].get_query_features,
            reward_model=st.session_state["reward_model"],
            state_space_dim=dimensionality,
        )
    if "policies" not in st.session_state:
        # load pre-computed policies
        with open(f"{str(DRIVER_PRECOMPUTED_POLICIES_PATH)}/policies.pkl", "rb") as f:
            st.session_state["policies"] = pickle.load(f)
    if "step" not in st.session_state:
        st.session_state["step"] = 0

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

        query_best, label, utility, queried_states = st.session_state[
            "agent"
        ].optimize_query(
            algorithm=algorithm,
            rollout_queries=rollout_queries,
            v=v,
            trajectories=trajectory_query,
        )
        st.session_state["env"].reset()
        if trajectory_query:
            fig_queries = st.session_state["env"].plot_query_trajectory_pair(
                queried_states[0], queried_states[1], label
            )
        else:
            fig_queries = st.session_state["env"].plot_query_states_pair(
                queried_states[0], queried_states[1], label
            )
        st.session_state["agent"].update_belief(query_best, label)
        return fig_queries

    def get_current_optimal_policy_video():
        theta_hat = st.session_state["agent"].get_parameters_estimate().squeeze()
        policy, *_ = st.session_state["env"].get_optimal_policy(theta=theta_hat)
        frames = st.session_state["env"].get_policy_frames(policy)
        save_video(frames, str(TMP_PATH / "optimal_policy.mp4"))
        video_bytes = open(str(TMP_PATH / "optimal_policy.mp4"), "rb").read()
        return video_bytes

    # left, right = st.columns(2)
    # with left:
    c1, c2 = st.columns((1, 1))

    if c1.button("Generate Query"):
        with st.spinner("Generating Query..."):
            fig = generate_query()
            c1.success("Done!")
            c1.pyplot(fig=fig)
    if c2.button("Show Optimal Policy"):
        with st.spinner("Computing Optimal Policy..."):
            c2.video(get_current_optimal_policy_video())
            c2.success("Done!")


if __name__ == "__main__":
    os.makedirs(TMP_PATH, exist_ok=True)
    st.title("Driver Environment Trajectory Comparisons")
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
