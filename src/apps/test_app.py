import streamlit as st


def fetch_data(project_name: str, row_index: int):
    """A simple example function to fetch data."""
    dbs = {
        "Boring Project": [1, 2, 3, 4, 5, 6, 7],
        "Interesting Project": [1, 2, 3, 4, 5, 6, 7],
    }
    return dbs[project_name][row_index]


if __name__ == "__main__":
    st.title("Driver Environment Trajectory Comparisons")

    # set variables in session state
    if "project" not in st.session_state:
        st.session_state.projects = ["Boring Project", "Interesting Project"]
    if "row_index" not in st.session_state:
        st.session_state.row_index = 0

    st.session_state.current_project = st.radio(
        "Select a project to work with:",
        st.session_state.projects,
    )
    # examples to update row index
    if st.button("Next", key="next_data"):
        data = fetch_data(st.session_state.current_project, st.session_state.row_index)
        st.session_state.row_index += 1
        st.text(st.session_state.row_index)

    if st.button("Previous", key="previous_data"):
        data = fetch_data(st.session_state.current_project, st.session_state.row_index)
        st.session_state.row_index -= 1
        st.text(st.session_state.row_index)
