import streamlit as st
import joblib
import html

from feature_labels import format_feature_name


ARTIFACT_PATH = "artifacts/readmission_model_bundle.joblib"
TEST_DATA_PATH = "data/test/held_out_encounters.joblib"


@st.cache_resource
def load_model():
    artifacts = joblib.load(ARTIFACT_PATH)
    return artifacts["logistic_model"]


@st.cache_data
def load_test_encounters():
    test_data = joblib.load(TEST_DATA_PATH)
    return test_data["X_test"], test_data["y_test"]


def get_prediction(model, patient):

    prediction = model.predict(patient)[0]
    probability = model.predict_proba(patient)[0][1]

    return prediction, probability


def get_contributions(model, patient):

    contributions = patient.iloc[0] * model.coef_[0]

    risk_increasing = (
        contributions[contributions > 0]
        .sort_values(ascending=False)
    )

    risk_decreasing = (
        contributions[contributions < 0]
        .sort_values(ascending=True)
    )

    return risk_increasing, risk_decreasing


def display_contributors(contributors, count):

    rows = []

    for feature, contribution in contributors.head(count).items():

        readable_name = html.escape(
            format_feature_name(feature)
        )

        rows.append(
            f"<tr>"
            f"<td class='factor-name'>{readable_name}</td>"
            f"<td class='factor-value'>{contribution:+.3f}</td>"
            f"</tr>"
        )

    table_html = (
        "<table class='contributor-table'>"
        "<thead>"
        "<tr>"
        "<th>Factor</th>"
        "<th class='factor-value'>Contribution</th>"
        "</tr>"
        "</thead>"
        "<tbody>"
        + "".join(rows)
        + "</tbody>"
        "</table>"
    )

    st.markdown(
        table_html,
        unsafe_allow_html=True
    )


def main():

    st.set_page_config(
        page_title="Hospital Readmission Risk",
        layout="wide"
    )

    st.markdown(
        """
        <style>
            .block-container {
                max-width: 96% !important;
                padding-top: 3.5rem !important;
                padding-left: 1.25rem !important;
                padding-right: 1.25rem !important;
                padding-bottom: 1rem !important;
            }

            [data-testid="stVerticalBlock"] {
                gap: 0.4rem !important;
            }

            h1 {
                font-size: 1.55rem !important;
                line-height: 1.2 !important;
                margin: 0 0 0.5rem 0 !important;
                padding: 0 !important;
            }

            h2 {
                font-size: 1.1rem !important;
                line-height: 1.2 !important;
                margin: 0.4rem 0 0.25rem 0 !important;
                padding: 0 !important;
            }

            h3 {
                font-size: 1rem !important;
                line-height: 1.2 !important;
                margin-top: 0.25rem !important;
                margin-bottom: 0.6rem !important;
                padding: 0 !important;
            }

            p {
                margin-top: 0.1rem !important;
                margin-bottom: 0.2rem !important;
            }

            .risk-label {
                font-size: 0.85rem;
                opacity: 0.75;
            }

            .risk-value,
            .outcome {
                font-size: 1.3rem;
                font-weight: 600;
                line-height: 1.2;
                margin: 0 !important;
            }

            /* Major separation between patient summary and factors */
            .factors-spacer {
                height: 1.75rem;
            }

            /* Contributor tables */
            .contributor-table {
                width: 100%;
                border-collapse: collapse;
                table-layout: fixed;
                font-size: 0.9rem;
                line-height: 1.25;
            }

            .contributor-table th,
            .contributor-table td {
                padding: 0.45rem 0.55rem;
                border-bottom: 1px solid rgba(128, 128, 128, 0.25);
                vertical-align: top;
            }

            .contributor-table th {
                text-align: left;
                font-weight: 600;
                opacity: 0.75;
            }

            /* Factor receives all remaining table width */
            .factor-name {
                width: auto;
                white-space: normal;
                overflow-wrap: break-word;
                word-break: normal;
                text-align: left;
            }

            /* Contribution remains visible at fixed width */
            .contributor-table .factor-value {
                width: 7rem;
                min-width: 7rem;
                white-space: nowrap;
                text-align: right;
                font-variant-numeric: tabular-nums;
            }

            [data-testid="stSelectbox"] {
                max-width: 18rem;
            }

            .stButton button {
                padding: 0.25rem 0.7rem !important;
                min-height: 2rem !important;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("Hospital Readmission Risk Prediction")

    model = load_model()
    X_test, y_test = load_test_encounters()

    patient_col, risk_col, outcome_col = st.columns(
        [1.2, 0.8, 1.5],
        vertical_alignment="bottom"
    )

    with patient_col:

        patient_index = st.selectbox(
            "Patient encounter",
            options=range(len(X_test)),
            index=None,
            placeholder="Select patient"
        )

    # Do not display prediction information until a patient is selected
    if patient_index is None:
        return

    patient = X_test.iloc[[patient_index]]
    actual = y_test.iloc[patient_index]

    prediction, probability = get_prediction(
        model,
        patient
    )

    risk_increasing, risk_decreasing = get_contributions(
        model,
        patient
    )

    # Reset expanded factor lists when changing patients
    if st.session_state.get("patient_index") != patient_index:
        st.session_state.patient_index = patient_index
        st.session_state.risk_count = 5
        st.session_state.decreasing_count = 5

    with risk_col:

        st.markdown(
            f"""
            <div class="risk-label">
                30-Day Readmission Risk
            </div>
            <div class="risk-value">
                {probability * 100:.1f}%
            </div>
            """,
            unsafe_allow_html=True
        )

    with outcome_col:

        outcome = (
            "Readmitted within 30 days"
            if actual == 1
            else "Not readmitted within 30 days"
        )

        st.markdown(
            f"""
            <div class="risk-label">
                Observed Outcome
            </div>
            <div class="outcome">
                {outcome}
            </div>
            """,
            unsafe_allow_html=True
        )

    # Separate prediction summary from explanation tables
    st.markdown(
        '<div class="factors-spacer"></div>',
        unsafe_allow_html=True
    )

    risk_column, decreasing_column = st.columns(
        2,
        gap="medium"
    )

    with risk_column:

        st.markdown("### Risk Increasing Factors")

        display_contributors(
            risk_increasing,
            st.session_state.risk_count
        )

        if st.session_state.risk_count < len(risk_increasing):

            if st.button(
                "Show 5 more",
                key="increase_risk"
            ):
                st.session_state.risk_count += 5
                st.rerun()

    with decreasing_column:

        st.markdown("### Risk Decreasing Factors")

        display_contributors(
            risk_decreasing,
            st.session_state.decreasing_count
        )

        if st.session_state.decreasing_count < len(risk_decreasing):

            if st.button(
                "Show 5 more",
                key="decrease_risk"
            ):
                st.session_state.decreasing_count += 5
                st.rerun()


if __name__ == "__main__":
    main()