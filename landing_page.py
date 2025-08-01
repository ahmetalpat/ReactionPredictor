import streamlit as st

def show_landing_page():
    st.set_page_config(page_title="ChemSynth AI", page_icon="🧪", layout="wide")

    # --- Hero Section ---
    st.title("Accelerate Chemical Discovery with AI")
    st.subheader("Go from reactants to products in seconds. Stop guessing, start predicting.")
    st.markdown("---")

    col1, col2 = st.columns([2, 1.5])
    with col1:
        st.write("""
        Our state-of-the-art Transformer model, fine-tuned on the vast USPTO patent dataset,
        achieves **97.5% top-1 accuracy** in predicting chemical reactions.

        Whether you're a student learning organic chemistry or a researcher screening
        synthetic routes, our tool provides instant, reliable results.
        """)
        st.link_button("Start Predicting Now (Free Demo)", "/app", type="primary", use_container_width=True)

    with col2:
        # Using a public URL for an image of a molecule
        st.image("https://media.istockphoto.com/id/1324429294/vector/serotonin-molecule-chemical-structure.jpg?s=612x612&w=0&k=20&c=22dcYd_0F5jT-S5n23gj4_iS-L5c0K_S3K2xS2GZngI=", caption="AI-Powered Synthesis Prediction")

    # --- How It Works ---
    st.header("How It Works in 3 Simple Steps")
    c1, c2, c3 = st.columns(3)
    c1.info("**1. Input Reactants**\n\nDraw molecules or paste SMILES strings into our intuitive interface.")
    c2.info("**2. Click Predict**\n\nOur AI model analyzes the inputs and explores potential reaction pathways.")
    c3.info("**3. Get Products**\n\nInstantly receive the most likely product structures, ready for your analysis.")

    st.markdown("---")
    st.write("A Project by uplabs.ai")

# This allows running the landing page standalone if needed for testing
if __name__ == "__main__":
    show_landing_page()