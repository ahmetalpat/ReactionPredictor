import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rdkit import Chem
from rdkit.Chem import Draw
from streamlit_ketcher import st_ketcher
import torch

# --- Page Configuration ---
st.set_page_config(
    page_title="Chemical Reaction Predictor",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Model Loading ---
# Use st.cache_resource to load the model only once
@st.cache_resource
def load_model():
    """
    Loads the T5 model and tokenizer from Hugging Face.
    Uses AutoModel for better compatibility.
    """
    model_name = "sagawa/ReactionT5v2-forward-USPTO_MIT"
    try:
        # Use Auto* classes for robustness
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        return model, tokenizer
    except Exception as e:
        # Provide more detailed error information
        st.error("An error occurred while loading the model.")
        st.error(f"Error Type: {type(e).__name__}")
        st.error(f"Error Details: {e}")
        # Add a hint about potential memory issues on Hugging Face Spaces
        st.info("Hint: Free tiers on Hugging Face Spaces have limited memory (RAM). "
                "If the app fails to load the model, it might be due to an Out-of-Memory error. "
                "Consider upgrading your Space for more resources.")
        return None, None

# --- Core Functions ---
def predict_product(reactants, reagents, model, tokenizer, num_predictions):
    """Predicts the reaction product using the T5 model."""
    # Format the input string as required by the model
    # Handle the case where reagents might be empty
    if reagents and reagents.strip():
        input_text = f"reactants>{reactants}.reagents>{reagents}>products>"
    else:
        input_text = f"reactants>{reactants}>products>"

    input_ids = tokenizer.encode(input_text, return_tensors='pt')

    # Generate predictions using beam search
    outputs = model.generate(
        input_ids,
        max_length=512,
        num_beams=num_predictions * 2,  # Generate more beams for better diversity
        num_return_sequences=num_predictions,
        early_stopping=True,
    )

    # Decode predictions
    predictions = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return predictions

def display_molecule(smiles_string, legend):
    """Generates and displays a molecule image from a SMILES string."""
    if not smiles_string:
        st.warning("Received an empty SMILES string.")
        return
    mol = Chem.MolFromSmiles(smiles_string)
    if mol:
        try:
            img = Draw.MolToImage(mol, size=(300, 300), legend=legend)
            st.image(img, use_column_width='auto')
        except Exception as e:
            st.warning(f"Could not generate image for SMILES: {smiles_string}. Error: {e}")
    else:
        st.warning(f"Invalid SMILES string provided: {smiles_string}")

# --- Initialize Session State ---
# This ensures that the state is preserved across reruns
if 'reactants' not in st.session_state:
    st.session_state.reactants = "CCO.O=C(O)C"  # Start with a default example
if 'reagents' not in st.session_state:
    st.session_state.reagents = ""

# --- Sidebar UI ---
with st.sidebar:
    st.title("🧪 Reaction Predictor")
    st.markdown("---")
    st.header("Controls and Information")

    # Example Reactions
    example_reactions = {
        "Esterification": ("CCO.O=C(O)C", ""),
        "Amide Formation": ("CCN.O=C(Cl)C", ""),
        "Suzuki Coupling": ("[B-](C1=CC=CC=C1)(F)(F)F.[K+].CC1=CC=C(Br)C=C1", "c1ccc(B(O)O)cc1"),
        "Clear Inputs": ("", "")
    }

    def load_example():
        # Callback to load selected example into session state
        example_key = st.session_state.example_select
        reactants, reagents = example_reactions[example_key]
        st.session_state.reactants = reactants
        st.session_state.reagents = reagents

    st.selectbox(
        "Load an Example Reaction",
        options=list(example_reactions.keys()),
        key="example_select",
        on_change=load_example
    )

    st.markdown("---")
    st.subheader("Prediction Parameters")
    num_predictions = st.slider("Number of Predictions to Generate", 1, 5, 1, help="How many potential products should the model suggest?")
    st.markdown("---")

    st.subheader("About")
    st.info(
        "This app uses the sagawa/ReactionT5v2-forward-USPTO_MIT model to predict chemical reaction products."
    )
    st.markdown("[View Model on Hugging Face](https://huggingface.co/sagawa/ReactionT5v2-forward-USPTO_MIT)")

# --- Main Application UI ---
st.title("Chemical Reaction Predictor")
st.markdown("A tool to predict chemical reactions using a state-of-the-art Transformer model.")

# --- Model Loading and Main Logic ---
with st.spinner("Loading the prediction model... This may take a moment on first startup."):
    model, tokenizer = load_model()

# Only proceed if the model loaded successfully
if model and tokenizer:
    st.success("Model loaded successfully!")

    # Input Section
    st.header("1. Provide Reactants and Reagents")
    input_tab1, input_tab2 = st.tabs(["✍️ Chemical Drawing Tool", "⌨️ SMILES Text Input"])

    with input_tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Reactants")
            # This component's value is now directly tied to the session state
            reactant_smiles_drawing = st_ketcher(st.session_state.reactants, key="ketcher_reactants")
            if reactant_smiles_drawing != st.session_state.reactants:
                st.session_state.reactants = reactant_smiles_drawing
                st.rerun() # Use the modern rerun command

        with col2:
            st.subheader("Reagents (Optional)")
            reagent_smiles_drawing = st_ketcher(st.session_state.reagents, key="ketcher_reagents")
            if reagent_smiles_drawing != st.session_state.reagents:
                st.session_state.reagents = reagent_smiles_drawing
                st.rerun()

    with input_tab2:
        st.subheader("Enter SMILES Strings")
        # Text inputs now also directly update the session state on change
        st.text_input("Reactants SMILES", key="reactant_text", value=st.session_state.reactants, on_change=lambda: setattr(st.session_state, 'reactants', st.session_state.reactant_text))
        st.text_input("Reagents SMILES", key="reagent_text", value=st.session_state.reagents, on_change=lambda: setattr(st.session_state, 'reagents', st.session_state.reagent_text))

    # Display the current state clearly
    st.info(f"**Current Reactants:** `{st.session_state.reactants}`")
    st.info(f"**Current Reagents:** `{st.session_state.reagents or 'None'}`")

    # Prediction Button
    st.header("2. Generate Prediction")
    if st.button("Predict Product", type="primary", use_container_width=True):
        if not st.session_state.reactants or not st.session_state.reactants.strip():
            st.error("Error: Reactants field cannot be empty. Please provide a molecule.")
        else:
            with st.spinner("Running prediction..."):
                predictions = predict_product(
                    st.session_state.reactants,
                    st.session_state.reagents,
                    model,
                    tokenizer,
                    num_predictions
                )
                st.header("3. Predicted Products")
                if not predictions:
                    st.warning("The model did not return any predictions.")
                else:
                    for i, product_smiles in enumerate(predictions):
                        st.subheader(f"Top Prediction #{i + 1}")
                        st.code(product_smiles, language="smiles")
                        display_molecule(product_smiles, f"Predicted Product #{i + 1}")

elif not model or not tokenizer:
    st.error("Application could not start because the model failed to load. Please check the error messages above.")