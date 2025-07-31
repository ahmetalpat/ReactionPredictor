import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
from rdkit import Chem
from rdkit.Chem import Draw
from streamlit_ketcher import st_ketcher

# Set page configuration
st.set_page_config(page_title="Chemical Reaction Predictor", layout="wide")

# Function to load the model and tokenizer
@st.cache_resource
def load_model():
    """Loads the T5 model and tokenizer from Hugging Face."""
    model_name = "sagawa/ReactionT5v2-forward-USPTO_MIT"
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    return model, tokenizer

# Function to predict the product
def predict_product(reactants, reagents, model, tokenizer, num_predictions):
    """Predicts the reaction product using the T5 model."""
    input_text = f"reactants>{reactants}.reagents>{reagents}>products>"
    input_ids = tokenizer.encode(input_text, return_tensors='pt')

    # Generate predictions
    outputs = model.generate(
        input_ids,
        max_length=512,
        num_beams=5,
        num_return_sequences=num_predictions,
        early_stopping=True
    )

    # Decode the predictions
    predictions = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return predictions

# Function to display molecules
def display_molecule(smiles_string, legend):
    """Displays a molecule from a SMILES string."""
    mol = Chem.MolFromSmiles(smiles_string)
    if mol:
        img = Draw.MolToImage(mol, size=(300, 300), legend=legend)
        st.image(img, use_column_width='auto')
    else:
        st.warning(f"Could not generate molecule for SMILES: {smiles_string}")

# --- UI Layout ---

# Header
st.title("Chemical Reaction Predictor")
st.markdown("Predict the products of chemical reactions using the `sagawa/ReactionT5v2-forward-USPTO_MIT` model.")

# Load Model
with st.spinner("Loading the prediction model..."):
    model, tokenizer = load_model()

# Sidebar
with st.sidebar:
    st.header("Controls and Information")

    # Example Reactions
    st.subheader("Example Reactions")
    example_reactions = {
        "Esterification": ("CCO.O=C(O)C", "C(C)(=O)O"),
        "Amide Formation": ("CCN.O=C(Cl)C", ""),
        "Suzuki Coupling": ("[B-](C1=CC=CC=C1)(F)(F)F.[K+].CC1=CC=C(Br)C=C1", "c1ccc(B(O)O)cc1"),
    }
    selected_example = st.selectbox("Choose an example:", list(example_reactions.keys()))

    if st.button("Load Example"):
        reactants_smiles_example, reagents_smiles_example = example_reactions[selected_example]
        st.session_state.reactants_smiles = reactants_smiles_example
        st.session_state.reagents_smiles = reagents_smiles_example
        st.session_state.ketcher_reactants = reactants_smiles_example
        st.session_state.ketcher_reagents = reagents_smiles_example


    # Prediction Parameters
    st.subheader("Prediction Parameters")
    num_predictions = st.slider("Number of Predictions to Generate", 1, 5, 1)

    # About Section
    st.subheader("About")
    st.info(
        "This app uses the `sagawa/ReactionT5v2-forward-USPTO_MIT` model to predict chemical reaction products. "
        "Draw or input the SMILES strings for reactants and reagents, then click 'Predict Product'."
    )
    st.markdown("[Model on Hugging Face](https://huggingface.co/sagawa/ReactionT5v2-forward-USPTO_MIT)")


# Main Content
st.header("Input Reactants and Reagents")

# Initialize session state for SMILES
if 'reactants_smiles' not in st.session_state:
    st.session_state.reactants_smiles = ""
if 'reagents_smiles' not in st.session_state:
    st.session_state.reagents_smiles = ""

# Input Tabs
input_tab1, input_tab2 = st.tabs(["Chemical Drawing Tool", "SMILES Text Input"])

with input_tab1:
    st.subheader("Draw Molecules")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Reactants")
        if 'ketcher_reactants' in st.session_state:
            reactant_smiles_from_drawing = st_ketcher(st.session_state.ketcher_reactants, key="ketcher_reactants")
        else:
            reactant_smiles_from_drawing = st_ketcher("", key="ketcher_reactants")


    with col2:
        st.write("Reagents")
        if 'ketcher_reagents' in st.session_state:
             reagent_smiles_from_drawing = st_ketcher(st.session_state.ketcher_reagents, key="ketcher_reagents")
        else:
             reagent_smiles_from_drawing = st_ketcher("", key="ketcher_reagents")


    if reactant_smiles_from_drawing != st.session_state.get('ketcher_reactants_val'):
        st.session_state.reactants_smiles = reactant_smiles_from_drawing
        st.session_state.ketcher_reactants_val = reactant_smiles_from_drawing

    if reagent_smiles_from_drawing != st.session_state.get('ketcher_reagents_val'):
        st.session_state.reagents_smiles = reagent_smiles_from_drawing
        st.session_state.ketcher_reagents_val = reagent_smiles_from_drawing

with input_tab2:
    st.subheader("Enter SMILES Strings")
    reactants_smiles = st.text_input("Reactants SMILES", st.session_state.reactants_smiles, key="reactants_text_input")
    reagents_smiles = st.text_input("Reagents SMILES", st.session_state.reagents_smiles, key="reagents_text_input")
    st.session_state.reactants_smiles = reactants_smiles
    st.session_state.reagents_smiles = reagents_smiles


# Prediction Button
if st.button("Predict Product", type="primary"):
    reactants_to_use = st.session_state.reactants_smiles
    reagents_to_use = st.session_state.reagents_smiles

    if not reactants_to_use:
        st.error("Please provide reactants.")
    else:
        with st.spinner("Predicting reaction..."):
            predictions = predict_product(reactants_to_use, reagents_to_use, model, tokenizer, num_predictions)

            st.header("Predicted Products")
            for i, product_smiles in enumerate(predictions):
                st.subheader(f"Prediction #{i+1}")
                st.code(product_smiles, language="smiles")
                display_molecule(product_smiles, f"Predicted Product {i+1}")