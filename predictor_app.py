import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rdkit import Chem
from rdkit.Chem import Draw
from streamlit_ketcher import st_ketcher

@st.cache_resource
def load_model():
    model_name = "sagawa/ReactionT5v2-forward-USPTO_MIT"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        return model, tokenizer
    except Exception as e:
        st.error(f"Fatal Error: Could not load the prediction model. {e}")
        return None, None

def predict_product(reactants, reagents, model, tokenizer, num_predictions):
    if reagents and reagents.strip():
        input_text = f"reactants>{reactants}.reagents>{reagents}>products>"
    else:
        input_text = f"reactants>{reactants}>products>"
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    outputs = model.generate(
        input_ids, max_length=512, num_beams=num_predictions * 2,
        num_return_sequences=num_predictions, early_stopping=True,
    )
    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

def display_molecule(smiles_string, legend):
    mol = Chem.MolFromSmiles(smiles_string)
    if mol:
        st.image(Draw.MolToImage(mol, size=(350, 350), legend=legend), use_column_width='always')
    else:
        st.warning(f"RDKit could not parse the predicted SMILES: `{smiles_string}`")

def show_predictor_page():
    st.sidebar.success(f"Logged in as: **{st.session_state.username}**")
    if st.sidebar.button("Logout"):
        # Clear session to log out
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    st.title("🧪 Chemical Reaction Predictor")
    st.sidebar.markdown("---")
    num_predictions = st.sidebar.slider("Number of Predictions", 1, 5, 1, help="How many potential products should the model suggest?")

    model, tokenizer = load_model()
    if not model or not tokenizer:
        return

    st.header("1. Provide Reactants & Reagents")
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.reactants_smiles = st_ketcher(st.session_state.get("reactants_smiles", "CCO.O=C(O)C"), key="ketcher_reactants")
    with col2:
        st.session_state.reagents_smiles = st_ketcher(st.session_state.get("reagents_smiles", ""), key="ketcher_reagents")

    if st.button("Predict Product", type="primary", use_container_width=True):
        if not st.session_state.reactants_smiles:
            st.error("Reactants field cannot be empty.")
        else:
            with st.spinner("Running prediction..."):
                predictions = predict_product(
                    st.session_state.reactants_smiles, st.session_state.reagents_smiles,
                    model, tokenizer, num_predictions
                )
                st.header("Predicted Products")
                for i, p in enumerate(predictions):
                    st.subheader(f"Prediction #{i + 1}")
                    st.code(p, language="smiles")
                    display_molecule(p, f"Product #{i+1}")