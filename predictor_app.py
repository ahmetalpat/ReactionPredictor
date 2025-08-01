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
        st.session_state.model_loaded = True
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
    # The model can predict multiple products separated by a ".".
    smiles_parts = smiles_string.split('.')
    
    # Keep track if we successfully display any molecule.
    displayed_something = False

    # Process each potential molecule part.
    for i, part in enumerate(smiles_parts):
        clean_part = part.strip()
        if not clean_part:
            continue

        # --- NEW LOGIC: Only attempt to draw fragments that are likely organic products ---
        # A simple heuristic is to check for the presence of a Carbon atom ('c' or 'C').
        if 'c' not in clean_part.lower():
            # You can optionally inform the user about filtered-out parts.
            # st.info(f"Ignoring simple ion/fragment: `{clean_part}`")
            continue

        mol = Chem.MolFromSmiles(clean_part)
        
        # Check if RDKit was able to parse the SMILES part.
        if mol:
            # If successful, generate and display the image.
            st.image(Draw.MolToImage(mol, size=(350, 350), legend=f"{legend}, Part {i+1}"))
            displayed_something = True
        else:
            # If it fails, show a warning for just the bad part.
            st.warning(f"RDKit could not parse the predicted SMILES fragment: `{clean_part}`")
    
    # If after checking all parts, nothing could be displayed, show a general error.
    if not displayed_something:
        st.error("RDKit could not parse any of the predicted organic product SMILES.")

def show_predictor_page():
    # --- SIDEBAR ---\
    st.sidebar.success(f"Logged in as: **{st.session_state.username}**")
    if st.sidebar.button("Logout"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.header("Controls and Information")
    
    # --- Example Reactions ---\
    example_reactions = {
        "Esterification": ("CCO.O=C(O)C", ""),
        "Amide Formation": ("CCN.O=C(Cl)C", ""),
        "Suzuki Coupling": ("[B-](C1=CC=CC=C1)(F)(F)F.[K+].CC1=CC=C(Br)C=C1", "c1ccc(B(O)O)cc1"),
        "Diels-Alder": ("C=CC=C.C=C", ""),
        "Grignard": ("C=O.C[Mg]Br", ""),
        "Jones Oxidation": ("c1ccccc1CC(O)C", "O=Cr(=O)(O)O"),
        "Clear Inputs": ("", "")
    }

    def load_example():
        example_key = st.session_state.example_select
        if example_key in example_reactions:
            reactants, reagents = example_reactions[example_key]
            st.session_state.reactants_smiles = reactants
            st.session_state.reagents_smiles = reagents

    st.sidebar.selectbox(
        "Load an Example Reaction",
        options=list(example_reactions.keys()),
        key="example_select",
        on_change=load_example,
        index=0
    )

    # --- Prediction Parameters ---\
    st.sidebar.markdown("---")
    st.sidebar.subheader("Prediction Parameters")
    num_predictions = st.sidebar.slider("Number of Predictions to Generate", 1, 5, 1)

    # --- MAIN PAGE ---\
    st.title("🧪 Chemical Reaction Predictor")
    st.markdown("A tool to predict chemical reactions using a state-of-the-art Transformer model.")
    
    # --- NEW, IMPROVED LOADING LOGIC ---
    # Use a spinner to show a loading message while the model is being loaded.
    # The @st.cache_resource decorator ensures load_model() only runs once.
    with st.spinner("Model is loading... This may take a moment on the first run."):
        model, tokenizer = load_model()

    if not model or not tokenizer:
        st.error("Model failed to load. The application cannot continue.")
        st.stop()
    
    st.success("Model loaded successfully!")
    
    # Initialize session state keys if they don't exist
    if "reactants_smiles" not in st.session_state:
        st.session_state.reactants_smiles = "CCO.O=C(O)C"
    if "reagents_smiles" not in st.session_state:
        st.session_state.reagents_smiles = ""

    # --- Input Tabs ---\
    st.header("1. Provide Reactants and Reagents")
    input_tab1, input_tab2 = st.tabs(["✍️ Chemical Drawing Tool", "⌨️ SMILES Text Input"])

    with input_tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Reactants")
            reactants_drawing = st_ketcher(st.session_state.reactants_smiles, key="ketcher_reactants")
            if reactants_drawing != st.session_state.reactants_smiles:
                st.session_state.reactants_smiles = reactants_drawing
                st.rerun()
        with col2:
            st.subheader("Reagents (Optional)")
            reagents_drawing = st_ketcher(st.session_state.reagents_smiles, key="ketcher_reagents")
            if reagents_drawing != st.session_state.reagents_smiles:
                st.session_state.reagents_smiles = reagents_drawing
                st.rerun()

    with input_tab2:
        st.subheader("Enter SMILES Strings")
        reactants_text = st.text_input("Reactants SMILES", st.session_state.reactants_smiles)
        if reactants_text != st.session_state.reactants_smiles:
            st.session_state.reactants_smiles = reactants_text
            st.rerun()
            
        reagents_text = st.text_input("Reagents SMILES", st.session_state.reagents_smiles)
        if reagents_text != st.session_state.reagents_smiles:
            st.session_state.reagents_smiles = reagents_text
            st.rerun()
    
    # --- Prediction ---\
    st.header("2. Generate Prediction")
    if st.button("Predict Product", type="primary", use_container_width=True):
        if not st.session_state.reactants_smiles:
            st.error("Reactants field cannot be empty.")
        else:
            with st.spinner("Running prediction..."):
                predictions = predict_product(
                    st.session_state.reactants_smiles, st.session_state.reagents_smiles,
                    model, tokenizer, num_predictions
                )
                st.session_state.predictions = predictions # Save predictions to session state
    
    # --- Display Results ---
    if 'predictions' in st.session_state:
        st.header("Predicted Products")
        for i, p in enumerate(st.session_state.predictions):
            st.subheader(f"Prediction #{i + 1}")
            st.code(p, language="smiles")
            display_molecule(p, f"Product #{i+1}")