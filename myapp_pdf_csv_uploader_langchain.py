import os
from streamlit_option_menu import option_menu
from streamlit_extras.no_default_selectbox import selectbox
from openai import OpenAI
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain_community.document_loaders import CSVLoader
import tempfile

os.environ["OPENAI_API_KEY"] = "Please provide key here"
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


if 'prompts' not in st.session_state:
    st.session_state.prompts = []
if 'responses' not in st.session_state:
    st.session_state.responses = []

global target_lang
global container
global response_container

def translate_text(text, source_language, target_language):
    prompt = f"Translate the following '{source_language}' text to '{target_language}': {text}"
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant that translates text."},
            {"role": "user", "content": prompt}
        ],
        model="gpt-3.5-turbo",
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.5
    )
    translation = response.choices[0].message.content.strip()
    return translation

def acquire_knowledge_base(pdf):
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    # split into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=100,
        chunk_overlap=20,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    # create embeddings
    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_texts(chunks, embeddings)
    return knowledge_base
    #return knowledge_base.as_retriever()

def acquire_knowledge_base_csv(csv_file):
    # use tempfile because CSVLoader only accepts a file_path
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(csv_file.getvalue())
        tmp_file_path = tmp_file.name
    loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={'delimiter': ','})
    data = loader.load()
    embeddings_csv = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(data, embeddings_csv)
    return vectorstore

def search_kb(knowledge_base, prompt):
    docs = knowledge_base.similarity_search(prompt)
    return docs[0].page_content

def search_kb_csv(knowledge_base, prompt):
    docs = knowledge_base.similarity_search(prompt)
    return docs[0].page_content

def conversational_chat(chain, query):
    result = chain({"question": query,
                    "chat_history": st.session_state['history']})
    st.session_state['history'].append((query, result["answer"]))
    return result["answer"]

def init_session_state(csv_file):
    if 'history' not in st.session_state:
        st.session_state['history'] = []
    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello ! Ask me anything about " + csv_file.name + " 🤗"]
    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey ! 👋"]

def send_click_csv(csv_file,prompt):
    if st.session_state.user != '':
        prompt = st.session_state.user
    if prompt:
        # extract the text
        if csv_file is not None:
            knowledge_base_csv = acquire_knowledge_base_csv(csv_file)
            #chain = ConversationalRetrievalChain.from_llm(llm=ChatOpenAI(temperature=0.0, model_name='gpt-3.5-turbo'),retriever=knowledge_base_csv.as_retriever())
            #response_csv=conversational_chat(chain, prompt)
            response_csv=search_kb_csv(knowledge_base_csv,prompt)
            st.write(response_csv)
            st.session_state['past'].append(prompt)
            st.session_state['generated'].append(response_csv)

def send_click(pdf,prompt):
    if st.session_state.user != '':
        prompt = st.session_state.user
    if prompt:
        # extract the text
        if pdf is not None:
            knowledge_base = acquire_knowledge_base(pdf)
            chain = ConversationalRetrievalChain.from_llm(llm=ChatOpenAI(temperature=0.0, model_name='gpt-3.5-turbo'),retriever=knowledge_base.as_retriever())
            response_pdf=conversational_chat(chain, prompt)
            st.write(response_pdf)
            st.session_state['past'].append(prompt)
            st.session_state['generated'].append(response_pdf)
            st.write("Translating to - "+ target_lang)
            source_language = "English"
            target_language = target_lang
            translated_text = translate_text(response_pdf, source_language, target_language)
            st.write(translated_text)
            #docs = knowledge_base.invoke(prompt)
            #st.write(docs[0].page_content)
            #st.write(search_kb(knowledge_base, prompt))
    # llm = OpenAI()
    # chain = load_qa_chain(llm, chain_type="stuff")
    # response=chain.run(input_documents=docs, question=prompt)
    #        with get_openai_callback() as cb:
    #              response = chain.run(input_documents=docs, question=prompt)
    # st.write(response)
    # st.session_state.prompts.append(prompt)
    # st.session_state.responses.append(response)


# Set page configuration
st.set_page_config(page_title="MyApps",
                   layout="wide",
                   page_icon="🧑‍⚕️")

# sidebar for navigation
with st.sidebar:
    selected = option_menu('My Applications',

                           ['File Uploader',
                            'TBD'
                            ],
                           menu_icon='hospital-fill',
                           icons=['activity', 'heart'],
                           default_index=0)

if selected == 'File Uploader':
    # page title
    st.title('My File Uploader Utility')

    # getting the input data from the user
    col1, col2 = st.columns(2)

    with col1:
        options = selectbox(
            "Choose file type",
            ["pdf", "csv"])

    st.write("You selected:", options)

    with col2:
        if  (options == 'pdf'):
            pdf = col2.file_uploader("Upload your PDF File", type="pdf")
        elif (options == 'csv'):
            csv_file = col2.file_uploader("Upload your CSV File", type="csv")

    with col1:
        prompt = st.text_input("Enter your question here:",key="user")
        options_lang = selectbox(
            "Choose the target language",
            ["English", "Spanish", "French"])
        target_lang = options_lang

    with col2:
        if (options == 'pdf'):
            context_search_button = st.button('Search', on_click=send_click(pdf,prompt))
        elif (options == 'csv'):
            context_search_button = st.button('Search', on_click=send_click_csv(csv_file, prompt))
        # show user input
        # st.text_input("Ask a question about your PDF:", key="user")
        # st.button("Send", on_click=send_click)
    if 'history' not in st.session_state:
        st.session_state['history'] = []
    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello ! Ask me anything about provided file "+" 🤗"]
    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey ! 👋"]
    response_container = st.container()
    # container for the user's text input
    container = st.container()

    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")