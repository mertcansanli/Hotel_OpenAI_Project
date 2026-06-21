import streamlit as st

from src.ingest import build_index
from src.rag import answer_question


st.set_page_config(page_title="Hotel Policy Assistant", page_icon="🏨")

st.title("Hotel Policy Assistant")
st.caption("Basic LangChain + Chroma + Streamlit RAG demo")


with st.sidebar:
    st.header("Hotel Knowledge Base")
    st.write("Put `.txt`, `.pdf`, or `.md` hotel policy files inside `data/docs/`.")
    st.write("Then click the button below or run `uv run python -m src.ingest --reset`.")

    if st.button("Re-index hotel documents"):
        with st.status("Indexing hotel documents...", expanded=True):
            try:
                result = build_index(reset=True)

                st.write(f"Documents loaded: {result['document_count']}")
                st.write(f"Chunks stored: {result['chunk_count']}")
                st.write(f"Collection: {result['collection_name']}")
                st.write(f"Persist directory: {result['persist_dir']}")

                st.success("Hotel index is refreshed.")
            except Exception as exc:
                st.error(str(exc))

    st.divider()
    st.write("Example Questions:")
    st.code("Which hotels accept pets?")
    st.code("Does Porto Family Resort allow pets?")
    st.code("Which hotels have free parking?")
    st.code("Which hotels are suitable for children?")


if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me a question about hotel policies."
        }
    ]


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message.get("content", ""))


question = st.chat_input(
    "Ask about pets, children, breakfast, parking, or cancellation policies"
)


if question:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Retrieving relevant hotel policy context..."):
                response = answer_question(question)

            st.markdown(response.answer)

            if response.sources:
                st.caption("Sources: " + ", ".join(response.sources))

            with st.expander("Retrieved context"):
                for i, chunk in enumerate(response.retrieved_chunks, start=1):
                    st.markdown(f"**Chunk {i} — {chunk['source']}**")
                    st.write(chunk["content"])

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response.answer
                }
            )

        except Exception as exc:
            error_message = f"Error: {exc}"
            st.error(error_message)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_message
                }
            )