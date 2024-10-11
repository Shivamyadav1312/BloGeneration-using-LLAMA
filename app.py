import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers  # Updated import


## Function to get response from LLama 2 model
def getLLamaresponse(input_text, no_words, blog_style):
    ### Correct LLama2 model initialization
    llm = CTransformers(model='models/llama-2-7b-chat.ggmlv3.q8_0.bin',
                        model_type='llama',
                        config={'max_new_tokens': 256, 'temperature': 0.01})
    
    # Check if model loads properly
    print("Model Loaded Successfully")
    
    ## Print arguments to verify their values
    print(f"Input text: {input_text}")
    print(f"Number of words: {no_words}")
    print(f"Blog style: {blog_style}")
    
    ## Prompt Template - use the template with placeholders
    template = """
        Write a blog for {blog_style} job profile on the topic {input_text}
        within {no_words} words.
    """
    
    prompt = PromptTemplate(
        input_variables=["blog_style", "input_text", "no_words"],
        template=template
    )
    
    # Format the prompt using the values for blog_style, input_text, and no_words
    formatted_prompt = prompt.format(blog_style=blog_style, input_text=input_text, no_words=no_words)

    print(f"Formatted Prompt: {formatted_prompt}")

    ## Generate the response using the `invoke()` method
    response = llm.invoke(formatted_prompt)
    
    print(f"Generated response: {response}")
    return response


# Streamlit app setup
st.set_page_config(page_title="Generate Blogs",
                   page_icon='🤖',
                   layout='centered',
                   initial_sidebar_state='collapsed')

st.header("Generate Blogs 🤖")

input_text = st.text_input("Enter the Blog Topic")

## Creating two more columns for additional fields
col1, col2 = st.columns([5, 5])

with col1:
    no_words = st.text_input('Number of Words')
with col2:
    blog_style = st.selectbox('Writing the blog for', ('Researchers', 'Data Scientist', 'Common People'), index=0)

submit = st.button("Generate")

## Final response
if submit:
    st.write(getLLamaresponse(input_text, no_words, blog_style))
