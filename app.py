
import os
import PIL.Image as img
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st

api_key = os.getenv('GOOGLE_API_KEY')

llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash-latest",google_api_key=api_key)

def get_response(image,prompt):
    if prompt=='':
        message = HumanMessage(content=[{"type": "image_url", "image_url": image}])
        response = llm.invoke([message])
        return response.content
    else:
        message = HumanMessage(content=[{"type": "text","text": prompt},
        {"type": "image_url", "image_url": image}])       
        response = llm.invoke([message])
        return response.content

def main():    

    st.set_page_config(page_title="Text Description")

    st.header("Image to Text Application (Google Gemini)")
    input=st.text_input("Input Prompt: ",key="input")
    uploaded_file = st.file_uploader("Upload an image...", 
                                    type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = img.open(uploaded_file) # DIsplay of the Image
        st.image(image, caption="Uploaded Image.", use_column_width=True)
    else:
        st.write("Image Not Uploaded")

    submit=st.button("Run")

    if submit:
        response=get_response(uploaded_file, input)
        st.subheader("The Response is")
        st.write(response)

if __name__=='__main__': 
    main()


