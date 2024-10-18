#데이터 프레임 가져오기​
from openai import OpenAI
import streamlit as st
import pandas as pd

# Read the dataset
df = pd.read_csv("./InkjetDB_preprocessing.csv")

# Display the dataframe
# st.write(df)
# st.line_chart(df)

##################################################################################################################


# 챗봇 가져오기


st.title("챗봇과 대화를 해보세요.")

openai_api_key = st.text_input("Enter your OpenAI API key")

client = OpenAI(api_key=openai_api_key)

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4o"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        )
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})
    
####################################################################################################################

# 달력창이 뜨는 수정가능한 테이블 삽입

# df = pd.read_csv("avocado.csv")
# df['date'] = df['date'].apply(lambda x: date.fromisoformat(x))

# new_df = st.data_editor(df,
#                         column_config={
#                             "date":st.column_config
#                         })

# 메트릭 제작
st.metric(label="Temperature", value="70 °F", delta="1.2 °F")