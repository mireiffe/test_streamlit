import os
from typing import List, Optional, TypedDict

import pandas as pd
from langchain.document_loaders import PyPDFLoader
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import END, StateGraph
from utils import *


class State(TypedDict):
    # 그래프 상태의 속성을 정의합니다.
    # 질문, LLM이 생성한 텍스트, 데이터, 코드를 저장합니다.
    question: str
    generation: str
    # 코드를 여러번 실행할 수 있도록, 생성한 코드와 결과를 저장하는 변수를 str형에서 List[str] 형으로 변경합니다.
    data: List[str]
    code: List[str]


class ExcelPDFChatbot:
    def __init__(
        self,
        df_data: Optional[pd.DataFrame] = None,
        df_description: Optional[str] = None,
        pdf_path: Optional[str] = None,
        pdf_description: Optional[str] = None,
    ) -> None:
        """
        Chatbot을 초기화합니다.

        Args:
            df_data (Optional[pd.DataFrame], optional): 엑셀 데이터 프레임. Defaults to None.
            df_description (Optional[str], optional): 엑셀 데이터 프레임 설명. df_data가 None이 아닐 경우 설명을 반드시 입력해야 합니다. Defaults to None.
            pdf_path (Optional[str], optional): PDF 파일 경로 리스트. Defaults to None.
            pdf_description (Optional[str], optional): PDF 파일 설명 리스트. pdf_path가 None이 아닐 경우, 설명을 반드시 입력해야 합니다. Defaults to None.
        """
        self.llm = ChatOllama(model="mistral:7b")
        self.route_llm = ChatOllama(model="mistral:7b", format="json")
        self.embeddings = OllamaEmbeddings(model="mistral:7b")

        self.df_data = df_data
        self.pdf_path = pdf_path
        self.hallucination_count = 0
        self.hallucination_limit = 3

        # 엑셀 데이터를 불러옵니다.
        if df_data is not None:
            self.df_data = df_data
            self.df_description = df_description
            self.df_columns = ", ".join(self.df_data.columns.tolist())
            if self.df_description is None:
                raise ValueError("Please provide a description for the Excel data.")

        # PDF 데이터를 불러옵니다.
        if pdf_path is not None:
            self.pdf_path = pdf_path
            self.pdf_description = pdf_description
            if self.pdf_description is None:
                raise ValueError("Please provide a description for the PDF data.")
            pdf_name = pdf_path.split(".")[0]
            if os.path.exists(pdf_name):
                self.vectorstore = FAISS.load_local(
                    pdf_name,
                    embeddings=self.embeddings,
                    allow_dangerous_deserialization=True,
                )
            else:
                loader = PyPDFLoader(pdf_path)
                docs = loader.load()
                self.vectorstore = FAISS.from_documents(docs, embedding=self.embeddings)
            self.db_retriever = self.vectorstore.as_retriever()
        # 그래프를 초기화합니다.
        self.graph = StateGraph(State)

        ## 그래프 구성

        # 앞서 정의한 Node를 모두 추가합니다.
        self.graph.add_node("init_answer", self.route_question)

        self.graph.add_node("excel_data", self.query)
        self.graph.add_node("rag", self.retrieval)

        self.graph.add_node("excel_plot", self.plot_graph)
        self.graph.add_node("answer_with_data", self.answer_with_data)
        self.graph.add_node("plain_answer", self.answer)
        self.graph.add_node("answer_with_retrieval", self.answer_with_retrieved_data)

        # 시작지점을 정의합니다.
        self.graph.set_entry_point("init_answer")

        # 간선을 정의합니다.
        # self.graph.add_edge("rag", "answer_with_retrieval")
        # END는 종결 지점을 의미합니다.
        self.graph.add_edge(
            "plain_answer", END
        )  # self.graph.set_finish_point("answer")와 동일합니다.
        self.graph.add_edge("answer_with_data", END)

        # 조건부 간선을 정의합니다.
        # init_answer 노드의 답변을 바탕으로 decide_query 함수에서 query 또는 answer로 분기합니다.
        self.graph.add_conditional_edges(
            "init_answer",
            self._extract_route,
            # 어떤 노드로 이동할지 mapping합니다. 없어도 무방하지만, Graph의 가독성을 높일 수 있습니다.
            {
                "excel_data": "excel_data",
                "rag": "rag",
                "excel_plot": "excel_plot",
                "plain_answer": "plain_answer",
            },
        )

        # 생성한 코드에서 에러가 발생할 경우 다시 코드를 생성하도록 설정하는 조건부 간선입니다.
        self.graph.add_conditional_edges(
            "excel_plot",
            self._check_generated_code_error,
            {"False": END, "True": "excel_plot"},
        )
        self.graph.add_conditional_edges(
            "excel_data",
            self._check_generated_code_error,
            {"False": "answer_with_data", "True": "excel_data"},
        )
        # [지시사항 2-A] Retrieve한 문서가 관련이 없는 것 같다면, 문서 없이 답변을 생성하도록 로직을 수정하세요.
        # Hint. self._judge_relenvance() 메서드를 활용합니다.
        # 이 메서드는 문서가 관련이 있다면 "Relevant", 아니라면 "Irrelevant"를 반환합니다.
        #################################################
        # 아래 간선을 지시사항을 참고하여 조건부 간선으로 바꿉니다.
        # self.graph.add_edge("rag", "answer_with_retrieval")
        self.graph.add_conditional_edges(
            "rag",
            self._judge_relenvance,
            {"Relevant": "answer_with_retrieval", "Irrelevant": "plain_answer"},
        )
        ##############################################

        # [지시사항 2-B] 생성된 답변이 할루시네이션이 의심된다면 답변을 다시 생성하도록 로직을 수정하세요.
        # Hint. self._judge_hallucination() 메서드를 활용합니다.
        # 이 메서드는 할루시네이션이 아니라면 "Relevant", 할루시네이션이라면 "Irrelevant"를 반환합니다.
        # 3회 이상 할루시네이션 판단이 나오면 "No Data"를 반환합니다.
        #################################################
        # 아래 간선을 지시사항을 참고하여 조건부 간선으로 바꿉니다.
        # self.graph.add_edge("answer_with_retrieval", END)
        self.graph.add_conditional_edges(
            "answer_with_retrieval",
            self._judge_hallucination,
            {
                "Relevant": END,
                "Irrelevant": "answer_with_retrieval",
                "NO DATA": "plain_answer",
            },
        )
        #################################################

        self.graph = self.graph.compile()

    def invoke(self, question) -> str:
        self.hallucination_count = 0
        answer = self.graph.invoke({"question": question})
        print("===생성 종료===")

        return answer

    def query(self, state: State):
        """
        데이터를 쿼리하는 코드를 생성하고, 실행하고, 그 결과를 포함한 State를 반환합니다.
        위 과정은 앞서 정의한 `find_data` 함수를 활용합니다.

        Args:
            state (dict): 현재 그래프 상태

        Returns:
            state (dict): 쿼리한 데이터를 포함한 새로운 State
        """
        print("---데이터 쿼리---")  # 현재 상태를 확인하기 위한 Print문
        question = state["question"]

        if self.df_data is None:
            raise ValueError(
                "Please provide Excel data to query while initializing the chatbot."
            )

        system_message = (
            f"당신은 주어진 {self.df_description} 데이터를 분석하는 데이터 분석가입니다.\n"
            f"{self.df_description} 데이터가 저장된 df DataFrame에서 데이터를 출력하여 "
            "주어진 질문에 답할 수 있는 파이썬 코드를 작성하세요.\n"
            f"`df` DataFrame에는 다음과 같은 열이 있습니다: {self.df_columns}\n"
            "데이터는 이미 로드되어 있으므로 데이터 로드 코드를 생략해야 합니다."
        )

        message_with_data_info = [
            ("system", system_message),
            ("human", "{question}"),
        ]

        if isinstance(state["data"], list) and len(state["data"]) > 0:
            # data에 다른 값이 이미 있다는 것은, 이전에 코드를 실행했을 때 에러가 발생했다는 것을 의미합니다.
            # State에 저장된 code와 data를 각각 assistant와 human 역할의 텍스트로 추가합니다.
            # 이때, human 역할의 텍스트에는 에러를 수정하라는 지사사항을 추가합니다.
            for _code, _data in zip(state["code"], state["data"]):
                # Replace with double bracelet
                _code = _code.replace(r"{", r"{{").replace(r"}", r"}}")
                _data = _data.replace(r"{", r"{{").replace(r"}", r"}}")
                message_with_data_info.append(("assistant", _code))
                message_with_data_info.append(("human", f"{_data}, 다시 생성하세요."))
        prompt_with_data_info = ChatPromptTemplate.from_messages(message_with_data_info)

        # 체인을 구성합니다.
        code_generate_chain = (
            {"question": RunnablePassthrough()}
            | prompt_with_data_info
            | self.llm
            | StrOutputParser()
            | python_code_parser
        )
        code = code_generate_chain.invoke(question)
        data = run_code(code, df=self.df_data)
        state["code"].append(code)
        state["data"].append(data)
        return {
            "question": question,
            "code": state["code"],
            "data": state["data"],
            "generation": code,
        }

    def answer_with_data(self, state: State):
        """
        쿼리한 데이터를 바탕으로 답변을 생성합니다.

        Args:
            state (dict): 현재 그래프 상태

        Returns:
            state (dict): LLM의 답변을 포함한 새로운 State
        """
        print("---데이터 기반 답변 생성---")  # 현재 상태를 확인하기 위한 Print문
        question = state["question"]
        data = state["data"][-1]  # 마지막에 생성된 데이터를 사용합니다.

        # 데이터를 바탕으로 질문에 대답하는 코드를 생성합니다.
        reasoning_system_message = (
            "당신은 데이터를 바탕으로 질문에 답하는 데이터 분석가입니다.\n"
        )
        reasoning_system_message += (
            f"사용자가 입력한 데이터를 바탕으로, 질문에 대답하세요."
        )

        reasoning_user_message = "데이터: {data}\n{question}"

        reasoning_with_data = [
            ("system", reasoning_system_message),
            ("human", reasoning_user_message),
        ]
        reasoning_with_data_chain = (
            ChatPromptTemplate.from_messages(reasoning_with_data)
            | self.llm
            | StrOutputParser()
        )

        # 대답 생성
        generation = reasoning_with_data_chain.invoke(
            {"data": data, "question": question}
        )
        return {
            "question": question,
            "code": state["code"],
            "data": state["data"],
            "generation": generation,
        }

    def answer(self, state: State):
        """
        데이터를 쿼리하지 않고 답변을 바로 생성합니다.

        Args:
            state (dict): 현재 그래프 상태

        Returns:
            state (dict): LLM의 답변을 포함한 새로운 State
        """
        print("---답변 생성---")  # 현재 상태를 확인하기 위한 Print문
        question = state["question"]

        return {
            "question": question,
            "generation": self.llm.invoke(question).content,
            "data": [],
            "code": [],
        }

    def plot_graph(self, state: State):
        """
        현재 그래프 상태를 시각화합니다.

        Args:
            state (dict): 현재 그래프 상태

        Returns:
            None
        """

        def change_plot_to_save(code: str) -> str:
            # [TODO] plt.plot()이 code 안에 있는지 확인합니다.
            cond = "plt.plot()" in code
            if cond:
                return code

            # [TODO] plt.plot() 뒤에 plt.savefig('plot.png')을 추가합니다.
            code = code.split("plt.plot()")[0]
            code += "plt.plot()\nplt.savefig('plot.png')"
            return code

        print("---그래프 시각화---")  # 현재 상태를 확인하기 위한 Print문
        question = state["question"]

        # 챗봇이 이미지를 안정적으로 불러올 수 있도록 프롬프트를 개선했습니다.
        # 그래프를 그릴 경우, 반드시 `plt.plot()` 으로 코드를 마무리해야 합니다.
        system_message = (
            f"당신은 주어진 {self.df_description} 데이터를 분석하는 데이터 분석가입니다.\n"
            f"{self.df_description} 데이터가 저장된 df DataFrame에서 데이터를 추출하여 "
            "사용자의 질문에 답할 수 있는 그래프를 그리는 plt.plot()으로 끝나는 코드를 작성하세요. "
            f"`df` DataFrame에는 다음과 같은 열이 있습니다: {self.df_columns}\n"
            "데이터는 이미 로드되어 있으므로 데이터 로드 코드를 생략해야 합니다."
        )

        message_with_data_info = [
            ("system", system_message),
            ("human", "{question}"),
        ]

        if isinstance(state["data"], list) and len(state["data"]) > 0:
            # data에 다른 값이 이미 있다는 것은, 이전에 코드를 실행했을 때 에러가 발생했다는 것을 의미합니다.
            # State에 저장된 code와 data를 각각 assistant와 human 역할의 텍스트로 추가합니다.
            # 이때, human 역할의 텍스트에는 에러를 수정하라는 지사사항을 추가합니다.
            for _code, _data in zip(state["code"], state["data"]):
                # Replace with double bracelet
                _code = _code.replace(r"{", r"{{").replace(r"}", r"}}")
                _data = _data.replace(r"{", r"{{").replace(r"}", r"}}")
                message_with_data_info.append(("assistant", _code))
                message_with_data_info.append(("human", f"{_data}, 다시 생성하세요."))
        prompt_with_data_info = ChatPromptTemplate.from_messages(message_with_data_info)

        # 체인을 구성합니다.
        code_generate_chain = (
            {"question": RunnablePassthrough()}
            | prompt_with_data_info
            | self.llm
            | StrOutputParser()
            | python_code_parser
            | change_plot_to_save  # plt.plot()를 plt.savefig('plot.png')로 변경합니다.
        )
        code = code_generate_chain.invoke(question)
        # 코드를 실행하고, 출력값 혹은 에러 메시지를 반환합니다.
        answer = run_code(code, df=self.df_data)
        # 챗봇이 `plot.png` 파일을 불러오도록 설정합니다.
        data = "plot.png"

        # 에러가 발생했을 경우, data를 에러 메시지로 설정합니다.
        if "Error" in answer:
            data = answer

        state["code"].append(code)
        state["data"].append(data)
        return {
            "question": question,
            "code": state["code"],
            "data": state["data"],
            "generation": code,
        }
        # return {"question": question, "code": code, "data": data, "generation": answer}

    def retrieval(self, state: State):
        """
        데이터 검색을 수행합니다.

        Args:
            state (dict): 현재 그래프 상태

        Returns:
            state (dict): 검색된 데이터를 포함한 새로운 State
        """

        def get_retrieved_text(docs):
            result = "\n".join([doc.page_content for doc in docs])
            return result

        print("---데이터 검색---")  # 현재 상태를 확인하기 위한 Print문
        question = state["question"]

        # Retrieval Chain
        retrieval_chain = self.db_retriever | get_retrieved_text

        data = retrieval_chain.invoke(question)

        return {
            "question": question,
            "data": data,
            "generation": None,
            "code": [],
        }

    def answer_with_retrieved_data(self, state: State):
        """
        검색된 데이터를 바탕으로 답변을 생성합니다.

        Args:
            state (dict): 현재 그래프 상태

        Returns:
            state (dict): LLM의 답변을 포함한 새로운 State
        """
        # role에는 "AI 어시스턴트"가, question에는 "당신을 소개해주세요."가 들어갈 수 있습니다.

        print(
            "---검색된 데이터를 바탕으로 답변 생성---"
        )  # 현재 상태를 확인하기 위한 Print문

        question = state["question"]
        data = state["data"]

        # 2챕터의 프롬프트와 체인을 활용합니다.
        messages_with_contexts = [
            (
                "system",
                "당신은 마케터를 위한 친절한 지원 챗봇입니다. 사용자가 입력하는 정보를 바탕으로 질문에 답하세요.",
            ),
            ("human", "정보: {context}.\n{question}."),
        ]
        prompt_with_context = ChatPromptTemplate.from_messages(messages_with_contexts)

        # 체인 구성
        qa_chain = prompt_with_context | self.llm | StrOutputParser()

        generation = qa_chain.invoke({"context": data, "question": question})
        return {
            "question": question,
            "data": data,
            "generation": generation,
            "code": [],
        }

    def _check_generated_code_error(self, state: State) -> str:
        """
        코드 실행 결과를 분석하여, 코드를 재생성할지 판단합니다.
        코드 실행 결과에 문제가 있는지 판단합니다. 문제가 있다면 "True"를 반환합니다.
        Args:
            code_exec_result (str): 코드 실행 결과

        Returns:
            str: 코드를 재생성할지 여부 ("True" 는 재생성)
        """
        code_exec_result_list = state["data"]
        result = "False"
        if len(code_exec_result_list) > 0:
            print(code_exec_result_list[-1])

        # 코드 실행 결과에 에러가 있는지 판단합니다.
        if len(code_exec_result_list) > 5:
            result = "False"
        elif "Error: " in code_exec_result_list[-1]:
            result = "True"
        return result

    def _judge_relenvance(self, state: State) -> str:
        """
        문서와 질문의 관련성을 평가합니다.
        문서와 질문이 관련이 있다면 "Relevant"를 리턴하고, 그렇지 않다면 "Irrelevant"를 리턴합니다.
        Args:
            state (dict): 현재 그래프 상태

        Returns:
            str: 문서와 질문의 관련성이 있다면 "Relevant"를 리턴하고, 그렇지 않다면 "Irrelevant"를 리턴합니다.
        """
        print("---관련성 판단---")
        system_message = (
            "당신은 사용자의 질문과 근거 문서의 관련성을 평가하는 전문가입니다. \n"
            "다음은 주어진 근거 문서입니다: {documents}\n"
            "사용자의 질문과 근거 문서의 관련성을 판단하여 'yes', 'no' 중 하나로 판단하세요. \n"
            "판단 결과를 `is_relevant` key에 저장한 JSON dictionary 형태로 답변하고, 다른 텍스트나 설명을 추가하지 마세요."
        )
        user_message = "질문: {question}"
        relevance_judge_prompt = ChatPromptTemplate.from_messages(
            [("system", system_message), ("human", user_message)]
        )

        relevance_judge_chain = (
            relevance_judge_prompt | self.route_llm | JsonOutputParser()
        )
        try:
            relevance_judge_result = relevance_judge_chain.invoke(
                {
                    "documents": state["data"],
                    "question": state["question"],
                }
            )["is_relevant"]
        except Exception as e:
            print(e)
            relevance_judge_result = "Irrelevant"

        relevance_judge_result = (
            "Relevant" if relevance_judge_result == "yes" else "Irrelevant"
        )

        return relevance_judge_result

    def _judge_hallucination(self, state: State) -> str:
        """
        문서의 관련성을 평가하고, 문서를 바탕으로 할루시네이션 여부를 평가합니다.
        할루시네이션일 경우 "Irrelevant"를 리턴하고, 그렇지 않다면 "Relevant"를 리턴합니다.
        만약 5번 연속으로 할루시네이션 판단이 나오면 "No Data"를 리턴합니다.
        Args:
            state (dict): 현재 그래프 상태

        Returns:
            str: 답변의 할루시네이션이 아니라면 "Relevant"를 리턴하고, 할루시네이션일 가능성이 높다면 "Irrelevant"를 리턴합니다.
        """
        print("---할루시네이션 판단---")
        # [지시사항 1-A] 할루시네이션 판단을 위한 프롬프트를 작성하세요.
        # Hint. _judge_relenvance 메서드의 구현체를 참고하세요.
        #################################################
        # 아래 raise 코드를 지우고, 코드를 여기에 작성하세요.
        # raise NotImplementedError("지시사항 1-A를 구현해주세요.")
        system_message = (
            "당신은 주어진 답변이 근거 문서에 근거를 두는지 여부를 판단하는 전문가입니다. \n"
            "다음은 주어진 근거 문서입니다: {documents}\n"
            "주어진 답변이 근거를 기반으로 하는지 여부를 'yes', 'no' 중 하나로 판단하세요. \n"
            "판단 결과를 `is_relevant` key에 저장한 JSON dictionary 형태로 답변하고, 다른 텍스트나 설명을 추가하지 마세요."
        )
        user_message = "답변: {answer}"
        hallucination_judge_prompt = ChatPromptTemplate.from_messages(
            [("system", system_message), ("human", user_message)]
        )
        #################################################

        hallucination_judge_chain = (  # 체인을 구성합니다.
            hallucination_judge_prompt | self.route_llm | JsonOutputParser()
        )

        # [지시사항 1-B] 할루시네이션 판단 로직을 작성하세요.
        # Hint. _judge_relenvance 메서드의 구현체를 참고하세요.
        #################################################
        # 아래 raise 코드를 지우고, 정답 코드를 여기에 작성하세요.
        # raise NotImplementedError("지시사항 1-B를 구현해주세요.")
        judge_out = hallucination_judge_chain.invoke(
            {"documents": state["data"], "answer": state["generation"]}
        )["is_relevant"]

        hallucination_judge_result = (
            "Relevant" if judge_out.lower().strip() == "yes" else "Irrelevant"
        )

        #################################################
        if hallucination_judge_result == "Irrelevant":
            self.hallucination_count += 1
            if self.hallucination_count >= self.hallucination_limit:
                hallucination_judge_result = "No Data"

        return hallucination_judge_result

    def _extract_route(self, state: State) -> str:
        """
        라우팅된 질문을 추출합니다.

        Args:
            state (dict): 현재 그래프 상태

        Returns:
            str: 라우팅된 질문
        """
        return state["generation"]

    def route_question(self, state: State):
        """
        질문을 라우팅합니다.

        Args:
            state (dict): 현재 그래프 상태

        Returns:
            state (dict): 라우팅된 질문을 포함한 새로운 State
        """
        print("---질문 라우팅---")
        # 시스템 메시지에 사용 가능한 툴과 각 툴을 사용할 상황을 명시합니다.
        # 수월한 선택을 위해 JSON 형식으로 출력하도록 프롬프트에 지정합니다.
        route_system_message = "당신은 사용자의 질문에 RAG, 엑셀 데이터 중 어떤 것을 활용할 수 있는지 결정하는 전문가입니다."

        usable_tools_list = ["`plain_answer`"]

        if self.df_data is not None:
            route_system_message += f"{self.df_description} 과 관련된 질문이라면 excel_data를 활용하세요. \n"
            route_system_message += (
                f"그래프를 그리는 질문이라면 excel_plot을 활용하세요. \n"
            )
            usable_tools_list.extend(["`excel_data`", "`excel_plot`"])

        if self.pdf_path is not None:
            route_system_message += (
                f"{self.pdf_description} 과 관련된 질문이라면 RAG를 활용하세요. \n"
            )
            usable_tools_list.append("`rag`")

        route_system_message += "그 외의 질문이라면 plain_answer로 충분합니다. \n"

        usable_tools_text = ", ".join(usable_tools_list)

        route_system_message += (
            f"주어진 질문에 맞춰 {usable_tools_text} 중 하나를 선택하세요. \n"
        )
        route_system_message += "답변은 `route` key 하나만 있는 JSON으로 답변하고, 다른 텍스트나 설명을 생성하지 마세요."
        route_user_message = "{question}"
        route_prompt = ChatPromptTemplate.from_messages(
            [("system", route_system_message), ("human", route_user_message)]
        )
        # 로직 선택용 ChatOllama 객체를 생성합니다.
        # 출력 양식을 json으로 명시하고, 같은 질문에 같은 로직을 적용하기 위해 temperature를 0으로 설정합니다.
        router_chain = route_prompt | self.route_llm | JsonOutputParser()
        route = router_chain.invoke({"question": state["question"]})["route"]
        return {
            "question": state["question"],
            "generation": route.lower().strip(),
            "code": [],
            "data": [],
        }
