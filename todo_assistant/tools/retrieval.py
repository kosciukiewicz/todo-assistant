from typing import Type

from langchain import hub
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.tools import BaseTool
from langchain_community.chat_models import ChatLiteLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.vectorstores import VectorStore
from pydantic.fields import Field
from pydantic.main import BaseModel


class TODORetrievalInput(BaseModel):
    input: str = Field(description="Maximum one sentence, query you want to run with the task name")


class TODORetrievalTool(BaseTool):
    name = "todo_query"
    description = (
        "Useful when you need to answer the question or get information related to single task by"
        " it's name"
    )
    args_schema: Type[BaseModel] = TODORetrievalInput
    retrieval_chain: Runnable

    @classmethod
    def from_llm_and_vectorstore(cls, llm: ChatLiteLLM, vectorstore: VectorStore):
        retriever = vectorstore.as_retriever()
        prompt = hub.pull("rlm/rag-prompt")

        retrieval_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        return TODORetrievalTool(retrieval_chain=retrieval_chain)

    def _run(self, input: str, run_manager: CallbackManagerForToolRun | None = None) -> str:
        s = self.retrieval_chain.invoke(input)
        return s
