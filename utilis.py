# utils.py
from vertexai.generative_models import GenerativeModel, Tool
import vertexai.preview.generative_models as generative_models

def get_tools_and_model():
    tools = [
        Tool.from_google_search_retrieval(
            google_search_retrieval=generative_models.grounding.GoogleSearchRetrieval()
        )
    ]

    model = GenerativeModel(
        model_name="gemini-1.5-flash",
        tools=tools,
    )

    return model, tools
