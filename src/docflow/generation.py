from .backends import LLMBackend
from .progress import maybe_tqdm
from .prompts import PromptTemplate
from .schemas import Document, Response


def generate_response(
    document: Document,
    backend: LLMBackend,
    user_prompt: PromptTemplate,
    system_prompt: PromptTemplate | None = None,
) -> Response:
    """
    Generate one LLM response for one document.
    """
    rendered_system_prompt = system_prompt.render() if system_prompt else None
    rendered_user_prompt = user_prompt.render(document_text=document.text)

    text = backend.complete(
        system_prompt=rendered_system_prompt,
        user_prompt=rendered_user_prompt,
    )

    return Response(
        document_id=document.id,
        text=text,
        backend_name=backend.name,
    )


def generate_responses(
    documents: list[Document],
    backend: LLMBackend,
    user_prompt: PromptTemplate,
    system_prompt: PromptTemplate | None = None,
    progress: bool = False,
) -> list[Response]:
    """
    Generate LLM responses for a collection of documents.
    """
    return [
        generate_response(
            document=document,
            backend=backend,
            user_prompt=user_prompt,
            system_prompt=system_prompt,
        )
        for document in maybe_tqdm(
            documents, enabled=progress, description="Generating responses"
        )
    ]
