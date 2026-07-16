from concurrent.futures import ThreadPoolExecutor, as_completed

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
    max_concurrency: int = 1,
) -> list[Response]:
    """
    Generate LLM responses for a collection of documents.
    """
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be at least 1.")

    if max_concurrency == 1:
        return [
            generate_response(
                document=document,
                backend=backend,
                user_prompt=user_prompt,
                system_prompt=system_prompt,
            )
            for document in maybe_tqdm(
                documents,
                enabled=progress,
                description="Generating responses",
                total=len(documents),
            )
        ]

    responses_by_index: dict[int, Response] = {}

    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        futures = {
            executor.submit(
                generate_response,
                document,
                backend,
                user_prompt,
                system_prompt,
            ): index
            for index, document in enumerate(documents)
        }

        for future in maybe_tqdm(
            as_completed(futures),
            enabled=progress,
            description="Generating responses",
            total=len(documents),
        ):
            index = futures[future]
            responses_by_index[index] = future.result()

    return [responses_by_index[index] for index in range(len(documents))]
