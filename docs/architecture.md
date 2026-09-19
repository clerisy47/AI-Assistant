%% Architecture diagram, Mermaid source.
%% Rendered natively by GitHub/GitLab when embedded in README.md.
%% See architecture.svg for a static, standalone version of the same system.

flowchart TB
    Client["Client<br/>curl · Swagger UI · any HTTP client"]

    subgraph Compose["Docker Compose environment"]
        direction TB

        subgraph FastAPIApp["FastAPI application (app container, :8080)"]
            direction TB
            Routers["API Routers<br/>/chat · /rag/ingest · /rag/query · /structured/summarize · /health"]
            Orchestrator["Agent Orchestrator<br/>multi-turn tool-calling loop + tool trace"]
            Tools["Tool Registry<br/>calculator · get_current_datetime · search_knowledge_base"]
            LLMIface["LLM Provider Interface<br/>strict tool schemas · native structured output"]
            RAG["RAG Pipeline<br/>Ingestion → Chunker → Local Embedder → Retriever"]

            Routers --> Orchestrator
            Orchestrator --> Tools
            Orchestrator --> LLMIface
            Tools --> RAG
        end

        Qdrant[("Qdrant<br/>Vector Database")]
        VLLM["vLLM (optional, profile local-llm)<br/>OpenAI-compatible server · Llama 3.1 / Mistral · GPU"]

        RAG --> Qdrant
        LLMIface --> VLLM
    end

    Anthropic["Anthropic Claude API<br/>(external, cloud)"]
    OpenAI["OpenAI API<br/>(external, cloud)"]

    Client -- HTTP --> Routers
    LLMIface -. cloud mode .-> Anthropic
    LLMIface -. cloud mode .-> OpenAI

    classDef app fill:#eff6ff,stroke:#2563eb,color:#1e3a8a;
    classDef container fill:#f0fdf4,stroke:#16a34a,color:#14532d;
    classDef cloud fill:#fff7ed,stroke:#ea580c,color:#7c2d12;
    classDef client fill:#f8fafc,stroke:#334155,color:#0f172a;

    class Client client;
    class Routers,Orchestrator,Tools,LLMIface,RAG app;
    class Qdrant,VLLM container;
    class Anthropic,OpenAI cloud;
