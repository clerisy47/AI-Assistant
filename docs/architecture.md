%% Architecture diagram, Mermaid source.
%% Rendered natively by GitHub/GitLab when embedded in README.md.
%% See architecture.svg for a static, standalone version of the same system.

flowchart TB
    Client["Client<br/>curl · Swagger UI · any HTTP client"]

    subgraph Compose["Docker Compose environment"]
        direction TB

        subgraph FastAPIApp["FastAPI application (app container, :8080)"]
            direction TB
            Routers["API Routers<br/>/chat · /research · /rag/* · /structured/summarize · /health"]

            subgraph Classic["Classic path (unchanged)"]
                Orchestrator["Agent Orchestrator<br/>multi-turn tool loop + tool trace"]
                ChatTools["Tool Registry<br/>calculator · datetime · search_knowledge_base"]
            end

            subgraph ResearchPath["Verified research path"]
                Supervisor["Supervisor<br/>budgets · stop reasons · token rollup"]
                Research["Research Agent<br/>search · load_skill · evidence notes · clarify"]
                Verifier["Verifier Agent<br/>draft + EvidenceNotes only"]
                Skill["skills/verified_research<br/>progressive disclosure"]
                Notes["EvidenceNotes<br/>external structured notes"]
            end

            LLMIface["LLM Provider Interface<br/>strict tools · structured output · usage"]
            RAG["RAG Pipeline<br/>Ingest → Chunk → Embed → Retrieve"]

            Routers --> Orchestrator
            Routers --> Supervisor
            Orchestrator --> ChatTools
            Orchestrator --> LLMIface
            ChatTools --> RAG

            Supervisor --> Research
            Supervisor --> Verifier
            Research --> Skill
            Research --> Notes
            Verifier --> Notes
            Research --> LLMIface
            Verifier --> LLMIface
            Research --> RAG
        end

        Qdrant[("Qdrant<br/>Vector Database")]
        VLLM["vLLM (optional, profile local-llm)<br/>OpenAI-compatible · Llama 3.1 / Mistral"]

        RAG --> Qdrant
        LLMIface --> VLLM
    end

    Anthropic["Anthropic Claude API<br/>(external, cloud)"]
    OpenAI["OpenAI API<br/>(external, cloud)"]

    subgraph MLOpsTrack["MLOps tracking (Track A)"]
        MLflowBox["MLflow params / metrics / step traces"]
        EvidentlyBox["Evidently golden regression"]
        AirflowBox["Airflow DAG / make airflow-dry-run"]
    end

    Client -- HTTP --> Routers
    LLMIface -. cloud mode .-> Anthropic
    LLMIface -. cloud mode .-> OpenAI
    Supervisor --> MLflowBox
    Supervisor --> EvidentlyBox
    AirflowBox --> EvidentlyBox
    AirflowBox --> MLflowBox

    classDef app fill:#eff6ff,stroke:#2563eb,color:#1e3a8a;
    classDef container fill:#f0fdf4,stroke:#16a34a,color:#14532d;
    classDef cloud fill:#fff7ed,stroke:#ea580c,color:#7c2d12;
    classDef client fill:#f8fafc,stroke:#334155,color:#0f172a;
    classDef research fill:#f5f3ff,stroke:#7c3aed,color:#4c1d95;
    classDef mlops fill:#ecfeff,stroke:#0891b2,color:#155e75;

    class Client client;
    class Routers,Orchestrator,ChatTools,LLMIface,RAG app;
    class Supervisor,Research,Verifier,Skill,Notes research;
    class Qdrant,VLLM container;
    class Anthropic,OpenAI cloud;
    class MLflowBox,EvidentlyBox,AirflowBox mlops;
