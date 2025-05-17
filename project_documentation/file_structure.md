LangChain Expression Language (LCEL) + LangChain Core Mono File Structure

backend/
│
├── src/
│   ├── main.py                  # FastAPI app entry point
│   ├── config.py                # Config/env loader
│   ├── dependencies/
│   │   └── auth.py              # Supabase JWT auth dependency
│   │
│   ├── routers/                 # Route entry points
│   │   ├── conversation.py      # conversation related routes
|   |   ├── messages.py          # for messages routes
│   │   ├── upload.py            # Document/image upload endpoints
│   │   └── user.py              # User info, metadata if needed
│   │
│   ├── services/                # Core logic handlers (tools)
│   │   ├── ocr.py               # OCR service (Cloud Vision, etc.)
│   │   ├── translate.py         # Translation logic
│   │   ├── extract.py           # Field extraction using templates
│   │   ├── warp.py              # Image enhancement or preprocessing
│   │   └── agent_tools.py       # Tool descriptions and callable mappings
│   │
│   ├── llm/                     # LLM API client + step runner
│   │   ├── gemini_client.py     # Gemini Pro API wrapper
│   │   └── orchestrator.py      # Agent step logic and handler call manager
│   │
│   ├── models/                  # Pydantic schemas
│   │   ├── tool_schema.py       # Tool input/output schema
│   │   ├── user.py              # Auth/user related schemas
│   │   └── document.py          # Document data schemas
│   │
│   ├── utils/                   # Utilities (cloud storage, parsing, etc.)
│   │   ├── storage.py           # Upload/download to Supabase Storage
│   │   ├── supabase.py          # Optional REST/RPC helpers to Supabase
│   │   └── logger.py            # Logging helper
│   │
│   └── constants.py             # Global constants (e.g. template mappings)
│
├── .env                         # Environment variables
├── requirements.txt             # Python dependencies
└── README.md

# 📄 Agent-Driven Document Assistant

A full-stack, chat-first platform that **ingests any PDF / Word / image, extracts key fields, asks the user only for missing data, translates or rewrites specific passages on demand, and returns a clean, versioned document**—all orchestrated by an agentic LLM (Gemini Pro) running through **LangChain Core + LCEL**.




