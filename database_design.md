# Wally-Chat Database Schema Documentation

This document outlines the complete database schema for the Wally-Chat application, including all tables, relationships, policies, and helper functions. The schema is designed to support a conversational AI application with rich features including user profiles, conversations, message storage, file handling, document versioning, templates, and RAG (Retrieval-Augmented Generation) capabilities.

## Core Components

### 1. User Management

**profiles**: Stores user profile information linked to Supabase auth users
- Includes id, full_name, avatar_url, and created_at
- Automatic profile creation via the handle_new_user() trigger function

### 2. Conversation System

**conversations**: Tracks user conversations
- Fields: id, profile_id, title, summary, is_active, created_at, updated_at
- Auto-updates updated_at when new messages are added via the touch_conversation() trigger

**messages**: Stores individual messages within conversations
- Fields: id, conversation_id, sender, kind, body, created_at
- sender can be: 'user', 'assistant', 'system', or 'tools'
- kind can be: 'text', 'file', 'action', 'file_card', 'buttons', or 'inputs'
- The body field contains plain text or JSON payloads depending on message kind

### 3. File Management

**file_objects**: Registry for all files in the system
- Fields: id, profile_id, bucket, object_key, mime_type, size_bytes, created_at
- Buckets include 'user_uploads' and 'documents'

**doc_versions**: Version history for editable documents
- Fields: id, base_file_id, rev, placeholder_json, llm_log, created_at
- Tracks document revisions with metadata

### 4. Templates System

**templates**: Catalogue of document templates
- Fields: id, doc_type, variation, object_key, placeholder_json, created_at
- Public read access (no RLS)

### 5. RAG Memory

**rag_memory**: Stores vector embeddings for semantic search
- Fields: id, conversation_id, content, embedding, meta, created_at
- Uses pgvector extension with 768-dimensional vectors (reduced from 1536)
- Includes IVFFlat index for fast cosine similarity search

## Helper Functions

- **add_message()**: Adds a message and its embedding to RAG memory
- **post_buttons()**: Helper to create button-type messages
- **post_inputs()**: Helper to create input-form messages
- **touch_conversation()**: Updates conversation timestamp when messages are added
- **handle_new_user()**: Creates profile entries for new auth users

## Views

- **v_conversation_latest_doc**: Returns the latest version of each document

## Security Model

The database implements Row Level Security (RLS) to ensure users can only access their own data:

- **profiles**: Users can only see their own profile
- **conversations**: Users can only see conversations they created
- **messages**: Users can only see messages in their own conversations
- **file_objects**: Users can only see files they uploaded
- **doc_versions**: Users can only see versions of documents they own
- **rag_memory**: Users can only access RAG data for their own conversations

Special policies:
- **assistant_write**: Allows the assistant to insert messages

## Schema Evolution

The schema has evolved through multiple iterations:

1. Initial setup with core tables and extensions
2. Expanded message types to include 'file_card', 'buttons', and 'inputs'
3. Added helper functions for creating UI components (buttons and inputs)
4. Modified sender types to include 'tool'/'tools' and 'system'
5. Optimized RAG memory by reducing vector dimensions from 1536 to 768

## Technical Details

- Uses pgvector extension for vector embeddings
- Uses pgcrypto for UUID generation
- Implements triggers for automated tasks
- Includes comprehensive Row Level Security
- Uses JSON/JSONB for flexible data structures

## Database Relationships

```
auth.users 1──→ N profiles 1──→ N conversations 1──→ N messages
                      │                                   ↑
                      │                                   │
                      └──→ N file_objects 1──→ N doc_versions
                                               
conversations 1──→ N rag_memory
```

This schema supports a sophisticated chatbot application with document handling capabilities, versioning, and semantic memory features.