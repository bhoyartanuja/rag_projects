# System Design — Enterprise RAG System 🏦

## The Question
"Design a production RAG system for a large bank. Employees ask questions about internal policy documents. 50,000 documents, 5,000 concurrent users, answers in under 3 seconds, fully auditable for compliance."

---

## Step 1 — Clarifying Questions (always ask first)
Never jump to design. Ask:
- Is batch processing acceptable or do we need real time?
- On premise or cloud? Which provider?
- What accuracy requirements? Any compliance constraints?
- Role based access — who sees which documents?
- What languages are documents and responses in?
- What happens when answer is not found?
- Where are documents currently stored?
- Any existing pipelines or infrastructure?

**Rule:** Interviewers are testing if you think before you build. Spend 2-3 minutes here.

---

## Step 2 — Requirements Summary
| Requirement | Answer |
|---|---|
| Documents | 50,000, updated weekly |
| Users | 5,000 concurrent |
| Latency | Under 3 seconds |
| Cloud | GCP |
| Access control | Department based (HR sees HR, Legal sees Legal) |
| Language | English only |
| Not found | Return "I don't know" — never hallucinate |
| Auditability | Every query logged with sources |

---

## Step 3 — High Level Design
```
INGESTION PIPELINE
Google Drive → Dataflow → Embed → Pinecone (with metadata)

SERVING LAYER
User → Load Balancer → Cloud Run Flask → Redis Cache → Pinecone → Gemini → BigQuery Audit Log → User
```

---

## Step 4 — Deep Dive: Ingestion Pipeline

### Flow:
```
Google Drive
     ↓
Cloud Scheduler (weekly trigger)
     ↓
Dataflow Pipeline
     ↓
Check metadata DB (Firestore) — new/updated/deleted?
     ↓
Batch documents (100 at a time)
     ↓
Embedding model (with rate limit + exponential backoff retry)
     ↓
Pinecone (with metadata)
     ↓
Update metadata DB — mark as processed
```

### Key decisions:

**Batching:**
- Process 100 documents at a time
- Handle embedding API rate limits with retry + backoff
- If batch fails → delete that batch's vectors by batch_id → reprocess

**Change detection:**
- Store in Firestore: `document_id | last_modified | last_embedded | status`
- Compare Drive's `last_modified` timestamp against `last_embedded`
- Only reprocess changed documents — not full rebuild every week

**Deleted documents:**
- Detect removal from Drive
- Delete vectors by `document_id` metadata filter
- Otherwise stale vectors remain forever

**Metadata per chunk:**
```python
metadata = {
    "document_id": "doc_123",
    "batch_number": "batch_4",
    "department": "HR",
    "document_name": "HR_Policy_v3.pdf",
    "document_version": "v3",
    "source": "HR_Policy_v3.pdf",
    "page": 3
}
```

---

## Step 5 — Deep Dive: Serving Layer

### Flow:
```
User Query
     ↓
Cloud Load Balancer
     ↓
Cloud Run Flask (autoscaled, minimum 2 warm instances)
     ↓
Redis Cache → cache hit? return in ~10ms
     ↓ cache miss
Embed query
     ↓
Pinecone — filter by user's department (RBAC)
     ↓
Top K chunks retrieved
     ↓
Gemini 2.5 Flash — stream response
     ↓
Write audit log → BigQuery
     ↓
Cache result in Redis
     ↓
Return answer + sources to user
```

### Key decisions:

**Autoscaling — Cloud Run:**
- Scales 1 → N instances automatically under load
- Set minimum 2 instances — avoids cold start latency
- All instances stateless — no local storage
- All state in Pinecone, Redis, BigQuery

**Latency budget (how to hit 3 seconds):**
```
Embedding query      ~100ms
Vector search        ~200ms
Gemini API call      ~1500ms
Network overhead     ~200ms
─────────────────────────────
Total                ~2000ms ✅
```

**Redis Cache:**
```python
cache_key = md5(question + department)
cached = redis.get(cache_key)
if cached:
    return cached  # ~10ms
```
Same question asked by 100 HR employees → answered from cache after first call.

**Streaming responses:**
Stream tokens back as Gemini generates. User sees response start in ~200ms even if full answer takes 2 seconds.

**Role Based Access Control (RBAC):**
Enforced at vector search level — filter by department metadata:
```python
results = pinecone.query(
    vector=query_embedding,
    filter={"department": user.department},
    top_k=3
)
```
HR user can never retrieve Legal department chunks.

---

## Step 6 — Compliance & Audit

### Audit log per query (written to BigQuery):
```json
{
    "audit_id": "uuid",
    "timestamp": "2026-05-21T03:00:00Z",
    "user_id": "emp_123",
    "department": "HR",
    "question": "What is the leave policy?",
    "answer": "Employees are entitled to...",
    "sources": [
        {
            "document_id": "doc_456",
            "document_name": "HR_Policy_v3.pdf",
            "document_version": "v3",
            "page": 4
        }
    ],
    "model_used": "gemini-2.5-flash",
    "latency_ms": 1823,
    "retrieved_chunks": 3,
    "user_feedback": null,
    "hallucination_flag": false
}
```

### Why BigQuery:
- Append only — never update, never delete (compliance requirement)
- SQL queryable — compliance team runs their own reports
- Scales to billions of rows
- Already used in production at current job

### Sample compliance queries:
```sql
-- Who asked about redundancy policy last month?
SELECT user_id, question, timestamp
FROM audit_logs
WHERE question LIKE '%redundancy%'
AND timestamp > '2026-04-01'

-- Which document version answered a specific query?
SELECT sources, document_version
FROM audit_logs
WHERE audit_id = 'uuid-123'
```

---

## Step 7 — Bottlenecks & How to Fix

| Bottleneck | Fix |
|---|---|
| Gemini API slow at peak | Redis cache for repeated queries |
| Cold start on Cloud Run | Minimum 2 warm instances |
| Embedding API rate limits | Exponential backoff + batching |
| Stale vectors after doc update | Change detection via Firestore metadata |
| RBAC enforcement | Metadata filter at Pinecone query time |
| User sees slow response | Stream tokens — perceived latency drops |

---

## Complete Requirements Coverage

| Requirement | Solution |
|---|---|
| 50,000 documents | Dataflow batch pipeline |
| Weekly updates | Cloud Scheduler + change detection |
| 5,000 concurrent users | Cloud Run autoscaling |
| Under 3 seconds | Redis cache + streaming |
| Role based access | Metadata filtering in Pinecone |
| Compliance audit | BigQuery audit table |
| Failure handling | Batch retry with metadata |
| Hallucination prevention | Grounded prompt + source citations |

---

## Your Killer Interview Statement:
*"I've worked with most of these components in production — Dataflow for pipelines, BigQuery for audit logging, GCP infrastructure for serving. This design is grounded in what I've actually built, not just theory."*

---

## Interview Structure — Always Follow This:
```
1. Clarify requirements     2 mins  — never skip
2. Estimate scale           2 mins
3. High level design        5 mins
4. Deep dive components    10 mins
5. Handle bottlenecks       5 mins
```

---

## Next — Classic System Design
- Rate Limiter
- URL Shortener
- Chat Application
- Notification System
