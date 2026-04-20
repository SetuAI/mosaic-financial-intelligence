"""
MOSAIC — Multi-Agent Orchestration System for Analyst Intelligence and Cognition
aws/lambda_handlers.py

Lambda function handlers for the AWS deployment.

Design principle — handlers are thin wrappers, nothing more:
    The business logic lives in the main codebase — agents, tools, graph.
    These handlers do only the AWS-specific work:
        1. Fetch API keys from Secrets Manager
        2. Download/upload files from/to S3
        3. Call the same pipeline functions we use locally
        4. Write results to DynamoDB

    This means zero duplication. A bug fixed locally is fixed in production.
    A new agent added locally works in production without touching this file.

Two handlers:
    edgar_poller_handler    — runs daily via EventBridge, ingests new filings
    pipeline_runner_handler — runs the full MOSAIC agent pipeline on demand
                              or triggered by a new file landing in S3

Lambda constraints to be aware of:
    - /tmp is the only writable directory — 512MB ephemeral storage
    - Timeout max 15 minutes — pipeline runs in ~5 minutes, fine
    - Cold start adds ~3-5 seconds — acceptable for our batch workload
    - Execution environment is reused on warm invocations — boto3 clients
      initialised at module level are reused, making warm calls faster
"""

import json
import os
import boto3

from config.logging_config import setup_logging, get_logger

# Initialise logging at module level — runs once on cold start,
# reused on warm invocations.
setup_logging()
logger = get_logger(__name__)


# ── AWS clients ────────────────────────────────────────────────────────────────
# Initialised once at module level — reused across warm Lambda invocations.
# Creating boto3 clients inside handlers adds 100-200ms per call unnecessarily.

s3_client      = boto3.client("s3")
secrets_client = boto3.client("secretsmanager")
dynamodb       = boto3.resource("dynamodb")


# ── Secret loading ─────────────────────────────────────────────────────────────

def _load_secrets() -> None:
    """
    Fetches API keys from Secrets Manager and injects them into os.environ.

    Called at the start of every handler. The SECRETS_LOADED flag prevents
    redundant Secrets Manager calls on warm invocations — the execution
    environment persists between calls so the keys are already in os.environ.

    Why Secrets Manager and not Lambda environment variables:
        Environment variables are visible to anyone with AWS console access.
        Secrets Manager encrypts at rest, audits every access, and supports
        rotation. For an API key that costs money if leaked, this matters.
    """
    # Skip on warm invocations — secrets already loaded from the cold start
    if os.environ.get("SECRETS_LOADED") == "true":
        return

    logger.info("Cold start — loading secrets from Secrets Manager")

    # Maps environment variable names (set in CDK) to the env var names
    # our pydantic settings expects. The ARN is set by infrastructure.py
    # when it creates the Lambda function.
    secret_map = {
        "OPENAI_SECRET_ARN":    "OPENAI_API_KEY",
        "LANGSMITH_SECRET_ARN": "LANGCHAIN_API_KEY",
        "SEC_SECRET_ARN":       "SEC_USER_AGENT",
    }

    for arn_env_key, target_env_key in secret_map.items():
        secret_arn = os.environ.get(arn_env_key)

        if not secret_arn:
            logger.warning(
                f"Secret ARN not configured: {arn_env_key} — "
                f"check Lambda environment variables in CDK"
            )
            continue

        try:
            response     = secrets_client.get_secret_value(SecretId=secret_arn)
            secret_value = response.get("SecretString", "")
            os.environ[target_env_key] = secret_value
            logger.info(f"Loaded secret into {target_env_key}")

        except Exception as e:
            logger.error(
                f"Failed to load secret {arn_env_key}: {e} — "
                f"this will cause downstream failures"
            )

    # Tell LangSmith to trace — Lambda always runs in production mode
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"]    = "mosaic-prod"

    # Mark secrets as loaded so we skip this block on warm invocations
    os.environ["SECRETS_LOADED"] = "true"


# ── File paths ─────────────────────────────────────────────────────────────────

def _configure_paths() -> None:
    """
    Points the document store and vector store at Lambda's /tmp directory.

    Our pydantic settings reads RAW_DATA_PATH and PROCESSED_DATA_PATH from
    environment variables. Setting them here redirects all file operations
    to /tmp without changing any application code.

    Lambda's /tmp is 512MB of ephemeral storage — sufficient for our
    document corpus and vector store. It is wiped between invocations
    unless the execution environment is reused (warm start).
    """
    os.environ["RAW_DATA_PATH"]       = "/tmp/data/raw"
    os.environ["PROCESSED_DATA_PATH"] = "/tmp/data/processed"


# ── S3 helpers ─────────────────────────────────────────────────────────────────

def _sync_vector_store_from_s3() -> None:
    """
    Downloads the FAISS vector store from S3 to /tmp before running agents.

    The vector store is built by run_processing.py and stored in S3 after
    each rebuild. Lambda downloads it on each invocation — the files are
    small enough (~50MB) that this adds only 2-3 seconds on a cold start.

    For warm invocations, /tmp may still have the files from the previous
    run — we re-download anyway to ensure we always use the latest index.
    """
    bucket = os.environ.get("S3_BUCKET", "mosaic-financial-data")

    files_to_sync = [
        ("vector_store/faiss.index",         "/tmp/data/vector_store/faiss.index"),
        ("vector_store/chunk_metadata.json", "/tmp/data/vector_store/chunk_metadata.json"),
    ]

    for s3_key, local_path in files_to_sync:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        try:
            s3_client.download_file(bucket, s3_key, local_path)
            logger.info(f"Downloaded s3://{bucket}/{s3_key}")
        except Exception as e:
            # Missing vector store is a hard failure — agents cannot run without it
            raise RuntimeError(
                f"Failed to download vector store from S3: {e}. "
                f"Run python -m processing.run_processing locally and "
                f"sync the data/vector_store/ folder to S3."
            )


def _upload_processed_to_s3() -> None:
    """
    Uploads all processed JSON documents from /tmp to S3 after ingestion.

    After the EDGAR poller runs, new documents are in /tmp/data/processed/.
    This function syncs them to S3 so the pipeline runner can access them
    and so they are durably stored beyond the Lambda invocation lifetime.
    """
    import glob

    bucket         = os.environ.get("S3_BUCKET", "mosaic-financial-data")
    processed_base = "/tmp/data/processed"
    uploaded       = 0

    for local_path in glob.glob(f"{processed_base}/**/*.json", recursive=True):
        # Convert local path to S3 key —
        # /tmp/data/processed/MSFT/10-K/file.json → processed/MSFT/10-K/file.json
        s3_key = local_path.replace("/tmp/data/", "")
        s3_client.upload_file(local_path, bucket, s3_key)
        uploaded += 1

    logger.info(f"Uploaded {uploaded} processed documents to s3://{bucket}/processed/")


def _upload_vector_store_to_s3() -> None:
    """
    Uploads the rebuilt vector store from /tmp to S3.

    Called after run_processing rebuilds the FAISS index with new documents.
    Subsequent Lambda invocations will download the updated index.
    """
    bucket = os.environ.get("S3_BUCKET", "mosaic-financial-data")

    files_to_upload = [
        ("/tmp/data/vector_store/faiss.index",         "vector_store/faiss.index"),
        ("/tmp/data/vector_store/chunk_metadata.json", "vector_store/chunk_metadata.json"),
        ("/tmp/data/vector_store/stats.json",          "vector_store/stats.json"),
    ]

    for local_path, s3_key in files_to_upload:
        if os.path.exists(local_path):
            s3_client.upload_file(local_path, bucket, s3_key)
            logger.info(f"Uploaded {local_path} to s3://{bucket}/{s3_key}")


def _write_brief_to_s3(result: dict) -> str:
    """
    Writes the final intelligence brief to S3 as a permanent record.

    Each brief is stored with its run_id as the filename — this gives
    us a complete history of every pipeline run's output in S3.

    Args:
        result: Final pipeline state containing final_brief and run_id.

    Returns:
        S3 key where the brief was written.
    """
    import tempfile

    run_id = result.get("run_id", "unknown")
    brief  = result.get("final_brief", "No brief generated.")
    bucket = os.environ.get("S3_BUCKET", "mosaic-financial-data")
    s3_key = f"briefs/{run_id}.txt"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(brief)
        tmp_path = f.name

    s3_client.upload_file(tmp_path, bucket, s3_key)
    logger.info(f"Brief written to s3://{bucket}/{s3_key}")

    return s3_key


# ── DynamoDB helpers ───────────────────────────────────────────────────────────

def _write_hitl_to_dynamodb(hitl_queue: list[dict]) -> None:
    """
    Writes pending HITL review items to DynamoDB.

    The HITL review interface reads from this table to present signals
    to analysts. Items have a 30-day TTL — if not reviewed within 30 days
    they expire automatically, keeping the table clean.

    Args:
        hitl_queue: HITL queue items from state["hitl_queue"].
    """
    if not hitl_queue:
        logger.info("No HITL items to write to DynamoDB")
        return

    table_name = os.environ.get("HITL_TABLE", "mosaic-hitl-audit")
    table      = dynamodb.Table(table_name)

    from datetime import datetime, timezone
    import time

    # TTL: 30 days from now
    ttl = int(time.time()) + (30 * 24 * 60 * 60)

    with table.batch_writer() as batch:
        for item in hitl_queue:
            # DynamoDB requires all values to be serialisable —
            # convert any float values to Decimal for numeric precision
            from decimal import Decimal
            serialised = json.loads(
                json.dumps(item),
                parse_float=Decimal
            )

            batch.put_item(Item={
                **serialised,
                "review_status": "pending",
                "created_at":    datetime.now(timezone.utc).isoformat(),
                "ttl":           ttl,
            })

    logger.info(f"Wrote {len(hitl_queue)} HITL items to DynamoDB table: {table_name}")


# ── Handler 1: EDGAR Poller ────────────────────────────────────────────────────

def edgar_poller_handler(event: dict, context) -> dict:
    """
    Daily ingestion Lambda — fetches new SEC filings from EDGAR.

    Triggered by EventBridge on a daily schedule (weekdays at 6 AM UTC).
    Runs the same ingestion pipeline we run locally, then uploads results
    to S3 and rebuilds the vector store.

    EventBridge event format:
        {"source": "aws.events", "detail-type": "Scheduled Event", ...}
        We do not use the event payload — the pipeline always fetches
        whatever is new since the last run.

    Args:
        event:   EventBridge scheduled event.
        context: Lambda context — use context.get_remaining_time_in_millis()
                 to check if we are close to timeout.

    Returns:
        Standard Lambda response dict with statusCode and body.
    """
    logger.info("EDGAR poller Lambda starting")

    _load_secrets()
    _configure_paths()

    try:
        from ingestion.run_ingestion import run_pipeline
        from ingestion.document_store import get_ingestion_summary
        from processing.run_processing import run_pipeline as run_processing

        # ── Step 1: Ingest new filings from EDGAR ──────────────────────────
        logger.info("Running EDGAR ingestion pipeline")
        run_pipeline()

        # ── Step 2: Upload processed documents to S3 ───────────────────────
        logger.info("Uploading processed documents to S3")
        _upload_processed_to_s3()

        # ── Step 3: Rebuild vector store with new documents ────────────────
        # Only rebuild if new documents were added — checks /tmp for content
        logger.info("Rebuilding vector store with new documents")
        run_processing()

        # ── Step 4: Upload updated vector store to S3 ──────────────────────
        logger.info("Uploading updated vector store to S3")
        _upload_vector_store_to_s3()

        # ── Step 5: Build summary for the response ─────────────────────────
        summary = get_ingestion_summary()
        total   = sum(
            count
            for ticker_data in summary.values()
            for count in ticker_data.values()
        )

        logger.info(f"EDGAR poller complete — {total} documents in corpus")

        return {
            "statusCode": 200,
            "body": json.dumps({
                "message":        "Ingestion complete",
                "total_documents": total,
                "summary":        summary,
            }),
        }

    except Exception as e:
        logger.error(f"EDGAR poller failed: {e}", exc_info=True)

        # Return 500 — CloudWatch alarm triggers on Lambda errors
        return {
            "statusCode": 500,
            "body": json.dumps({
                "error":   str(e),
                "message": "Ingestion failed — check CloudWatch logs",
            }),
        }


# ── Handler 2: Pipeline Runner ─────────────────────────────────────────────────

def pipeline_runner_handler(event: dict, context) -> dict:
    """
    Agent pipeline Lambda — runs the full MOSAIC analysis.

    Can be invoked two ways:

    1. Direct invocation (on-demand analysis):
       Payload: {
           "query":          "Analyse MSFT for risk signals",
           "tickers":        ["MSFT", "NVDA"],
           "primary_ticker": "MSFT"
       }

    2. Scheduled (daily full-universe run):
       No payload needed — uses default tickers from settings.

    Steps:
        1. Download vector store from S3 to /tmp
        2. Run the full MOSAIC pipeline via run_mosaic()
        3. Write the intelligence brief to S3
        4. Write HITL queue items to DynamoDB
        5. Return summary

    Args:
        event:   Invocation payload — see format above.
        context: Lambda context.

    Returns:
        Standard Lambda response dict with run summary.
    """
    logger.info("Pipeline runner Lambda starting")

    _load_secrets()
    _configure_paths()

    try:
        # ── Step 1: Download vector store from S3 ──────────────────────────
        logger.info("Syncing vector store from S3")
        _sync_vector_store_from_s3()

        # ── Step 2: Parse the invocation payload ───────────────────────────
        # Support both direct JSON payload and API Gateway wrapped payload
        if isinstance(event.get("body"), str):
            payload = json.loads(event["body"])
        else:
            payload = event

        query = payload.get(
            "query",
            "Analyse all tickers for risk signals across all dimensions"
        )
        tickers = payload.get(
            "tickers",
            ["AMZN", "MSFT", "GOOG", "ORCL", "AAPL", "NVDA"]
        )
        primary_ticker = payload.get("primary_ticker", tickers[0] if tickers else None)

        logger.info(
            f"Pipeline config — "
            f"tickers={tickers} "
            f"query='{query[:60]}'"
        )

        # ── Step 3: Run the full MOSAIC pipeline ───────────────────────────
        from graph.graph_builder import run_mosaic

        result = run_mosaic(
            query          = query,
            tickers        = tickers,
            primary_ticker = primary_ticker,
            triggered_by   = "lambda_invocation",
        )

        # ── Step 4: Write brief to S3 ──────────────────────────────────────
        brief_s3_key = _write_brief_to_s3(result)

        # ── Step 5: Write HITL queue to DynamoDB ───────────────────────────
        _write_hitl_to_dynamodb(result.get("hitl_queue", []))

        # ── Step 6: Build response summary ─────────────────────────────────
        run_summary = result.get("run_summary", {})

        logger.info(
            f"Pipeline complete — "
            f"run_id={result.get('run_id')} "
            f"signals={run_summary.get('total_signals', 0)} "
            f"hitl={run_summary.get('hitl_required', 0)} "
            f"risk={run_summary.get('overall_risk', 'unknown')}"
        )

        return {
            "statusCode": 200,
            "body": json.dumps({
                "run_id":          result.get("run_id"),
                "total_signals":   run_summary.get("total_signals", 0),
                "auto_approved":   run_summary.get("auto_approved", 0),
                "hitl_required":   run_summary.get("hitl_required", 0),
                "overall_risk":    run_summary.get("overall_risk", "unknown"),
                "brief_location":  f"s3://mosaic-financial-data/{brief_s3_key}",
                "tickers_covered": run_summary.get("tickers_covered", []),
            }),
        }

    except Exception as e:
        logger.error(f"Pipeline runner failed: {e}", exc_info=True)

        return {
            "statusCode": 500,
            "body": json.dumps({
                "error":   str(e),
                "message": "Pipeline failed — check CloudWatch logs and LangSmith traces",
            }),
        }