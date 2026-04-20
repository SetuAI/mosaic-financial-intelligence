"""
MOSAIC — Multi-Agent Orchestration System for Analyst Intelligence and Cognition
aws/infrastructure.py

AWS CDK stack that provisions all MOSAIC infrastructure.

What this creates:
    - S3 bucket for document storage (raw + processed + vector store)
    - Lambda functions for ingestion and pipeline execution
    - EventBridge rule for daily ingestion scheduling
    - RDS PostgreSQL instance with pgvector extension
    - DynamoDB table for HITL audit log
    - Secrets Manager secrets for API keys
    - CloudWatch log groups and alarms
    - IAM roles with least-privilege permissions

Prerequisites:
    pip install aws-cdk-lib constructs
    aws configure  (set your AWS credentials)
    cdk bootstrap  (one-time per account/region)

Deploy:
    cd aws/
    cdk deploy

Estimated monthly cost: $37-71 (see README for breakdown)
"""

import aws_cdk as cdk

from aws_cdk import (
    Stack,
    Duration,
    RemovalPolicy,
    aws_s3 as s3,
    aws_lambda as lambda_,
    aws_events as events,
    aws_events_targets as targets,
    aws_rds as rds,
    aws_ec2 as ec2,
    aws_dynamodb as dynamodb,
    aws_secretsmanager as secretsmanager,
    aws_logs as logs,
    aws_iam as iam,
    aws_cloudwatch as cloudwatch,
    aws_cloudwatch_actions as cw_actions,
    aws_sns as sns,
)
from constructs import Construct


class MosaicStack(Stack):
    """
    Complete MOSAIC infrastructure stack.

    Deploys all resources needed to run MOSAIC in production on AWS.
    Resources are tagged with project=mosaic for cost tracking.
    """

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # ── S3 — Document Store ────────────────────────────────────────────

        self.document_bucket = s3.Bucket(
            self, "MosaicDocumentBucket",
            bucket_name       = "mosaic-financial-data",
            versioned         = True,    # keeps raw originals if re-parsed
            removal_policy    = RemovalPolicy.RETAIN,   # never auto-delete on stack destroy
            lifecycle_rules   = [
                s3.LifecycleRule(
                    # Raw HTML files are large and only needed for re-parsing.
                    # Move to Infrequent Access after 30 days — cheaper storage.
                    id         = "raw-to-ia",
                    prefix     = "raw/",
                    transitions=[
                        s3.Transition(
                            storage_class        = s3.StorageClass.INFREQUENT_ACCESS,
                            transition_after     = Duration.days(30),
                        )
                    ],
                )
            ],
        )

        # ── Secrets Manager — API Keys ─────────────────────────────────────
        # Never hardcode keys in Lambda environment variables — they are
        # visible in the console. Secrets Manager encrypts at rest and
        # supports rotation. Lambda fetches secrets at runtime.

        self.openai_secret = secretsmanager.Secret(
            self, "OpenAISecret",
            secret_name        = "mosaic/openai-api-key",
            description        = "OpenAI API key for MOSAIC LLM calls",
        )

        self.langsmith_secret = secretsmanager.Secret(
            self, "LangSmithSecret",
            secret_name        = "mosaic/langsmith-api-key",
            description        = "LangSmith API key for MOSAIC tracing",
        )

        self.sec_agent_secret = secretsmanager.Secret(
            self, "SECAgentSecret",
            secret_name        = "mosaic/sec-user-agent",
            description        = "SEC EDGAR user agent string",
        )

        # ── VPC — Network Isolation ────────────────────────────────────────
        # RDS must be in a VPC. We use a minimal VPC with private subnets
        # for the database and public subnets for Lambda internet access.

        self.vpc = ec2.Vpc(
            self, "MosaicVpc",
            max_azs          = 2,    # two AZs for RDS Multi-AZ standby
            nat_gateways     = 1,    # one NAT for Lambda outbound internet
            subnet_configuration=[
                ec2.SubnetConfiguration(
                    name       = "public",
                    subnet_type= ec2.SubnetType.PUBLIC,
                    cidr_mask  = 24,
                ),
                ec2.SubnetConfiguration(
                    name       = "private",
                    subnet_type= ec2.SubnetType.PRIVATE_WITH_EGRESS,
                    cidr_mask  = 24,
                ),
            ],
        )

        # ── RDS PostgreSQL + pgvector ──────────────────────────────────────
        # pgvector replaces our local FAISS index in production.
        # Aurora Serverless v2 scales to zero when idle — no cost when
        # MOSAIC is not running. Perfect for our batch workload pattern.

        self.db_secret = rds.DatabaseSecret(
            self, "MosaicDBSecret",
            username    = "mosaic_admin",
            secret_name = "mosaic/rds-credentials",
        )

        self.database = rds.DatabaseInstance(
            self, "MosaicDatabase",
            engine              = rds.DatabaseInstanceEngine.postgres(
                version=rds.PostgresEngineVersion.VER_16_3
            ),
            instance_type       = ec2.InstanceType.of(
                ec2.InstanceClass.BURSTABLE3,
                ec2.InstanceSize.MICRO,    # t3.micro — sufficient for our scale
            ),
            vpc                 = self.vpc,
            vpc_subnets         = ec2.SubnetSelection(
                subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS
            ),
            credentials         = rds.Credentials.from_secret(self.db_secret),
            database_name       = "mosaic",
            allocated_storage   = 20,      # GB — plenty for our chunk metadata
            multi_az            = False,   # single-AZ for dev/demo cost control
            deletion_protection = True,    # prevents accidental destroy
            removal_policy      = RemovalPolicy.SNAPSHOT,  # snapshot before delete
            cloudwatch_logs_exports=["postgresql"],
        )

        # ── DynamoDB — HITL Audit Log ──────────────────────────────────────
        # Append-only audit log for all HITL review decisions.
        # DynamoDB is the right choice here — it is effectively a document
        # store for our JSONL entries, and we never need relational queries.

        self.hitl_table = dynamodb.Table(
            self, "HITLAuditTable",
            table_name      = "mosaic-hitl-audit",
            partition_key   = dynamodb.Attribute(
                name="signal_id",
                type=dynamodb.AttributeType.STRING,
            ),
            sort_key        = dynamodb.Attribute(
                name="review_timestamp",
                type=dynamodb.AttributeType.STRING,
            ),
            billing_mode    = dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy  = RemovalPolicy.RETAIN,
            point_in_time_recovery=True,    # allows restore to any point in time
        )

        # Add a GSI so we can query by ticker — useful for the audit log viewer
        self.hitl_table.add_global_secondary_index(
            index_name  = "ticker-index",
            partition_key=dynamodb.Attribute(
                name="ticker",
                type=dynamodb.AttributeType.STRING,
            ),
            sort_key    = dynamodb.Attribute(
                name="review_timestamp",
                type=dynamodb.AttributeType.STRING,
            ),
        )

        # ── Lambda IAM Role ────────────────────────────────────────────────
        # Least-privilege role for all MOSAIC Lambda functions.
        # Each function gets only what it needs — no wildcards.

        self.lambda_role = iam.Role(
            self, "MosaicLambdaRole",
            assumed_by   = iam.ServicePrincipal("lambda.amazonaws.com"),
            description  = "Execution role for all MOSAIC Lambda functions",
        )

        # S3 access — read/write to document bucket only
        self.document_bucket.grant_read_write(self.lambda_role)

        # Secrets access — read secrets at runtime
        self.openai_secret.grant_read(self.lambda_role)
        self.langsmith_secret.grant_read(self.lambda_role)
        self.sec_agent_secret.grant_read(self.lambda_role)
        self.db_secret.grant_read(self.lambda_role)

        # DynamoDB access — write to HITL table
        self.hitl_table.grant_write_data(self.lambda_role)

        # CloudWatch Logs — standard Lambda logging
        self.lambda_role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name(
                "service-role/AWSLambdaVPCAccessExecutionRole"
            )
        )

        # ── Lambda — EDGAR Poller ──────────────────────────────────────────
        # Runs daily via EventBridge. Checks EDGAR for new filings and
        # downloads any that are not already in S3.

        self.edgar_poller = lambda_.Function(
            self, "EdgarPoller",
            function_name  = "mosaic-edgar-poller",
            runtime        = lambda_.Runtime.PYTHON_3_12,
            handler        = "lambda_handlers.edgar_poller_handler",
            code           = lambda_.Code.from_asset(".."),   # project root
            role           = self.lambda_role,
            timeout        = Duration.minutes(10),   # ingestion can be slow
            memory_size    = 512,
            vpc            = self.vpc,
            environment    = {
                "S3_BUCKET":            self.document_bucket.bucket_name,
                "OPENAI_SECRET_ARN":    self.openai_secret.secret_arn,
                "LANGSMITH_SECRET_ARN": self.langsmith_secret.secret_arn,
                "SEC_SECRET_ARN":       self.sec_agent_secret.secret_arn,
                "DB_SECRET_ARN":        self.db_secret.secret_arn,
            },
        )

        # ── Lambda — Pipeline Runner ───────────────────────────────────────
        # Triggered when a new processed document lands in S3.
        # Runs the full MOSAIC agent pipeline and writes results back to S3.

        self.pipeline_runner = lambda_.Function(
            self, "PipelineRunner",
            function_name  = "mosaic-pipeline-runner",
            runtime        = lambda_.Runtime.PYTHON_3_12,
            handler        = "lambda_handlers.pipeline_runner_handler",
            code           = lambda_.Code.from_asset(".."),
            role           = self.lambda_role,
            timeout        = Duration.minutes(15),   # full pipeline ~5 min
            memory_size    = 1024,    # agents need more memory than ingestion
            vpc            = self.vpc,
            environment    = {
                "S3_BUCKET":            self.document_bucket.bucket_name,
                "OPENAI_SECRET_ARN":    self.openai_secret.secret_arn,
                "LANGSMITH_SECRET_ARN": self.langsmith_secret.secret_arn,
                "DB_SECRET_ARN":        self.db_secret.secret_arn,
                "HITL_TABLE":           self.hitl_table.table_name,
            },
        )

        # ── EventBridge — Daily Ingestion Schedule ─────────────────────────
        # Runs the EDGAR poller every day at 6 AM UTC.
        # Most companies file earnings within a day of the event —
        # daily polling ensures we catch new filings within 24 hours.

        self.daily_schedule = events.Rule(
            self, "DailyIngestionSchedule",
            rule_name   = "mosaic-daily-ingestion",
            description = "Triggers EDGAR poller daily at 6 AM UTC",
            schedule    = events.Schedule.cron(
                minute="0",
                hour="6",
                month="*",
                week_day="MON-FRI",   # weekdays only — no weekend filings
                year="*",
            ),
        )

        self.daily_schedule.add_target(
            targets.LambdaFunction(self.edgar_poller)
        )

        # ── CloudWatch — Monitoring ────────────────────────────────────────

        # Alert if the EDGAR poller fails — we should know within minutes
        # if a daily ingestion run threw an error.
        self.alert_topic = sns.Topic(
            self, "MosaicAlerts",
            topic_name  = "mosaic-alerts",
            display_name= "MOSAIC Pipeline Alerts",
        )

        cloudwatch.Alarm(
            self, "EdgarPollerErrors",
            alarm_name   = "mosaic-edgar-poller-errors",
            alarm_description= "EDGAR poller Lambda is throwing errors",
            metric       = self.edgar_poller.metric_errors(
                period=Duration.minutes(5)
            ),
            threshold            = 1,
            evaluation_periods   = 1,
            comparison_operator  = cloudwatch.ComparisonOperator.GREATER_THAN_OR_EQUAL_TO_THRESHOLD,
            treat_missing_data   = cloudwatch.TreatMissingData.NOT_BREACHING,
        )

        # ── Stack outputs ──────────────────────────────────────────────────
        # These values are printed after cdk deploy and needed for configuration.

        cdk.CfnOutput(self, "BucketName",
            value       = self.document_bucket.bucket_name,
            description = "S3 bucket for MOSAIC documents",
        )

        cdk.CfnOutput(self, "DatabaseEndpoint",
            value       = self.database.db_instance_endpoint_address,
            description = "RDS PostgreSQL endpoint for pgvector",
        )

        cdk.CfnOutput(self, "HITLTableName",
            value       = self.hitl_table.table_name,
            description = "DynamoDB table for HITL audit log",
        )


# ── CDK App entry point ────────────────────────────────────────────────────────

app   = cdk.App()
stack = MosaicStack(
    app, "MosaicStack",
    env=cdk.Environment(
        account = app.node.try_get_context("account"),
        region  = app.node.try_get_context("region") or "us-east-1",
    ),
    tags={"project": "mosaic", "environment": "production"},
)
app.synth()