#!/bin/bash
set -e

PROJECT_ID="tms-gemini-enterprise"
SERVICE_NAME="ge-a2a-middleware"
REGION="asia-southeast1"

echo "🚀 Starting Direct Source Deployment to Google Cloud Run..."
echo "Project: $PROJECT_ID | Service: $SERVICE_NAME | Region: $REGION"

gcloud run deploy "$SERVICE_NAME" \
  --source . \
  --platform managed \
  --region "$REGION" \
  --allow-unauthenticated \
  --port 8080

echo "✅ Deployment Successful!"
