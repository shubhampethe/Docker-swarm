# Docker Swarm Deployment Script for Custom Embedding Service
# Deploys service with 1 idle replica baseline and automatic scaling to 3 replicas under load

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  Docker Swarm: Deploying Auto-Scaling Embedding Service" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Ensure Swarm is active
$swarmStatus = (docker info --format '{{.Swarm.LocalNodeState}}').Trim()
if ($swarmStatus -ne "active") {
    Write-Host "[1/4] Initializing Docker Swarm..." -ForegroundColor Yellow
    docker swarm init
} else {
    Write-Host "[1/4] Docker Swarm is active." -ForegroundColor Green
}

# 2. Build the embedding service Docker image
Write-Host "[2/4] Building custom-embedding-service:latest Docker image..." -ForegroundColor Yellow
docker build -t custom-embedding-service:latest ./embedding_service

# 3. Deploy stack to Swarm
Write-Host "[3/4] Deploying stack 'rag_stack' (baseline: 1 replica)..." -ForegroundColor Yellow
docker stack deploy -c docker-compose.yml rag_stack

# 4. Check service status
Write-Host "[4/4] Checking Swarm service status..." -ForegroundColor Yellow
Start-Sleep -Seconds 3
docker service ls
docker service ps rag_stack_embedding-service

Write-Host "`n✓ Deployment complete! Embedding service is accessible at http://localhost:8001" -ForegroundColor Green
Write-Host "✓ Baseline: 1 replica running." -ForegroundColor Green
Write-Host "⚡ Automated Scaling: When ingesting documents with 100+ chunks, the pipeline automatically" -ForegroundColor Cyan
Write-Host "   scales the service up to 3 replicas (docker service scale rag_stack_embedding-service=3)" -ForegroundColor Cyan
Write-Host "   and scales back down to 1 replica when finished!" -ForegroundColor Cyan
