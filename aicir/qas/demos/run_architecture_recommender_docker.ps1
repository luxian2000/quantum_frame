$ErrorActionPreference = "Stop"

$container = "2b37975e80d1"
$containerRepo = "/workspace/quantum_frame_qas_1"
$containerPython = "/opt/conda/envs/myproject/bin/python"
$containerOutput = "$containerRepo/aicir/qas/demos/vqe_qas_architecture_recommender_results.txt"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..\..")
$localOutput = Join-Path $repoRoot "aicir\qas\demos\vqe_qas_architecture_recommender_results.txt"
$localLog = Join-Path $repoRoot "aicir\qas\demos\vqe_qas_architecture_recommender_run.log"

docker exec $container bash -lc "cd $containerRepo && $containerPython -m aicir.qas.demos.VQE_QAS_demo_architecture_recommender --input aicir/qas/demos/vqe_qas_architecture_recommender_input_h2.json --output aicir/qas/demos/vqe_qas_architecture_recommender_results.txt" *> $localLog
if ($LASTEXITCODE -ne 0) {
    Write-Host "Architecture recommender failed. Log: $localLog"
    exit $LASTEXITCODE
}

docker cp "${container}:$containerOutput" $localOutput
if ($LASTEXITCODE -ne 0) {
    Write-Host "Result copy failed. Log: $localLog"
    exit $LASTEXITCODE
}

Write-Host "Done."
Write-Host "Result TXT: $localOutput"
Write-Host "Run log: $localLog"
