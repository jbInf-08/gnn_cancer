# Monitor script progress
$logFile = "monitoring_log.txt"
$processId = 21808

while ($true) {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    
    # Check if process is still running
    $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
    if ($process) {
        $cpu = $process.CPU
        $memory = [math]::Round($process.WorkingSet / 1MB, 2)
        $runtime = (Get-Date) - $process.StartTime
        
        # Count chunk files
        $chunkFiles = Get-ChildItem "data/processed/cnv_matrix_chunk_*.csv" -ErrorAction SilentlyContinue
        $chunkCount = $chunkFiles.Count
        $latestChunk = if ($chunkFiles) { ($chunkFiles | Sort-Object LastWriteTime -Descending | Select-Object -First 1).Name } else { "None" }
        
        # Calculate progress
        $totalFiles = 1000
        $filesProcessed = $chunkCount * 100
        $progressPercent = [math]::Round(($filesProcessed / $totalFiles) * 100, 1)
        
        $status = @"
[$timestamp] PROCESS RUNNING
- CPU Usage: $cpu units
- Memory Usage: $memory MB
- Runtime: $($runtime.ToString("hh\:mm\:ss"))
- Chunks Completed: $chunkCount/10
- Files Processed: $filesProcessed/$totalFiles ($progressPercent%)
- Latest Chunk: $latestChunk
"@
        
        Write-Host $status
        Add-Content $logFile $status
        
    } else {
        $status = "[$timestamp] PROCESS COMPLETED OR STOPPED"
        Write-Host $status
        Add-Content $logFile $status
        break
    }
    
    # Wait 2 minutes before next check
    Start-Sleep -Seconds 120
} 