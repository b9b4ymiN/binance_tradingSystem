param(
    [string]$Scenario = "rsi_buy",
    [string]$EnvFile = ".env",
    [string]$WebhookUrl = "http://crypto-dasimoa.duckdns.org/webhook"
)

function Load-EnvFile {
    param([string]$Path)
    if (-not (Test-Path $Path)) { return }
    foreach ($raw in Get-Content $Path) {
        $line = $raw.Trim()
        if (-not $line -or $line.StartsWith('#') -or -not $line.Contains('=')) { continue }
        $parts = $line.Split('=', 2)
        $key = $parts[0].Trim()
        if (-not $key) { continue }
        $value = $parts[1].Trim().Trim('"')
        Set-Item -Path ("Env:{0}" -f $key) -Value $value -ErrorAction SilentlyContinue
    }
}

Load-EnvFile -Path $EnvFile

$webhookSecret = $env:WEBHOOK_SECRET
if (-not $webhookSecret) {
    Write-Error "WEBHOOK_SECRET is not set. Provide it via environment variable or .env file."
    exit 1
}

switch ($Scenario.ToLower()) {
    "rsi_buy" {
        $payload = '{"action":"buy","symbol":"BTCUSDT","price":115732,"strategy":"rsi_bollinger_scalping","stop_loss":115700,"take_profit":118130,"confidence":0.82,"notes":"RSI oversold"}'
    }
    "rsi_sell" {
        $payload = '{"action":"sell","symbol":"BTCUSDT","price":65500,"strategy":"rsi_bollinger_scalping","stop_loss":66800,"take_profit":64000,"confidence":0.78,"notes":"RSI overbought"}'
    }
    "breakout_buy" {
        $payload = '{"action":"buy","symbol":"ETHUSDT","price":3200,"strategy":"breakout_swing","stop_loss":3136,"take_profit":3680,"confidence":0.65,"timeframe":"4h"}'
    }
    "breakout_sell" {
        $payload = '{"action":"sell","symbol":"ETHUSDT","price":3150,"strategy":"breakout_swing","stop_loss":3280,"take_profit":2890,"confidence":0.60,"timeframe":"4h"}'
    }
    "manual_buy" {
        $payload = '{"action":"buy","symbol":"ADAUSDT","price":0.45,"strategy":"manual","stop_loss":0.43,"take_profit":0.49,"notes":"Manual override"}'
    }
    default {
        Write-Error "Unknown scenario '$Scenario'. Use rsi_buy, rsi_sell, breakout_buy, breakout_sell, manual_buy."
        exit 1
    }
}

$encoding = [System.Text.Encoding]::UTF8
$secretBytes = $encoding.GetBytes($webhookSecret)
$payloadBytes = $encoding.GetBytes($payload)
$hmac = [System.Security.Cryptography.HMACSHA256]::new($secretBytes)
$signatureBytes = $hmac.ComputeHash($payloadBytes)
$signature = -join ($signatureBytes | ForEach-Object { $_.ToString("x2") })

try {
    $response = Invoke-RestMethod -Method Post -Uri $WebhookUrl -Headers @{ "X-Webhook-Signature" = $signature } -ContentType "application/json" -Body $payload
    Write-Output ($response | ConvertTo-Json -Depth 5)
} catch {
    Write-Error $_
    if ($_.Exception.Response) {
        $reader = New-Object System.IO.StreamReader($_.Exception.Response.GetResponseStream())
        $body = $reader.ReadToEnd()
        Write-Error "Response body: $body"
    }
    exit 1
}
