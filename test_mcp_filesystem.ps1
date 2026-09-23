# ============================================================
# Agent MCP Filesystem 能力测试脚本 (v4)
#
# v4 关键修正：
#   - 请求 Content-Type 从 application/json 改成
#     application/x-www-form-urlencoded（FastAPI 端点用的是 Form(...)）
#   - 所有 Form 参数补全（history / workspace / skills）
#   - 4xx/5xx 不抛异常，保留响应体供诊断
#   - 日志文件同时记录请求和响应
#
# 其他：
#   - 用 .NET HttpClient，不依赖 curl.exe
#   - 每个用例独立 try/catch，失败不终止
#   - 三层输出：控制台 + JSON 汇总 + 每个用例的原始日志
#   - 每个用例独立超时
#
# 用法：
#   cd D:\ycy\LLM\LLM
#   .\test_mcp_filesystem.ps1
#   .\test_mcp_filesystem.ps1 -Only 1,2,3
#   .\test_mcp_filesystem.ps1 -SkipPrepare
#
# 保存：UTF-8 with BOM（VSCode: 右下角编码 → Save with Encoding）
# ============================================================

param(
    [string]$Base = "http://localhost:8000",
    [string]$DataDir = "D:/ycy/LLM/LLM/data",
    [int]$TimeoutSec = 300,
    [switch]$SkipPrepare,
    [string]$Only = ""
)

$ErrorActionPreference = "Continue"
$ProgressPreference = "SilentlyContinue"

# ── 输出目录 ──
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$OutputRoot = Join-Path $ScriptDir "test_results_$Timestamp"
$LogDir = Join-Path $OutputRoot "logs"
$SummaryFile = Join-Path $OutputRoot "summary.json"
$SummaryTxtFile = Join-Path $OutputRoot "summary.txt"

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

# ── 颜色工具 ──
function Write-Section($text) {
    Write-Host ""
    Write-Host ("=" * 70) -ForegroundColor DarkCyan
    Write-Host "  $text" -ForegroundColor Cyan
    Write-Host ("=" * 70) -ForegroundColor DarkCyan
}
function Write-SubSection($text) {
    Write-Host ""
    Write-Host "── $text" -ForegroundColor Yellow
}
function Write-Ok($text)   { Write-Host "  [OK]   $text" -ForegroundColor Green }
function Write-Fail($text) { Write-Host "  [FAIL] $text" -ForegroundColor Red }
function Write-Info($text) { Write-Host "  [INFO] $text" -ForegroundColor Gray }
function Write-Warn($text) { Write-Host "  [WARN] $text" -ForegroundColor Magenta }

# ── 结果收集 ──
$script:Results = New-Object System.Collections.ArrayList
function Add-Result($name, $passed, $detail, $durationSec = 0.0, $logFile = "") {
    $null = $script:Results.Add([PSCustomObject]@{
        name = $name
        passed = $passed
        detail = $detail
        duration_sec = [Math]::Round($durationSec, 2)
        log_file = $logFile
        timestamp = (Get-Date).ToString("o")
    })
}

# ── 用例过滤 ──
$OnlySet = @{}
if ($Only -ne "") {
    foreach ($n in ($Only -split ",")) {
        $trimmed = $n.Trim()
        if ($trimmed -ne "") { $OnlySet[$trimmed] = $true }
    }
}
function Should-Run($caseNum) {
    if ($OnlySet.Count -eq 0) { return $true }
    return $OnlySet.ContainsKey("$caseNum")
}

# ── 加载 .NET 类型 ──
Add-Type -AssemblyName System.Net.Http

# ============================================================
# HTTP 请求封装（form-urlencoded）
# ============================================================

function Invoke-AgentTask {
    param(
        [string]$Message,
        [string]$Mode = "ask",
        [int]$Timeout = 300,
        [string]$LogName = ""
    )

    # ── 构造 form-urlencoded body ──
    # FastAPI 端点是 Form(...) 参数，不接受 application/json。
    # 所有 Form 字段都要传，未指定的用默认值。
    $formParts = @(
        "message=$([Uri]::EscapeDataString($Message))",
        "mode=$([Uri]::EscapeDataString($Mode))",
        "history=[]",
        "workspace=[]",
        "skills=[]"
    )
    $formBody = $formParts -join "&"

    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $raw = ""
    $err = $null
    $httpStatus = $null
    $httpReason = ""

    try {
        $handler = New-Object System.Net.Http.HttpClientHandler
        $handler.AutomaticDecompression = (
            [System.Net.DecompressionMethods]::GZip -bor
            [System.Net.DecompressionMethods]::Deflate
        )
        $handler.UseProxy = $false

        $client = New-Object System.Net.Http.HttpClient($handler)
        $client.Timeout = [TimeSpan]::FromSeconds($Timeout)

        $content = New-Object System.Net.Http.StringContent(
            $formBody,
            [System.Text.Encoding]::UTF8,
            "application/x-www-form-urlencoded"
        )

        $request = New-Object System.Net.Http.HttpRequestMessage(
            [System.Net.Http.HttpMethod]::Post,
            "$Base/api/agent/tasks/stream"
        )
        $request.Content = $content

        $response = $client.SendAsync(
            $request,
            [System.Net.Http.HttpCompletionOption]::ResponseContentRead
        ).GetAwaiter().GetResult()

        $httpStatus = [int]$response.StatusCode
        $httpReason = $response.ReasonPhrase
        $raw = $response.Content.ReadAsStringAsync().GetAwaiter().GetResult()

        if (-not $response.IsSuccessStatusCode) {
            $err = "HTTP $httpStatus $httpReason"
        }

        $response.Dispose()
        $request.Dispose()
        $content.Dispose()
        $client.Dispose()
        $handler.Dispose()
    } catch {
        $err = "$_"
        if ($httpStatus) {
            $err = "HTTP $httpStatus | $err"
        }
    }
    $sw.Stop()

    # ── 保存原始日志（请求 + 响应 + 错误） ──
    $logFile = ""
    if ($LogName -ne "") {
        $logFile = Join-Path $LogDir "$LogName.txt"
        try {
            $logLines = @(
                "=== REQUEST ===",
                "POST $Base/api/agent/tasks/stream",
                "Content-Type: application/x-www-form-urlencoded",
                "Body: $formBody",
                "",
                "=== RESPONSE (HTTP $httpStatus $httpReason) ===",
                $raw
            )
            if ($err) {
                $logLines += ""
                $logLines += "=== ERROR ==="
                $logLines += $err
            }
            $logLines -join "`r`n" | Out-File -Encoding utf8 $logFile
        } catch {
            $logFile = ""
        }
    }

    return @{
        Raw = $raw
        Duration = $sw.Elapsed.TotalSeconds
        Error = $err
        LogFile = $logFile
        HttpStatus = $httpStatus
        HttpReason = $httpReason
    }
}

# ============================================================
# SSE 解析
# ============================================================

function Parse-SseEvents {
    param([string]$Raw)
    $events = @()
    if (-not $Raw) { return $events }

    $normalized = $Raw -replace "`r`n", "`n" -replace "`r", "`n"

    foreach ($line in ($normalized -split "`n")) {
        $line = $line.Trim()
        if (-not $line.StartsWith("data:")) { continue }
        $json = $line.Substring(5).Trim()
        if ($json -eq "[DONE]" -or $json -eq "") { continue }
        try {
            $obj = $json | ConvertFrom-Json
            $events += $obj
        } catch {
            # 忽略单行解析失败
        }
    }
    return $events
}

# ============================================================
# 用例结果展示
# ============================================================

function Show-TaskSummary {
    param($Result, $Name)

    # ── 4xx/5xx 诊断 ──
    if ($Result.HttpStatus -and $Result.HttpStatus -ge 400) {
        Write-Fail "HTTP $($Result.HttpStatus) $($Result.HttpReason) —— 请求被后端拒绝"
        if ($Result.Raw) {
            $previewLen = [Math]::Min(500, $Result.Raw.Length)
            Write-Info "响应体前 $previewLen 字符:"
            Write-Host "    $($Result.Raw.Substring(0, $previewLen))" -ForegroundColor DarkYellow
        }
        return @{
            Events = @()
            ToolCalls = @()
            Errors = @($Result.Error)
            FinalAnswer = ""
            DoneStatus = ""
        }
    }

    # ── 空响应诊断 ──
    if (-not $Result.Raw -or $Result.Raw.Length -lt 10) {
        Write-Warn "响应为空或过短 (len=$($Result.Raw.Length))"
        if ($Result.Error) {
            Write-Fail "错误: $($Result.Error)"
        }
    }

    $events = Parse-SseEvents $Result.Raw
    Write-Info "SSE 事件数: $($events.Count)"

    if ($events.Count -eq 0 -and $Result.Raw.Length -gt 0) {
        $previewLen = [Math]::Min(500, $Result.Raw.Length)
        Write-Warn "有响应但未解析出 SSE 事件，前 $previewLen 字符:"
        Write-Host "    $($Result.Raw.Substring(0, $previewLen))" -ForegroundColor DarkYellow
    }

    $toolCalls = @()
    $errors = @()
    $finalAnswer = ""
    $doneStatus = ""

    foreach ($ev in $events) {
        switch ($ev.type) {
            "task" {
                Write-Info "task_id=$($ev.task_id)"
            }
            "step" {
                $step = $ev.step
                if ($step) {
                    if ($step.kind -eq "action") {
                        $toolCalls += $step.label
                    } elseif ($step.kind -eq "answer") {
                        $finalAnswer = $step.log
                    }
                }
            }
            "error" {
                $errors += $ev.message
            }
            "done" {
                $doneStatus = $ev.task_status
            }
        }
    }

    Write-Host "  耗时: $([Math]::Round($Result.Duration,1))s" -ForegroundColor Gray

    if ($toolCalls.Count -gt 0) {
        Write-Host "  工具调用 ($($toolCalls.Count)):" -ForegroundColor White
        $i = 1
        foreach ($t in $toolCalls) {
            Write-Host "    $i. $t" -ForegroundColor Gray
            $i++
        }
    } else {
        Write-Warn "无工具调用"
    }

    if ($errors.Count -gt 0) {
        Write-Fail "错误 ($($errors.Count)):"
        foreach ($e in $errors) {
            Write-Host "    - $e" -ForegroundColor Red
        }
    }

    if ($finalAnswer) {
        $preview = $finalAnswer
        if ($preview.Length -gt 200) {
            $preview = $preview.Substring(0, 200) + "..."
        }
        Write-Host "  最终回答:" -ForegroundColor White
        Write-Host "    $preview" -ForegroundColor Gray
    }

    return @{
        Events = $events
        ToolCalls = $toolCalls
        Errors = $errors
        FinalAnswer = $finalAnswer
        DoneStatus = $doneStatus
    }
}

# ============================================================
# 用例包装器
# ============================================================

function Run-Case {
    param(
        [int]$CaseNum,
        [string]$CaseName,
        [scriptblock]$Body
    )

    if (-not (Should-Run $CaseNum)) {
        Write-Info "跳过用例 $CaseNum (未在 -Only 中)"
        return
    }

    Write-Section "$CaseNum. $CaseName"

    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $passed = $false
    $detail = ""
    $logFile = ""

    try {
        $result = & $Body
        $sw.Stop()
        if ($result -is [hashtable]) {
            $passed = [bool]$result.passed
            $detail = [string]$result.detail
            if ($result.log_file) { $logFile = $result.log_file }
        } else {
            $detail = "用例返回非法类型"
        }
    } catch {
        $sw.Stop()
        $passed = $false
        $detail = "用例异常: $($_.Exception.Message)"
        Write-Fail $detail
    }

    Add-Result "用例$CaseNum`: $CaseName" $passed $detail $sw.Elapsed.TotalSeconds $logFile

    if ($passed) {
        Write-Ok "[通过] $CaseName — $detail"
    } else {
        Write-Fail "[失败] $CaseName — $detail"
    }
}

# ============================================================
# 前置检查
# ============================================================

Write-Section "0. 前置检查"
Write-Info "输出目录: $OutputRoot"
Write-Info "Base: $Base"
Write-Info "DataDir: $DataDir"

function Get-JsonUrl {
    param([string]$Url, [int]$Timeout = 5)
    try {
        $handler = New-Object System.Net.Http.HttpClientHandler
        $handler.UseProxy = $false
        $client = New-Object System.Net.Http.HttpClient($handler)
        $client.Timeout = [TimeSpan]::FromSeconds($Timeout)

        $resp = $client.GetAsync($Url).GetAwaiter().GetResult()
        $body = $resp.Content.ReadAsStringAsync().GetAwaiter().GetResult()

        $resp.Dispose()
        $client.Dispose()
        $handler.Dispose()

        return $body | ConvertFrom-Json
    } catch {
        return $null
    }
}

# 0.1 服务健康
Write-SubSection "0.1 服务健康检查"
$health = Get-JsonUrl "$Base/"
if ($health -and $health.status -eq "ok") {
    Write-Ok "服务运行中"
    Add-Result "0.1 服务健康检查" $true "ok"
} else {
    Write-Fail "服务不可达或响应异常"
    Add-Result "0.1 服务健康检查" $false "不可达"
}

# 0.2 模块状态
Write-SubSection "0.2 惰性模块预加载状态"
$mods = Get-JsonUrl "$Base/api/modules/status"
if ($mods) {
    $summary = $mods.summary
    Write-Info "total=$($summary.total) loaded=$($summary.loaded) loading=$($summary.loading) failed=$($summary.failed)"
    if ($summary.all_ready) {
        Write-Ok "所有模块已就绪"
        Add-Result "0.2 模块预加载" $true "全部就绪"
    } else {
        Write-Warn "有模块未就绪"
        Add-Result "0.2 模块预加载" $false "未就绪"
    }
} else {
    Write-Warn "模块状态查询失败"
    Add-Result "0.2 模块预加载" $false "查询失败"
}

# 0.3 MCP 状态
Write-SubSection "0.3 MCP 服务状态"
$mcp = Get-JsonUrl "$Base/api/mcp/status"
if ($mcp) {
    $fs = $mcp.servers | Where-Object { $_.name -eq "filesystem" }
    if ($fs -and $fs.status -eq "connected") {
        Write-Ok "filesystem MCP 已连接，工具数: $($fs.tools)"
        Add-Result "0.3 MCP 状态" $true "$($fs.tools) tools"
    } else {
        Write-Fail "filesystem MCP 未连接"
        Add-Result "0.3 MCP 状态" $false "未连接"
    }
} else {
    Write-Fail "MCP 状态查询失败"
    Add-Result "0.3 MCP 状态" $false "查询失败"
}

# 0.4 RAG 状态
Write-SubSection "0.4 RAG 知识库状态"
$rag = Get-JsonUrl "$Base/api/rag/stats"
if ($rag) {
    Write-Info "files=$($rag.total_files) chunks=$($rag.total_chunks)"
    Add-Result "0.4 RAG 状态" $true "files=$($rag.total_files) chunks=$($rag.total_chunks)"
} else {
    Write-Warn "RAG 状态查询失败"
    Add-Result "0.4 RAG 状态" $false "查询失败"
}

# 0.5 准备测试文件
if (-not $SkipPrepare) {
    Write-SubSection "0.5 准备测试文件"
    try {
        $dataPath = $DataDir -replace "/", "\"

        if (-not (Test-Path $dataPath)) {
            New-Item -ItemType Directory -Force -Path $dataPath | Out-Null
            Write-Ok "创建目录: $dataPath"
        } else {
            Write-Info "目录已存在: $dataPath"
        }

        # 清理旧测试文件
        $testFiles = @(
            "hello.txt", "notes.txt", "people.csv", "config.json",
            "greeting.txt", "stats.txt", "combined.txt",
            "step1.txt", "step2.txt", "step3.txt", "summary.txt",
            "multi.txt", "output.txt"
        )
        foreach ($f in $testFiles) {
            $p = Join-Path $dataPath $f
            if (Test-Path $p) { Remove-Item -Force $p -ErrorAction SilentlyContinue }
        }
        foreach ($d in @("subdir", "newdir", "test_output")) {
            $subdir = Join-Path $dataPath $d
            if (Test-Path $subdir) {
                Remove-Item -Recurse -Force $subdir -ErrorAction SilentlyContinue
            }
        }

        $utf8NoBom = New-Object System.Text.UTF8Encoding $false

        [System.IO.File]::WriteAllText(
            (Join-Path $dataPath "hello.txt"),
            "Hello MCP filesystem test",
            $utf8NoBom
        )
        [System.IO.File]::WriteAllText(
            (Join-Path $dataPath "notes.txt"),
            "第一行：测试内容`r`n第二行：Agent 文件操作`r`n第三行：完成`r`n",
            $utf8NoBom
        )
        [System.IO.File]::WriteAllText(
            (Join-Path $dataPath "people.csv"),
            "name,age,city`r`nAlice,30,Beijing`r`nBob,25,Shanghai`r`nCharlie,35,Guangzhou`r`n",
            $utf8NoBom
        )
        [System.IO.File]::WriteAllText(
            (Join-Path $dataPath "config.json"),
            '{"key": "value", "count": 42, "tags": ["a", "b"]}',
            $utf8NoBom
        )

        New-Item -ItemType Directory -Force -Path (Join-Path $dataPath "subdir") | Out-Null
        [System.IO.File]::WriteAllText(
            (Join-Path $dataPath "subdir\nested.txt"),
            "nested content from subdir",
            $utf8NoBom
        )

        Write-Ok "测试文件已准备"
        Add-Result "0.5 准备测试文件" $true "OK"
    } catch {
        Write-Fail "准备测试文件失败: $_"
        Add-Result "0.5 准备测试文件" $false "$_"
    }
}

# ============================================================
# 用例定义
# ============================================================

$D = $DataDir
$DataPath = $D -replace "/", "\"

# ── 用例 1：列出目录 ──
Run-Case 1 "列出目录" {
    $r = Invoke-AgentTask -Message "列出 $D 目录下的所有文件" -LogName "case01_list_directory"
    $s = Show-TaskSummary $r "case01"

    $matched = $s.ToolCalls | Where-Object { $_ -match "list_directory" }
    if (-not $matched) {
        return @{ passed = $false; detail = "未调用 list_directory (tools=$($s.ToolCalls.Count) err=$($s.Errors.Count))"; log_file = $r.LogFile }
    }
    return @{ passed = $true; detail = "$($matched[0])"; log_file = $r.LogFile }
}

# ── 用例 2：读取文件 ──
Run-Case 2 "读取文件" {
    $r = Invoke-AgentTask -Message "读取 $D/hello.txt 的内容" -LogName "case02_read_file"
    $s = Show-TaskSummary $r "case02"

    $matched = $s.ToolCalls | Where-Object { $_ -match "read_text_file|read_file" }
    if (-not $matched) {
        return @{ passed = $false; detail = "未调用读取工具"; log_file = $r.LogFile }
    }
    $hasContent = ($s.FinalAnswer -match "Hello|MCP")
    return @{
        passed = $hasContent
        detail = "调用=$($matched[0]) 含内容=$hasContent"
        log_file = $r.LogFile
    }
}

# ── 用例 3：写入文件 ──
Run-Case 3 "写入文件" {
    $r = Invoke-AgentTask -Message "在 $D/ 创建新文件 greeting.txt，内容为 'Hello from Agent'" -LogName "case03_write_file"
    $s = Show-TaskSummary $r "case03"
    Start-Sleep -Milliseconds 500

    $filePath = Join-Path $DataPath "greeting.txt"
    $exists = Test-Path $filePath -PathType Leaf
    $contentOk = $false
    if ($exists) {
        $content = Get-Content $filePath -Raw
        $contentOk = $content -match "Hello from Agent"
    }
    return @{
        passed = $exists -and $contentOk
        detail = "exists=$exists content_ok=$contentOk"
        log_file = $r.LogFile
    }
}

# ── 用例 4：目录树 ──
Run-Case 4 "目录树" {
    $r = Invoke-AgentTask -Message "显示 $D 的完整目录树" -LogName "case04_directory_tree"
    $s = Show-TaskSummary $r "case04"

    $matched = $s.ToolCalls | Where-Object { $_ -match "directory_tree" }
    if (-not $matched) {
        return @{ passed = $false; detail = "未调用 directory_tree"; log_file = $r.LogFile }
    }
    return @{ passed = $true; detail = "$($matched[0])"; log_file = $r.LogFile }
}

# ── 用例 5：文件信息 ──
Run-Case 5 "文件信息" {
    $r = Invoke-AgentTask -Message "查看 $D/people.csv 的文件信息" -LogName "case05_get_file_info"
    $s = Show-TaskSummary $r "case05"

    $matched = $s.ToolCalls | Where-Object { $_ -match "get_file_info" }
    if (-not $matched) {
        return @{ passed = $false; detail = "未调用 get_file_info"; log_file = $r.LogFile }
    }
    return @{ passed = $true; detail = "$($matched[0])"; log_file = $r.LogFile }
}

# ── 用例 6：搜索文件 ──
Run-Case 6 "搜索文件" {
    $r = Invoke-AgentTask -Message "在 $D 下搜索内容包含 'Agent' 的文件" -LogName "case06_search_files"
    $s = Show-TaskSummary $r "case06"

    $matched = $s.ToolCalls | Where-Object { $_ -match "search_files" }
    if (-not $matched) {
        return @{ passed = $false; detail = "未调用 search_files"; log_file = $r.LogFile }
    }
    return @{ passed = $true; detail = "$($matched[0])"; log_file = $r.LogFile }
}

# ── 用例 7：读取多个文件 ──
Run-Case 7 "读取多个文件" {
    $r = Invoke-AgentTask -Message "同时读取 $D/hello.txt 和 $D/notes.txt 的内容" -LogName "case07_read_multiple_files"
    $s = Show-TaskSummary $r "case07"

    $multi = $s.ToolCalls | Where-Object { $_ -match "read_multiple_files" }
    if ($multi) {
        return @{ passed = $true; detail = "read_multiple_files"; log_file = $r.LogFile }
    }
    $readCount = ($s.ToolCalls | Where-Object { $_ -match "read_text_file|read_file" }).Count
    if ($readCount -ge 2) {
        return @{ passed = $true; detail = "$readCount 次单文件读取"; log_file = $r.LogFile }
    }
    return @{ passed = $false; detail = "无有效读取 (readCount=$readCount)"; log_file = $r.LogFile }
}

# ── 用例 8：创建目录 ──
Run-Case 8 "创建目录" {
    $r = Invoke-AgentTask -Message "在 $D 下创建名为 'newdir' 的空目录" -LogName "case08_create_directory"
    $s = Show-TaskSummary $r "case08"
    Start-Sleep -Milliseconds 300

    $dirPath = Join-Path $DataPath "newdir"
    $exists = Test-Path $dirPath -PathType Container
    return @{
        passed = $exists
        detail = "目录存在=$exists"
        log_file = $r.LogFile
    }
}

# ── 用例 9：移动文件 ──
Run-Case 9 "移动文件" {
    $r = Invoke-AgentTask -Message "把 $D/greeting.txt 移动到 $D/subdir/moved_greeting.txt" -LogName "case09_move_file"
    $s = Show-TaskSummary $r "case09"
    Start-Sleep -Milliseconds 300

    $targetPath = Join-Path $DataPath "subdir\moved_greeting.txt"
    $exists = Test-Path $targetPath -PathType Leaf
    return @{
        passed = $exists
        detail = "目标文件存在=$exists"
        log_file = $r.LogFile
    }
}

# ── 用例 10：编辑文件 ──
Run-Case 10 "编辑文件" {
    $utf8NoBom = New-Object System.Text.UTF8Encoding $false
    [System.IO.File]::WriteAllText(
        (Join-Path $DataPath "hello.txt"),
        "Hello test content",
        $utf8NoBom
    )

    $r = Invoke-AgentTask -Message "把 $D/hello.txt 里的 'test' 改成 'production'" -LogName "case10_edit_file"
    $s = Show-TaskSummary $r "case10"
    Start-Sleep -Milliseconds 300

    $filePath = Join-Path $DataPath "hello.txt"
    $content = if (Test-Path $filePath) { Get-Content $filePath -Raw } else { "" }
    $hasProduction = $content -match "production"
    return @{
        passed = $hasProduction
        detail = "内容含 production=$hasProduction"
        log_file = $r.LogFile
    }
}

# ── 用例 11：复合（列出 + 读取 + 计数） ──
Run-Case 11 "复合指令：列出目录 + 读取文件 + 计数" {
    $msg = @"
请完成以下任务：
1. 列出 $D 目录下的所有文件
2. 读取 $D/people.csv 的内容
3. 告诉我 people.csv 里有几个人
"@
    $r = Invoke-AgentTask -Message $msg -LogName "case11_composite_list_read_count"
    $s = Show-TaskSummary $r "case11"

    $hasList = ($s.ToolCalls | Where-Object { $_ -match "list_directory" }).Count -gt 0
    $hasRead = ($s.ToolCalls | Where-Object { $_ -match "read_text_file|read_file" }).Count -gt 0
    $hasCount = ($s.FinalAnswer -match "\b3\b|三")
    Write-Host "  检查: 列出=$hasList 读取=$hasRead 计数=$hasCount" -ForegroundColor Gray
    return @{
        passed = $hasList -and $hasRead -and $hasCount
        detail = "list=$hasList read=$hasRead count=$hasCount"
        log_file = $r.LogFile
    }
}

# ── 用例 12：复合（多读 + 总结） ──
Run-Case 12 "复合指令：多文件读取 + 总结" {
    $msg = @"
请依次完成：
1. 读取 $D/hello.txt
2. 读取 $D/notes.txt
3. 读取 $D/config.json
4. 用一段话总结这三个文件分别是什么内容
"@
    $r = Invoke-AgentTask -Message $msg -LogName "case12_composite_multi_read_summary"
    $s = Show-TaskSummary $r "case12"

    $readCount = ($s.ToolCalls | Where-Object { $_ -match "read" }).Count
    $hasSummary = $s.FinalAnswer.Length -gt 50
    Write-Host "  检查: 读取调用=$readCount 有总结=$hasSummary" -ForegroundColor Gray
    return @{
        passed = ($readCount -ge 2) -and $hasSummary
        detail = "readCalls=$readCount summary=$hasSummary"
        log_file = $r.LogFile
    }
}

# ── 用例 13：复合（读 → 写 → 验证） ──
Run-Case 13 "复合指令：读取 → 处理 → 写入 → 验证" {
    $msg = @"
请按顺序完成：
1. 读取 $D/people.csv，统计有多少行数据（不含表头）
2. 把统计结果写入 $D/stats.txt，内容格式为 '共 N 行数据'
3. 读取 $D/stats.txt 验证写入结果
"@
    $r = Invoke-AgentTask -Message $msg -LogName "case13_composite_read_write_verify"
    $s = Show-TaskSummary $r "case13"
    Start-Sleep -Milliseconds 500

    $hasRead = ($s.ToolCalls | Where-Object { $_ -match "read" }).Count -ge 1
    $hasWrite = ($s.ToolCalls | Where-Object { $_ -match "write_file" }).Count -ge 1

    $statsPath = Join-Path $DataPath "stats.txt"
    $statsExists = Test-Path $statsPath -PathType Leaf
    $statsContent = if ($statsExists) { Get-Content $statsPath -Raw } else { "" }
    $statsOk = $statsContent -match "\b3\b|三"

    Write-Host "  检查: 读取=$hasRead 写入=$hasWrite 文件=$statsExists 内容含3=$statsOk" -ForegroundColor Gray
    return @{
        passed = $hasRead -and $hasWrite -and $statsExists -and $statsOk
        detail = "read=$hasRead write=$hasWrite file=$statsExists content_ok=$statsOk"
        log_file = $r.LogFile
    }
}

# ── 用例 14：边界（不存在的目录） ──
Run-Case 14 "边界：目录不存在" {
    $r = Invoke-AgentTask -Message "列出 $D/nonexistent_dir_xyz 目录下的文件" -LogName "case14_nonexistent_dir"
    $s = Show-TaskSummary $r "case14"

    $graceful = $s.FinalAnswer.Length -gt 0
    return @{
        passed = $graceful
        detail = "有最终回答=$graceful (toolCalls=$($s.ToolCalls.Count) errors=$($s.Errors.Count))"
        log_file = $r.LogFile
    }
}

# ── 用例 15：边界（无 RAG 关键词） ──
Run-Case 15 "边界：普通问候不触发 RAG" {
    $r = Invoke-AgentTask -Message "你好，介绍一下你自己" -LogName "case15_no_rag_keyword"
    $s = Show-TaskSummary $r "case15"

    $hasRagCall = ($s.ToolCalls | Where-Object { $_ -match "knowledge|rag" }).Count -gt 0
    return @{
        passed = -not $hasRagCall
        detail = "rag_called=$hasRagCall"
        log_file = $r.LogFile
    }
}

# ── 用例 16：边界（有 RAG 关键词） ──
Run-Case 16 "边界：知识库相关提问应触发 RAG" {
    $r = Invoke-AgentTask -Message "根据文档资料，实验指导书讲了什么" -LogName "case16_rag_keyword"
    $s = Show-TaskSummary $r "case16"

    $hasRagHint = $false
    foreach ($ev in $s.Events) {
        if ($ev.type -eq "step" -and $ev.step.kind -eq "resource") {
            $hasRagHint = $true
            break
        }
        if ($ev.type -eq "step" -and $ev.step.label -match "知识库|资源") {
            $hasRagHint = $true
            break
        }
    }
    $hasAnswer = $s.FinalAnswer.Length -gt 20
    return @{
        passed = $hasRagHint -or $hasAnswer
        detail = "rag_hint=$hasRagHint has_answer=$hasAnswer"
        log_file = $r.LogFile
    }
}

# ============================================================
# 汇总
# ============================================================

Write-Section "测试汇总"

$passed = ($script:Results | Where-Object { $_.passed }).Count
$failed = ($script:Results | Where-Object { -not $_.passed }).Count
$total = $script:Results.Count

Write-Host ""
$color = if ($failed -eq 0) { "Green" } else { "Yellow" }
Write-Host "  总计: $total  通过: $passed  失败: $failed" -ForegroundColor $color
Write-Host ""

foreach ($r in $script:Results) {
    $icon = if ($r.passed) { "[PASS]" } else { "[FAIL]" }
    $c = if ($r.passed) { "Green" } else { "Red" }
    Write-Host "  $icon $($r.name)" -ForegroundColor $c
    Write-Host "       $($r.detail)  ($($r.duration_sec)s)" -ForegroundColor Gray
}

Write-Host ""
if ($failed -eq 0) {
    Write-Host "  All tests passed!" -ForegroundColor Green
} else {
    Write-Host "  $failed test(s) failed." -ForegroundColor Yellow
}

# ── JSON 汇总 ──
try {
    $summaryObj = [PSCustomObject]@{
        timestamp = (Get-Date).ToString("o")
        base_url = $Base
        data_dir = $DataDir
        total = $total
        passed = $passed
        failed = $failed
        results = $script:Results
    }
    $summaryObj | ConvertTo-Json -Depth 5 | Out-File -Encoding utf8 $SummaryFile
    Write-Info "JSON 汇总: $SummaryFile"
} catch {
    Write-Warn "保存 JSON 汇总失败: $_"
}

# ── TXT 汇总 ──
try {
    $lines = @()
    $lines += "测试时间: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    $lines += "Base URL: $Base"
    $lines += "Data Dir: $DataDir"
    $lines += ""
    $lines += "总计: $total  通过: $passed  失败: $failed"
    $lines += ""
    foreach ($r in $script:Results) {
        $icon = if ($r.passed) { "[PASS]" } else { "[FAIL]" }
        $lines += "$icon $($r.name) ($($r.duration_sec)s)"
        $lines += "       $($r.detail)"
        if ($r.log_file) {
            $lines += "       log: $($r.log_file)"
        }
    }
    $lines -join "`r`n" | Out-File -Encoding utf8 $SummaryTxtFile
    Write-Info "TXT 汇总: $SummaryTxtFile"
} catch {
    Write-Warn "保存 TXT 汇总失败: $_"
}

Write-Info "日志目录: $LogDir"
Write-Host ""