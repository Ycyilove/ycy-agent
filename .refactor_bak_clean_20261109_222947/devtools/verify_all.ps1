cd D:\Projects\111\LLM\LLM
$ErrorActionPreference = "Continue"

Write-Host "=== 1. import ===" -ForegroundColor Cyan
python -c "from tools.semantics import infer_essential_when; print('infer_essential_when OK')"

Write-Host "`n=== 2. manager infer ===" -ForegroundColor Cyan
if (Test-Path test_manager_infer.py) { python test_manager_infer.py } else { Write-Host "test_manager_infer.py missing" -ForegroundColor Yellow }

Write-Host "`n=== 3. selection ===" -ForegroundColor Cyan
python test_selection.py 2>&1 | Select-String -Pattern "PASS|FAIL|summary|pos=|essential_when 新增" | ForEach-Object { $_.Line }

Write-Host "`n=== 4. essential values ===" -ForegroundColor Cyan
python -c "
from tools.semantics import infer_essential_when
cases = [
    ('create_pptx', 'Create a PowerPoint presentation', ('generate','presentation','ppt','pptx')),
    ('render_document', 'Render a document to PDF', ('generate','pdf','document')),
    ('read_pdf', 'Read a PDF file', ('read','pdf','document')),
    ('convert_to_pdf', 'Convert docx to PDF', ('convert','pdf')),
    ('mcp__foo__create_slides', 'Create presentation slides', ('generate','presentation')),
    ('mcp__bar__make_excel', 'Make an Excel file', ()),
]
for name, desc, caps in cases:
    print(f'{name}: {infer_essential_when(name, desc, caps)}')
"
python -c "from tools.semantics import infer_essential_when; print(infer_essential_when('tavily_search', 'Search the web for current information on any topic', ()))"

Write-Host "`n=== 5. backend import ===" -ForegroundColor Cyan
python -c "import backend.app; print('backend OK')"