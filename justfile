# SAM3 Gradio App Development Justfile
# Usage: just <recipe>
# Run `just --list` to see all available recipes

# Default recipe (show help)
default:
    @just --list

# ========================================
# Setup and Environment
# ========================================

# Sync dependencies with uv
sync:
    uv sync

# Sync development dependencies
sync-dev:
    uv sync --extra dev

# Sync notebook dependencies
sync-notebooks:
    uv sync --extra notebooks

# Sync all dependencies
sync-all:
    uv sync --extra dev --extra notebooks

# Check environment setup
check-env:
    @echo "=== Environment Check ==="
    uv run python --version
    uv run python -c "import torch; print(f'PyTorch: {torch.__version__}')"
    uv run python -c "import gradio; print(f'Gradio: {gradio.__version__}')"
    uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
    @echo "=== Model Check ==="
    @if [ -f models/sam3.pt ]; then echo "✅ models/sam3.pt exists"; else echo "❌ models/sam3.pt NOT FOUND"; fi

# ========================================
# Testing
# ========================================

# Run all tests
test:
    uv run pytest tests/ -v

# Run tests with coverage
test-cov:
    uv run pytest tests/ -v --cov=sam3 --cov-report=term-missing

# Run gradio app tests only
test-gradio:
    uv run pytest tests/gradio_app/ -v

# Run gradio app tests with coverage
test-gradio-cov:
    uv run pytest tests/gradio_app/ -v --cov=sam3/gradio_app --cov-report=html

# Run tests excluding slow tests
test-fast:
    uv run pytest tests/ -v -m "not slow"

# Run only slow tests
test-slow:
    uv run pytest tests/ -v -m slow

# Run specific test file
test-file FILE:
    uv run pytest {{FILE}} -v

# Run specific test function
test-func FILE FUNC:
    uv run pytest {{FILE}}::{{FUNC}} -vv -s

# ========================================
# Code Quality
# ========================================

# Format code with black
format:
    uv run black sam3/gradio_app/ tests/gradio_app/ --line-length 88

# Check code formatting without modifying
format-check:
    uv run black sam3/gradio_app/ tests/gradio_app/ --check --line-length 88

# Lint code with ruff
lint:
    uv run ruff check sam3/gradio_app/ tests/gradio_app/

# Fix linting issues automatically
lint-fix:
    uv run ruff check sam3/gradio_app/ tests/gradio_app/ --fix

# Type check with mypy
typecheck:
    uv run mypy sam3/gradio_app/

# Run all quality checks
quality: format-check lint typecheck
    @echo "✅ All quality checks passed"

# Fix all auto-fixable issues
fix: format lint-fix
    @echo "✅ Auto-fixes applied"

# ========================================
# Gradio App
# ========================================

# Run Gradio app (development mode)
run:
    uv run python -m scripts.gradio_app.sam3_gradio_app

# Run Gradio app on specific port
run-port PORT:
    uv run python -m scripts.gradio_app.sam3_gradio_app --server-port {{PORT}}

# Run Gradio app in background
run-bg:
    uv run python -m scripts.gradio_app.sam3_gradio_app > temp/app.log 2>&1 &
    @echo "App started in background. Check temp/app.log for logs."

# Stop background Gradio app
stop:
    @pkill -f "sam3_gradio_app" || echo "No running app found"

# Restart Gradio app
restart: stop
    @sleep 2
    @just run-bg
    @echo "✅ App restarted"

# Restart and check status
restart-check: restart
    @sleep 5
    @just status

# View app logs (tail)
logs:
    tail -f temp/app.log

# View error logs only
logs-error:
    @grep -E "(ERROR|Failed|Exception)" temp/app.log || echo "No errors found"

# Check if app is running
status:
    @curl -s http://localhost:7860 > /dev/null && echo "✅ App is running on http://localhost:7860" || echo "❌ App is not running"

# Quick health check (status + recent errors)
health:
    @echo "=== App Status ==="
    @just status
    @echo "\n=== Recent Errors (last 5) ==="
    @tail -100 temp/app.log 2>/dev/null | grep -E "(ERROR|Failed|Exception)" | tail -5 || echo "No recent errors"

# ========================================
# Development Utilities
# ========================================

# Create necessary directories
init-dirs:
    mkdir -p temp/gradio_outputs
    mkdir -p temp/test_data
    mkdir -p models
    @echo "✅ Directories created"

# Generate mini test data from assets
gen-test-data:
    uv run python -c "from PIL import Image; img = Image.open('assets/images/test_image.jpg'); img.resize((256, 256)).save('temp/test_data/mini_test_image.jpg')"
    @echo "✅ Mini test data generated at temp/test_data/mini_test_image.jpg"

# Clean temporary files
clean:
    rm -rf temp/gradio_outputs/*
    rm -rf temp/*.log
    rm -rf .pytest_cache
    rm -rf htmlcov
    rm -rf .coverage
    @echo "✅ Temporary files cleaned"

# Clean all generated files (including test data)
clean-all: clean
    rm -rf temp/test_data/*
    rm -rf __pycache__
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    @echo "✅ All generated files cleaned"

# Show current date and time (for logging)
timestamp:
    date "+%Y-%m-%d %H:%M:%S %Z%z"

# ========================================
# Playwright Testing
# ========================================

# Start app for Playwright MCP testing
playwright-start:
    @just restart-check
    @echo "✅ App ready for Playwright MCP testing at http://localhost:7860"

# Stop app after Playwright testing
playwright-stop:
    @just stop
    @echo "✅ Playwright testing session ended"

# Run Playwright tests (requires app to be running)
test-playwright:
    @echo "Starting Gradio app for Playwright testing..."
    @just run-bg
    @sleep 10
    @echo "Running Playwright tests..."
    # Playwright tests will be executed via MCP
    @echo "Note: Use Playwright MCP tool to interact with the app"
    @just stop

# ========================================
# Work Log
# ========================================

# Add entry to work log
log ENTRY:
    @echo "| $(date '+%Y-%m-%d') | $(date '+%H:%M:%S %Z%z') | $(whoami) | {{ENTRY}} | |" >> temp/work_plan_gradio_app.md

# View work log
show-log:
    tail -20 temp/work_plan_gradio_app.md

# ========================================
# Complete Workflows
# ========================================

# Run complete development cycle
dev: sync-dev init-dirs format lint test-fast
    @echo "✅ Development cycle complete"

# Run complete CI/CD pipeline
ci: sync-dev format-check lint typecheck test
    @echo "✅ CI pipeline complete"

# Prepare for deployment
deploy: sync quality test
    @echo "✅ Ready for deployment"
