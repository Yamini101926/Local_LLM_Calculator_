# Local LLM Calculator

A command-line calculator that uses a local Ollama language model to understand a request and propose a mathematical expression. A restricted Python AST calculator validates and evaluates that expression; the model never supplies the numeric answer. No `eval()`, cloud API, or third-party Python package is used.

## Requirements

- Windows 10/11 and Python 3.10 or newer. Install Python from [python.org](https://www.python.org/downloads/windows/) and select **Add python.exe to PATH**, or use the Windows `py` launcher.
- [Ollama for Windows](https://ollama.com/download).
- A chat model downloaded in Ollama. The default is `qwen2.5:3b` (allow several GB of disk space and RAM). You can select another compatible local model with `OLLAMA_MODEL`.
- Python dependencies: none; standard library only.

## Set up in PowerShell

Open PowerShell in this project folder, then run:

```powershell
py --version
py -3 main.py "What is the square root of 144?"
```

If `py` is not recognized, install Python 3.10+ and enable the Python launcher, then reopen PowerShell. If the launcher exists but reports that no Python is installed, install Python from python.org and select the launcher option.

Install and start Ollama, then download the default model once:

```powershell
ollama pull qwen2.5:3b
ollama serve
```

Keep Ollama running in that window (or use the Ollama desktop app), open another PowerShell window in the project folder, and run:

```powershell
py -3 main.py "What is 25 percent of 360?"
```

The first request loads the model and may take a little while. Ollama's local chat endpoint is `http://localhost:11434` by default. Optional model/host overrides for the current PowerShell session:

```powershell
$env:OLLAMA_MODEL = "qwen2.5:3b"
$env:OLLAMA_HOST = "http://localhost:11434"
py -3 main.py "average of 4, 8, 12, and 16"
```

## Example requests

```powershell
py -3 main.py "square root of 81"
py -3 main.py "What is 2 to the power of 10?"
py -3 main.py "15 percent of 240"
py -3 main.py "average of 12, 15, 18, 21"
py -3 main.py "factorial of 7"
py -3 main.py "sine of 30 degrees"
py -3 main.py "natural logarithm of 10"
py -3 main.py "(5 + 3) * 2 squared"
py -3 main.py "convert 5 km to miles"
py -3 main.py "convert 32 Fahrenheit to Celsius"
```

The expression calculator also supports trigonometric and inverse-trigonometric functions, hyperbolic functions, logarithms, rounding, combinations/permutations, GCD/LCM, means, medians, sample and population variance/standard deviation, and common length/mass/volume and temperature conversions. Use radians for direct `sin`, `cos`, and `tan` expressions unless your prompt specifies degrees.

## How it works

1. The program sends the request to Ollama's local `/api/chat` endpoint and asks the LLM to return an expression and a short public status.
2. The expression is independently parsed against a strict AST allowlist and computed by Python's deterministic math functions.
3. A local natural-language translator handles common phrasing if the model is unavailable or returns unusable output. Unsupported or ambiguous requests produce an error instead of executing arbitrary code.

The model output is treated as untrusted input. Attribute access, imports, strings, comprehensions, arbitrary names, and unapproved calls are rejected. Exponents, expression length, AST size, and result magnitude are bounded. Trigonometric outputs may have ordinary floating-point rounding.

## Troubleshooting

- **`py` is not recognized:** install Python for Windows and enable the Python launcher, or try `python --version` and replace `py -3` with `python` in the commands.
- **`No installed Python found`:** install Python 3.10+; the launcher is present but has no registered interpreter.
- **Ollama connection error:** start the Ollama app or run `ollama serve`. Check that the host is reachable at `http://localhost:11434`.
- **Model not found:** run `ollama pull qwen2.5:3b`, or set `$env:OLLAMA_MODEL` to a model already downloaded by Ollama.
- **Model response cannot be used:** the local translator handles the supported common expressions; rephrase an unsupported request or use explicit syntax such as `sqrt(81)`, `mean(2, 4, 6)`, and `sin(radians(30))`.
- **Out-of-domain / invalid calculation:** verify the units and domain (for example, real square roots need nonnegative inputs); complex results are not supported.

## Files

- `main.py`: Ollama client, natural-language translator, validated AST calculator, and CLI.
- `requirements.txt`: empty dependency list; no package installation is needed.
- `.gitignore`: ignores Python bytecode, virtual environments, and local notes/cache files.
