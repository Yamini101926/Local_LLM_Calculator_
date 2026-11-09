"""Local LLM assisted calculator. Math is always evaluated by the safe AST tool."""
from __future__ import annotations

import ast
import json
import math
import operator
import os
import re
import sys
import urllib.error
import urllib.request
from typing import Any

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
MAX_EXPRESSION_LENGTH = 300
MAX_AST_NODES = 80
MAX_RESULT = 1e100

FUNCTIONS = {
    "abs": abs, "round": round, "min": min, "max": max,
    "sqrt": math.sqrt, "cbrt": lambda x: math.copysign(abs(x) ** (1 / 3), x),
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan,
    "sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh,
    "log": math.log, "log10": math.log10, "log2": math.log2, "exp": math.exp,
    "floor": math.floor, "ceil": math.ceil, "trunc": math.trunc,
    "degrees": math.degrees, "radians": math.radians,
    "factorial": math.factorial, "gcd": math.gcd, "lcm": math.lcm,
    "comb": math.comb, "perm": math.perm, "hypot": math.hypot,
    "mean": lambda *xs: sum(xs) / len(xs),
    "median": lambda *xs: __import__("statistics").median(xs),
    "stdev": lambda *xs: __import__("statistics").stdev(xs),
    "variance": lambda *xs: __import__("statistics").variance(xs),
    "pstdev": lambda *xs: __import__("statistics").pstdev(xs),
    "pvariance": lambda *xs: __import__("statistics").pvariance(xs),
}
CONSTANTS = {"pi": math.pi, "e": math.e, "tau": math.tau}
BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
          ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
          ast.Mod: operator.mod, ast.Pow: operator.pow}
UNOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def calculate(expression: str) -> int | float:
    """Evaluate a deliberately small Python-expression grammar without eval()."""
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("Expression must be non-empty text.")
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise ValueError("Expression is too long.")
    expression = expression.replace("^", "**")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError("I couldn't form a valid math expression.") from exc
    if sum(1 for _ in ast.walk(tree)) > MAX_AST_NODES:
        raise ValueError("Expression is too complex.")

    def checked(value: Any) -> int | float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Only real numeric results are supported.")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Result is not finite.")
        if abs(value) > MAX_RESULT:
            raise ValueError("Result is too large.")
        return value

    def visit(node: ast.AST) -> int | float:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return checked(node.value)
        if isinstance(node, ast.Name):
            if node.id not in CONSTANTS:
                raise ValueError(f"Unknown constant: {node.id}")
            return CONSTANTS[node.id]
        if isinstance(node, ast.UnaryOp) and type(node.op) in UNOPS:
            return checked(UNOPS[type(node.op)](visit(node.operand)))
        if isinstance(node, ast.BinOp) and type(node.op) in BINOPS:
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("Power exponent must be between -100 and 100.")
            return checked(BINOPS[type(node.op)](left, right))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name = node.func.id
            if name not in FUNCTIONS or node.keywords:
                raise ValueError("That function is not allowed.")
            args = [visit(arg) for arg in node.args]
            if not args or len(args) > 20:
                raise ValueError(f"{name} needs between 1 and 20 values.")
            if name in ("mean", "median", "stdev", "variance", "pstdev", "pvariance") and len(args) < (2 if name in ("stdev", "variance") else 1):
                raise ValueError(f"{name} needs more values.")
            if name == "factorial" and (len(args) != 1 or type(args[0]) is not int or not 0 <= args[0] <= 170):
                raise ValueError("Factorial needs one integer from 0 to 170.")
            if name in ("comb", "perm", "gcd", "lcm") and any(type(x) is not int for x in args):
                raise ValueError(f"{name} needs integer arguments.")
            if name in ("comb", "perm") and (len(args) != 2 or any(x < 0 or x > 10000 for x in args)):
                raise ValueError(f"{name} needs two integers from 0 to 10000.")
            if name in ("gcd", "lcm") and len(args) < 2:
                raise ValueError(f"{name} needs at least two integers.")
            result = FUNCTIONS[name](*args)
            return checked(result)
        raise ValueError("Only approved numbers, operators, constants and functions are allowed.")

    try:
        return checked(visit(tree))
    except (ArithmeticError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Calculation error: {exc}") from exc


def _number_words(text: str) -> str:
    """Convert common English number words; leave other language for the LLM."""
    units = {"zero":"0", "one":"1", "two":"2", "three":"3", "four":"4", "five":"5", "six":"6", "seven":"7", "eight":"8", "nine":"9", "ten":"10", "eleven":"11", "twelve":"12", "thirteen":"13", "fourteen":"14", "fifteen":"15", "sixteen":"16", "seventeen":"17", "eighteen":"18", "nineteen":"19", "twenty":"20", "thirty":"30", "forty":"40", "fifty":"50", "sixty":"60", "seventy":"70", "eighty":"80", "ninety":"90", "hundred":"100", "thousand":"1000"}
    for word, digit in units.items():
        text = re.sub(rf"\b{word}\b", digit, text, flags=re.I)
    return text


def normalize_request(text: str) -> str:
    """Turn common natural math language into an expression; never computes it."""
    s = _number_words(text.lower().replace("−", "-").replace("×", "*").replace("÷", "/").replace("π", " pi "))
    for fn, words in (("sin", "sine|sin"), ("cos", "cosine|cos"), ("tan", "tangent|tan")):
        s = re.sub(rf"\b(?:{words})\s+of\s*(-?\d+(?:\.\d+)?)\s*degrees\b", rf"{fn}(radians(\1))", s)
    s = re.sub(r"\b(what is|what's|calculate|compute|evaluate|find|please|tell me|the value of|result of)\b", " ", s)
    s = re.sub(r"\b(square root of|square root)\s*", "sqrt(", s)
    # Close a simple function wrapper before a trailing request phrase/operator.
    if "sqrt(" in s and not s.count("(") > s.count(")"):
        pass
    s = re.sub(r"\b(cube root of|cube root)\s*", "cbrt(", s)
    s = re.sub(r"\b(squared|to the power of two)\b", "**2", s)
    s = re.sub(r"\b(cubed|to the power of three)\b", "**3", s)
    s = re.sub(r"\b(to the power of|raised to)\b", "**", s)
    s = re.sub(r"\b(plus|added to|add)\b", "+", s)
    s = re.sub(r"\b(minus|subtract|subtracted from)\b", "-", s)
    s = re.sub(r"\b(times|multiplied by|multiply)\b", "*", s)
    s = re.sub(r"\b(divided by|divide by|over)\b", "/", s)
    s = re.sub(r"\b(modulo|mod)\b", "%", s)
    s = re.sub(r"\b(average|mean) of\b", "mean(", s)
    s = re.sub(r"\b(median) of\b", "median(", s)
    s = re.sub(r"\b(population standard deviation|population std dev) of\b", "pstdev(", s)
    s = re.sub(r"\b(standard deviation|std dev) of\b", "stdev(", s)
    s = re.sub(r"\bvariance of\b", "variance(", s)
    s = re.sub(r"\bfactorial of\s*", "factorial(", s)
    # Percent of a number, and percentage increase/decrease.
    s = re.sub(r"(\d+(?:\.\d+)?)\s*percent\s+of\s+(\d+(?:\.\d+)?)", r"(\1/100)*\2", s)
    s = re.sub(r"(\d+(?:\.\d+)?)\s*%\s+of\s+(\d+(?:\.\d+)?)", r"(\1/100)*\2", s)
    s = re.sub(r"(\d+(?:\.\d+)?)\s*percent", r"(\1/100)", s)
    s = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"(\1/100)", s)
    s = re.sub(r"\b(sine|sin) of\s*", "sin(", s)
    s = re.sub(r"\b(cosine|cos) of\s*", "cos(", s)
    s = re.sub(r"\b(tangent|tan) of\s*", "tan(", s)
    s = re.sub(r"\b(logarithm base 10 of|common logarithm of)\s*", "log10(", s)
    s = re.sub(r"\b(natural logarithm of|ln of)\s*", "log(", s)
    s = re.sub(r"\b(log base 2 of)\s*", "log2(", s)
    s = re.sub(r"\bof\b", "*", s)
    # Degrees are intuitive in worded trig prompts. Explicit radians() remains available.
    s = re.sub(r"\b(\d+(?:\.\d+)?)\s*degrees\b", r"radians(\1)", s)
    s = re.sub(r"\b(degrees|degree)\b", " ", s)
    s = re.sub(r"\bplus or minus\b", "+/-", s)
    s = re.sub(r"\b(and|then|is|equals?|please)\b", " ", s)
    s = s.strip().rstrip("?.!,;:")
    # List separators within statistical function calls.
    s = re.sub(r"(?<=\d)\s*,\s*(?=[-\d])", ",", s)
    s = re.sub(r",\s*and\s*(?=-?\d)", ",", s)
    s = re.sub(r"(?<=\d)\s+and\s+(?=[-\d])", ",", s)
    # Parenthesize simple unary-word transformations, e.g. sqrt(144).
    for name in ("sqrt", "cbrt", "factorial", "sin", "cos", "tan", "log", "log2", "log10"):
        s = re.sub(rf"\b{name}\(\s*(-?\d+(?:\.\d+)?|pi|e|tau)\s*\)?", rf"{name}(\1)", s)
    # Close one-argument functions followed by punctuation; multi-item stats consume the numeric list.
    s = re.sub(
        r"\b(mean|median|stdev|variance|pstdev|pvariance)\(([-\d.,\s]+)$",
        lambda m: m.group(1) + "(" + re.sub(r"\s+", "", m.group(2)) + ")",
        s,
    )
    # Remove harmless surrounding prose but retain only a safe math-like alphabet.
    s = re.sub(r"\b(percent|percentage|result|equals?)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_expression(text: str) -> str:
    """Use local language rules as a dependable expression fallback."""
    unit_match = re.search(
        r"(?:convert\s+)?(-?\d+(?:\.\d+)?)\s*(km|kilometers?|miles?|mi|meters?|metres?|cm|centimeters?|mm|feet|ft|inches?|in|kg|kilograms?|grams?|g|pounds?|lb|ounces?|oz|liters?|litres?|l|ml|gallons?|gal|celsius|fahrenheit|°c|°f)\s+(?:to|in)\s+(km|kilometers?|miles?|mi|meters?|metres?|cm|centimeters?|mm|feet|ft|inches?|in|kg|kilograms?|grams?|g|pounds?|lb|ounces?|oz|liters?|litres?|l|ml|gallons?|gal|celsius|fahrenheit|°c|°f)",
        text, flags=re.I,
    )
    if unit_match:
        value = float(unit_match.group(1))
        source, target = unit_match.group(2).lower(), unit_match.group(3).lower()
        def key(unit: str) -> str:
            return {"kilometer":"km", "kilometers":"km", "kilometre":"km", "kilometres":"km", "mile":"mi", "miles":"mi", "meter":"m", "meters":"m", "metre":"m", "metres":"m", "centimeter":"cm", "centimeters":"cm", "millimeter":"mm", "millimeters":"mm", "foot":"ft", "feet":"ft", "inch":"in", "inches":"in", "kilogram":"kg", "kilograms":"kg", "gram":"g", "grams":"g", "pound":"lb", "pounds":"lb", "ounce":"oz", "ounces":"oz", "liter":"l", "liters":"l", "litre":"l", "litres":"l", "milliliter":"ml", "milliliters":"ml", "gallon":"gal", "gallons":"gal", "°c":"celsius", "°f":"fahrenheit"}.get(unit, unit)
        source, target = key(source), key(target)
        if source in ("celsius", "fahrenheit") and target in ("celsius", "fahrenheit"):
            expression = f"({value}*9/5+32)" if source == "celsius" and target == "fahrenheit" else f"(({value}-32)*5/9)" if source == "fahrenheit" and target == "celsius" else str(value)
            return expression
        to_base = {"km":1000, "mi":1609.344, "m":1, "cm":.01, "mm":.001, "ft":.3048, "in":.0254, "kg":1, "g":.001, "lb":.45359237, "oz":.028349523125, "l":1, "ml":.001, "gal":3.785411784}
        groups = {"km":"length", "mi":"length", "m":"length", "cm":"length", "mm":"length", "ft":"length", "in":"length", "kg":"mass", "g":"mass", "lb":"mass", "oz":"mass", "l":"volume", "ml":"volume", "gal":"volume"}
        if source in groups and target in groups and groups[source] == groups[target]:
            return f"({value!r}*{to_base[source]!r}/{to_base[target]!r})"
        raise ValueError("Those units are incompatible. Supported conversions: length, mass, volume, Celsius and Fahrenheit.")
    s = normalize_request(text)
    # Convert remaining function phrase forms and parse until trailing punctuation.
    s = re.sub(r"\s*([+*/%(),])\s*", r"\1", s)
    s = re.sub(r"\s*-\s*", "-", s)
    s = re.sub(r"\s*\*\*\s*", "**", s)
    s = s.strip()
    if not s:
        raise ValueError("I couldn't identify a calculation. Try a request such as 'square root of 81'.")
    # Reject prose/units instead of accidentally interpreting part of a sentence.
    if not re.fullmatch(r"[0-9a-zA-Z_.,()+*/%\-^\s]+", s):
        raise ValueError("I couldn't translate that request into a math expression.")
    return s


def parse_model_json(content: str) -> dict[str, Any]:
    """Accept a JSON object even if the model wraps it in brief prose/fences."""
    content = content.strip()
    content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.I)
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.S)
        if not match:
            raise ValueError("Model response did not contain a usable object.")
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("Model response must be an object.")
    return data


class OllamaLLM:
    def __init__(self) -> None:
        self.history: list[dict[str, str]] = []

    def understand(self, prompt: str) -> dict[str, Any]:
        system = (
            "You are a math request interpreter. Extract exactly one calculator expression "
            "from the user's request; do not calculate it. Allowed syntax: numbers, + - * / // % **, "
            "pi, e, tau, and functions sqrt,cbrt,sin,cos,tan,asin,acos,atan,log,log10,log2,exp, "
            "factorial,mean,median,stdev,variance,pstdev,pvariance,gcd,lcm,comb,perm,round,abs. "
            "Trig input is radians unless the user explicitly says degrees, then wrap in radians(...). "
            "For percent of, use (percent/100)*base. Reply as JSON {\"expression\":\"...\",\"plan\":\"short status\"}. "
            "Do not include an answer or numeric result. If not a math request, expression is empty."
        )
        messages = [{"role": "system", "content": system}, *self.history[-6:], {"role": "user", "content": prompt}]
        payload = json.dumps({"model": OLLAMA_MODEL, "messages": messages, "stream": False, "format": "json", "options": {"temperature": 0}}).encode()
        request = urllib.request.Request(OLLAMA_HOST + "/api/chat", data=payload, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise RuntimeError(f"Cannot reach Ollama at {OLLAMA_HOST}. Start Ollama and pull {OLLAMA_MODEL}. ({exc})") from exc
        content = body.get("message", {}).get("content", "")
        if not isinstance(content, str):
            raise RuntimeError("Ollama returned no text content.")
        result = parse_model_json(content)
        expression = result.get("expression", "")
        if not isinstance(expression, str) or len(expression) > MAX_EXPRESSION_LENGTH:
            raise ValueError("Model expression must be a short string.")
        # Validate model expression with same AST calculator before trusting it.
        if expression:
            try:
                calculate(expression)
            except ValueError as exc:
                raise ValueError(f"Model proposed an invalid expression: {exc}") from exc
        self.history.extend([{"role": "user", "content": prompt}, {"role": "assistant", "content": content}])
        return {"expression": expression, "plan": str(result.get("plan", "Interpreting the request"))[:160]}


def format_result(result: int | float) -> str:
    if isinstance(result, float):
        return f"{result:.12g}"
    return str(result)


def main(argv: list[str]) -> int:
    prompt = " ".join(argv[1:]).strip()
    if not prompt:
        print('Usage: py main.py "square root of 144"')
        return 2
    try:
        llm = OllamaLLM()
        print(f"[LLM] Asking {OLLAMA_MODEL} to interpret the request...")
        try:
            interpreted = llm.understand(prompt)
            expression = interpreted["expression"]
            print(f"[Plan] {interpreted['plan']}")
        except (ValueError, RuntimeError, KeyError, json.JSONDecodeError) as exc:
            # The local translator keeps common math usable when the model output is malformed.
            print(f"[LLM note] {exc}")
            expression = ""
        if not expression:
            expression = extract_expression(prompt)
        result = calculate(expression)
        print(f"[Expression] {expression}")
        print(f"[Result] {format_result(result)}")
        return 0
    except (ValueError, RuntimeError) as exc:
        print(f"[Error] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
