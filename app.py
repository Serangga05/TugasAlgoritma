from flask import Flask, render_template, request, jsonify, send_file
import re
import io
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from datetime import datetime
import json

app = Flask(__name__)

# -------------------------
# LEXER (scanner) — left-to-right with line/column tracking
# - improved: explicit BAR token for '|' and correct handling of NEWLINE/WHITESPACE
# -------------------------
TOKEN_SPECS = [
    ("COMMENT", r"//[^\n]*"),                 # comments
    ("NEWLINE", r"\n"),                       # newline (track line/col)
    ("WHITESPACE", r"[ \t\r]+"),              # spaces and tabs (skip)
    ("ARROW", r"->"),                         # arrow (guard separator)
    ("ASSIGN", r":="),                        # assignment
    ("OP", r"==|<=|>=|!=|[+\-*/<>]"),         # operators
    ("LPAREN", r"\("),
    ("RPAREN", r"\)"),
    ("SEMI", r";"),
    ("BAR", r"\|"),                           # '|' guard separator
    ("KEYWORD", r"\b(if|then|else|fi|do|od)\b"),  # keywords
    ("NUMBER", r"\b\d+\b"),
    ("IDENT", r"[A-Za-z_][A-Za-z0-9_]*"),
    ("UNKNOWN", r"."),                        # any other single char -> error
]

# Compose master regex; order matters (longer tokens first above)
master_regex = re.compile("|".join("(?P<%s>%s)" % pair for pair in TOKEN_SPECS), re.MULTILINE)

def lex(code: str):
    """
    Return list of tokens: dicts with type, value, line, col, index.
    Skips whitespace and comments. UNKNOWN tokens become ERROR tokens.
    """
    tokens = []
    line = 1
    col = 1
    pos = 0
    length = len(code)

    while pos < length:
        m = master_regex.match(code, pos)
        if not m:
            # fallback (shouldn't happen because UNKNOWN matches any char)
            tokens.append({"type": "ERROR", "value": code[pos], "line": line, "col": col, "index": pos})
            pos += 1
            col += 1
            continue

        kind = m.lastgroup
        value = m.group(kind)
        start = pos
        pos = m.end()

        if kind == "NEWLINE":
            line += 1
            col = 1
            continue
        if kind == "WHITESPACE" or kind == "COMMENT":
            # update column but skip token emission
            # handle possible \r in windows CRLF already matched by WHITESPACE
            col += len(value)
            continue

        if kind == "UNKNOWN":
            # mark as error token so parser can report
            tokens.append({"type": "ERROR", "value": value, "line": line, "col": col, "index": start})
            col += len(value)
            continue

        # Normal token
        tokens.append({"type": kind, "value": value, "line": line, "col": col, "index": start})
        col += len(value)

    return tokens

# -------------------------
# PARSER (recursive descent) for a small GCL subset
# Grammar (simplified):
# program    ::= stmt_list
# stmt_list  ::= stmt ( ';' stmt )*
# stmt       ::= if_stmt | do_stmt | assignment
# if_stmt    ::= 'if' guard_list 'fi'
# do_stmt    ::= 'do' guard_list 'od'
# guard_list ::= guard ( '|' guard )*
# guard      ::= expr '->' stmt_list
# assignment ::= IDENT ':=' expr
# expr       ::= equality
# ... (precedence implemented)
# -------------------------

class ParseError(Exception):
    def __init__(self, msg, token=None):
        super().__init__(msg)
        self.token = token

class Parser:
    def __init__(self, tokens):
        # tokens is a list of token dicts; parser expects lexical errors filtered out beforehand
        self.tokens = tokens
        self.pos = 0

    def peek(self):
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return {"type": "EOF", "value": "", "line": -1, "col": -1}

    def consume(self, expected_type=None, expected_value=None):
        tok = self.peek()
        if expected_type and tok["type"] != expected_type:
            raise ParseError(f"Expected token type {expected_type} but found {tok['type']} ('{tok['value']}')", tok)
        if expected_value and tok.get("value") != expected_value:
            raise ParseError(f"Expected token value {expected_value} but found {tok.get('value')}", tok)
        self.pos += 1
        return tok

    def parse_program(self):
        body = self.parse_stmt_list()
        return {"node": "Program", "body": body}

    def parse_stmt_list(self):
        stmts = []
        # Loop until EOF or closing keyword encountered
        while True:
            tok = self.peek()
            # stop conditions: EOF or end of guarded block
            if tok["type"] == "EOF":
                break
            if tok["type"] == "KEYWORD" and tok["value"] in ("fi", "od"):
                break
            # if token cannot start a statement, stop
            if tok["type"] not in ("KEYWORD", "IDENT"):
                break
            stmt = self.parse_stmt()
            stmts.append(stmt)
            # optional semicolon
            if self.peek()["type"] == "SEMI":
                self.consume("SEMI")
            # continue parsing further statements
        return stmts

    def parse_stmt(self):
        tok = self.peek()
        if tok["type"] == "KEYWORD" and tok["value"] == "if":
            return self.parse_if()
        if tok["type"] == "KEYWORD" and tok["value"] == "do":
            return self.parse_do()
        if tok["type"] == "IDENT":
            return self.parse_assignment()
        raise ParseError(f"Unexpected token in statement: {tok['type']} '{tok['value']}'", tok)

    def parse_if(self):
        self.consume("KEYWORD", "if")
        guards = self.parse_guard_list()
        # expect 'fi'
        tok = self.peek()
        if tok["type"] == "KEYWORD" and tok["value"] == "fi":
            self.consume("KEYWORD", "fi")
            return {"node": "If", "guards": guards}
        raise ParseError("Expected 'fi' to close 'if'", tok)

    def parse_do(self):
        self.consume("KEYWORD", "do")
        guards = self.parse_guard_list()
        tok = self.peek()
        if tok["type"] == "KEYWORD" and tok["value"] == "od":
            self.consume("KEYWORD", "od")
            return {"node": "Do", "guards": guards}
        raise ParseError("Expected 'od' to close 'do'", tok)

    def parse_guard_list(self):
        guards = []
        guards.append(self.parse_guard())
        while self.peek()["type"] == "BAR":
            self.consume("BAR")
            guards.append(self.parse_guard())
        return guards

    def parse_guard(self):
        cond = self.parse_expr()
        tok = self.peek()
        if tok["type"] == "ARROW":
            self.consume("ARROW")
        else:
            raise ParseError("Expected '->' in guard", tok)
        body = self.parse_stmt_list()
        return {"node": "Guard", "cond": cond, "body": body}

    def parse_assignment(self):
        ident = self.consume("IDENT")
        if self.peek()["type"] == "ASSIGN":
            self.consume("ASSIGN")
        else:
            raise ParseError("Expected ':=' in assignment", self.peek())
        expr = self.parse_expr()
        return {"node": "Assign", "target": ident["value"], "expr": expr}

    # Expression parsing (precedence)
    def parse_expr(self):
        return self.parse_equality()

    def parse_equality(self):
        left = self.parse_relational()
        while self.peek()["type"] == "OP" and self.peek()["value"] in ("==", "!="):
            op = self.consume("OP")["value"]
            right = self.parse_relational()
            left = {"node": "BinaryOp", "op": op, "left": left, "right": right}
        return left

    def parse_relational(self):
        left = self.parse_additive()
        while self.peek()["type"] == "OP" and self.peek()["value"] in ("<", ">", "<=", ">="):
            op = self.consume("OP")["value"]
            right = self.parse_additive()
            left = {"node": "BinaryOp", "op": op, "left": left, "right": right}
        return left

    def parse_additive(self):
        left = self.parse_term()
        while self.peek()["type"] == "OP" and self.peek()["value"] in ("+", "-"):
            op = self.consume("OP")["value"]
            right = self.parse_term()
            left = {"node": "BinaryOp", "op": op, "left": left, "right": right}
        return left

    def parse_term(self):
        left = self.parse_factor()
        while self.peek()["type"] == "OP" and self.peek()["value"] in ("*", "/"):
            op = self.consume("OP")["value"]
            right = self.parse_factor()
            left = {"node": "BinaryOp", "op": op, "left": left, "right": right}
        return left

    def parse_factor(self):
        tok = self.peek()
        if tok["type"] == "NUMBER":
            self.consume("NUMBER")
            return {"node": "Number", "value": int(tok["value"])}
        if tok["type"] == "IDENT":
            self.consume("IDENT")
            return {"node": "Ident", "name": tok["value"]}
        if tok["type"] == "LPAREN":
            self.consume("LPAREN")
            e = self.parse_expr()
            if self.peek()["type"] != "RPAREN":
                raise ParseError("Missing closing ')'", self.peek())
            self.consume("RPAREN")
            return e
        raise ParseError("Unexpected token in factor", tok)

def parse_tokens(tokens):
    # If lexical errors present, raise with the first error token
    lex_errors = [t for t in tokens if t["type"] == "ERROR"]
    if lex_errors:
        raise ParseError("Lexical errors present", lex_errors[0])

    # append EOF sentinel for parser convenience
    toks = [t for t in tokens]
    toks.append({"type": "EOF", "value": "", "line": -1, "col": -1})
    parser = Parser(toks)
    ast = parser.parse_program()
    return ast

# -------------------------
# PDF generation (reportlab) - same approach
# -------------------------
def pdf_from_tokens_and_ast(tokens, ast, title="GCL Analysis"):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4
    margin = 40
    y = height - margin

    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, y, title)
    c.setFont("Helvetica", 10)
    y -= 20
    c.drawString(margin, y, f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%SZ')}")
    y -= 25

    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Tokens:")
    y -= 18
    c.setFont("Helvetica", 9)

    for t in tokens:
        line = f"{t.get('line','?')}:{t.get('col','?')}  {t['type']:7}  {t['value']}"
        c.drawString(margin, y, line)
        y -= 12
        if y < margin + 80:
            c.showPage()
            y = height - margin

    # new page for AST
    c.showPage()
    y = height - margin
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "AST (JSON):")
    y -= 18
    c.setFont("Courier", 8)

    ast_text = json.dumps(ast, indent=2)
    for line in ast_text.splitlines():
        # naive wrap
        c.drawString(margin, y, line[:100])
        y -= 10
        if y < margin + 40:
            c.showPage()
            y = height - margin

    c.save()
    buf.seek(0)
    return buf

# -------------------------
# Flask routes
# -------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/scan", methods=["POST"])
def route_scan():
    code = request.form.get("code", "")
    tokens = lex(code)
    return jsonify(tokens)

@app.route("/parse", methods=["POST"])
def route_parse():
    code = request.form.get("code", "")
    tokens = lex(code)
    try:
        ast = parse_tokens(tokens)
        return jsonify({"ok": True, "ast": ast, "tokens": tokens})
    except ParseError as e:
        tok = getattr(e, "token", None)
        info = {"ok": False, "message": str(e)}
        if isinstance(tok, dict):
            info["token"] = tok
        info["tokens"] = tokens
        return jsonify(info), 400

@app.route("/export_pdf", methods=["POST"])
def route_export_pdf():
    code = request.form.get("code", "")
    tokens = lex(code)
    try:
        ast = parse_tokens(tokens)
    except ParseError:
        ast = {"error": "Parse failed - see tokens / lexical errors"}
    pdf_buf = pdf_from_tokens_and_ast(tokens, ast, title="GCL Lexical & AST Report")
    return send_file(pdf_buf, mimetype="application/pdf", as_attachment=True, download_name="gcl_report.pdf")

if __name__ == "__main__":
    app.run(debug=True)
