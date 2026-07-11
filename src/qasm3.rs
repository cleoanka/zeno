//! OpenQASM 3 front end (documented subset): hand-written lexer +
//! recursive-descent parser, in the same style as [`super`] (the
//! OpenQASM 2.0 front end).
//!
//! Reached exclusively through [`super::parse_str`]'s version dispatch:
//! sources that open with `OPENQASM 3;` or `OPENQASM 3.0;` are parsed here,
//! into the exact same [`Program`] IR and with the same
//! [`QasmError`] type.
//!
//! Supported subset (frozen; see `docs/QASM3.md` for the full matrix):
//! - `OPENQASM 3;` / `OPENQASM 3.0;` header (mandatory, first statement).
//! - `include "stdgates.inc";` — satisfied *internally*; the crate's native
//!   gate set is a superset. Additional OpenQASM 3 aliases: `U` → `u3`,
//!   `CX` → `cx`, `phase` → `p`, `cphase` → `cp`.
//! - `qubit[n] name;` / `qubit name;` (size 1) and `bit[n] name;` /
//!   `bit name;` declarations (declare-before-use, like OpenQASM 2).
//! - Gate calls with OpenQASM 2-style register broadcasting; `name[i]`
//!   indexing; user `gate` definitions expanded inline (nesting ≤ 128).
//! - Measurement both ways: `c = measure q;` / `c[i] = measure q[j];`
//!   (assignment form) and `measure q -> c;` (arrow form), with register
//!   broadcasting when sizes match.
//! - `reset`, `barrier` (no arguments = all qubits), and
//!   `if (creg == n) op;` plus the block form `if (creg == n) { op; ... }`,
//!   lowered to one [`Instr::If`] per contained op.
//! - Parameter expressions: the OpenQASM 2 grammar plus `π` (= `pi`) and
//!   `tau`/`τ` (= 2π).
//!
//! Everything else in OpenQASM 3 (for/while, def, gate modifiers, timing,
//! typed declarations, input/output, arrays, switch, extern, pragmas,
//! annotations, …) is rejected with an explicit error naming the feature.
//!
//! Every error carries an accurate 1-based line and column, names the
//! offending token and says what was expected.

use super::QasmError;
use crate::gates;
use crate::ir::{GateInstr, Instr, Program, Reg};
use std::collections::{HashMap, HashSet};

/// Maximum nesting depth when expanding user-defined gates (same limit as
/// the OpenQASM 2 front end).
const MAX_EXPANSION_DEPTH: usize = 128;

/// Maximum total number of qubits across all `qubit` declarations (same
/// limit as the OpenQASM 2 front end).
const MAX_TOTAL_QUBITS: u32 = 48;

/// Words that cannot be used as register, gate or parameter names.
const RESERVED: &[&str] = &[
    "OPENQASM", "include", "qubit", "bit", "qreg", "creg", "gate", "opaque", "measure", "reset",
    "barrier", "if", "else", "pi", "π", "tau", "τ", "U", "CX", "for", "while", "in", "def",
    "defcal", "cal", "ctrl", "negctrl", "inv", "pow", "delay", "duration", "stretch", "float",
    "int", "uint", "angle", "bool", "complex", "const", "input", "output", "array", "switch",
    "case", "default", "extern", "pragma", "let", "box", "return", "break", "continue", "end",
];

fn err(line: usize, col: usize, msg: impl Into<String>) -> QasmError {
    QasmError {
        line,
        col,
        msg: msg.into(),
    }
}

/// If `word` opens an OpenQASM 3 construct that is outside zeno's subset,
/// return the human name of the feature for the rejection message.
fn unsupported_feature(word: &str) -> Option<&'static str> {
    Some(match word {
        "for" => "for loops",
        "while" => "while loops",
        "def" => "def subroutines",
        "defcal" | "cal" => "calibration blocks ('defcal'/'cal')",
        "ctrl" | "negctrl" | "inv" | "pow" => "gate modifiers ('ctrl@'/'negctrl@'/'inv@'/'pow@')",
        "delay" => "'delay' instructions",
        "duration" | "stretch" => "'duration'/'stretch' timing types",
        "float" | "int" | "uint" | "angle" | "bool" | "complex" => {
            "typed classical declarations ('float'/'int'/'uint'/'angle'/'bool'/'complex')"
        }
        "const" => "'const' declarations",
        "input" | "output" => "'input'/'output' parameters",
        "array" => "arrays beyond 1-D qubit/bit registers",
        "switch" => "'switch' statements",
        "extern" => "'extern' declarations",
        "pragma" | "#pragma" => "pragma directives",
        "else" => "'else' clauses",
        "let" => "'let' register aliases",
        "box" => "'box' scopes",
        "return" | "break" | "continue" => "'return'/'break'/'continue' statements",
        _ => return None,
    })
}

/// Build the standard rejection error for a named unsupported feature.
fn unsupported(line: usize, col: usize, feature: &str) -> QasmError {
    err(
        line,
        col,
        format!("{feature} are not supported in zeno's OpenQASM 3 subset (see docs/QASM3.md)"),
    )
}

/// Parse OpenQASM 3 (subset) source into a [`Program`].
pub(super) fn parse_str(src: &str) -> Result<Program, QasmError> {
    let tokens = lex(src)?;
    Parser::new(tokens).parse_program()
}

// ---------------------------------------------------------------------------
// Lexer
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Ident(String),
    /// Numeric literal; `text` is the raw lexeme (kept for exact integer
    /// parsing), `value` its `f64` value, `is_int` whether it is a plain
    /// unsigned integer (no `.`, no exponent).
    Number {
        text: String,
        value: f64,
        is_int: bool,
    },
    Str(String),
    Semi,
    Comma,
    LParen,
    RParen,
    LBracket,
    RBracket,
    LBrace,
    RBrace,
    Plus,
    Minus,
    Star,
    Slash,
    Caret,
    Arrow,
    Eq,
    EqEq,
    At,
    Colon,
    Eof,
}

impl Tok {
    /// Human-readable description used in error messages.
    fn describe(&self) -> String {
        match self {
            Tok::Ident(s) => format!("'{s}'"),
            Tok::Number { text, .. } => format!("number '{text}'"),
            Tok::Str(s) => format!("string \"{s}\""),
            Tok::Semi => "';'".into(),
            Tok::Comma => "','".into(),
            Tok::LParen => "'('".into(),
            Tok::RParen => "')'".into(),
            Tok::LBracket => "'['".into(),
            Tok::RBracket => "']'".into(),
            Tok::LBrace => "'{'".into(),
            Tok::RBrace => "'}'".into(),
            Tok::Plus => "'+'".into(),
            Tok::Minus => "'-'".into(),
            Tok::Star => "'*'".into(),
            Tok::Slash => "'/'".into(),
            Tok::Caret => "'^'".into(),
            Tok::Arrow => "'->'".into(),
            Tok::Eq => "'='".into(),
            Tok::EqEq => "'=='".into(),
            Tok::At => "'@'".into(),
            Tok::Colon => "':'".into(),
            Tok::Eof => "end of input".into(),
        }
    }
}

#[derive(Debug, Clone)]
struct Token {
    tok: Tok,
    line: usize,
    col: usize,
}

/// Character stream with 1-based line/column tracking.
struct Chars<'a> {
    it: std::iter::Peekable<std::str::Chars<'a>>,
    line: usize,
    col: usize,
}

impl<'a> Chars<'a> {
    fn new(src: &'a str) -> Self {
        Chars {
            it: src.chars().peekable(),
            line: 1,
            col: 1,
        }
    }

    fn peek(&mut self) -> Option<char> {
        self.it.peek().copied()
    }

    fn bump(&mut self) -> Option<char> {
        let c = self.it.next()?;
        if c == '\n' {
            self.line += 1;
            self.col = 1;
        } else {
            self.col += 1;
        }
        Some(c)
    }
}

fn lex(src: &str) -> Result<Vec<Token>, QasmError> {
    let mut cs = Chars::new(src);
    let mut out = Vec::new();
    loop {
        // Skip whitespace.
        while matches!(cs.peek(), Some(c) if c.is_whitespace()) {
            cs.bump();
        }
        let (line, col) = (cs.line, cs.col);
        let Some(c) = cs.peek() else {
            out.push(Token {
                tok: Tok::Eof,
                line,
                col,
            });
            return Ok(out);
        };
        let tok = match c {
            '/' => {
                cs.bump();
                match cs.peek() {
                    Some('/') => {
                        while matches!(cs.peek(), Some(c) if c != '\n') {
                            cs.bump();
                        }
                        continue;
                    }
                    Some('*') => {
                        cs.bump();
                        let mut prev = '\0';
                        loop {
                            match cs.bump() {
                                Some('/') if prev == '*' => break,
                                Some(c) => prev = c,
                                None => {
                                    return Err(err(line, col, "unterminated block comment"));
                                }
                            }
                        }
                        continue;
                    }
                    _ => Tok::Slash,
                }
            }
            ';' => {
                cs.bump();
                Tok::Semi
            }
            ',' => {
                cs.bump();
                Tok::Comma
            }
            '(' => {
                cs.bump();
                Tok::LParen
            }
            ')' => {
                cs.bump();
                Tok::RParen
            }
            '[' => {
                cs.bump();
                Tok::LBracket
            }
            ']' => {
                cs.bump();
                Tok::RBracket
            }
            '{' => {
                cs.bump();
                Tok::LBrace
            }
            '}' => {
                cs.bump();
                Tok::RBrace
            }
            '+' => {
                cs.bump();
                Tok::Plus
            }
            '*' => {
                cs.bump();
                Tok::Star
            }
            '^' => {
                cs.bump();
                Tok::Caret
            }
            '@' => {
                cs.bump();
                Tok::At
            }
            ':' => {
                cs.bump();
                Tok::Colon
            }
            '-' => {
                cs.bump();
                if cs.peek() == Some('>') {
                    cs.bump();
                    Tok::Arrow
                } else {
                    Tok::Minus
                }
            }
            '=' => {
                cs.bump();
                if cs.peek() == Some('=') {
                    cs.bump();
                    Tok::EqEq
                } else {
                    Tok::Eq
                }
            }
            '#' => {
                cs.bump();
                let mut s = String::from("#");
                while matches!(cs.peek(), Some(c) if c.is_ascii_alphanumeric() || c == '_') {
                    s.push(cs.bump().unwrap());
                }
                if s == "#pragma" {
                    Tok::Ident(s)
                } else {
                    return Err(err(
                        line,
                        col,
                        format!(
                            "unexpected character '#' (in '{s}'; only '#pragma' is lexed, \
                                 and pragma directives are rejected)"
                        ),
                    ));
                }
            }
            '"' => {
                cs.bump();
                let mut s = String::new();
                loop {
                    match cs.bump() {
                        Some('"') => break,
                        Some('\n') | None => {
                            return Err(err(line, col, "unterminated string literal"));
                        }
                        Some(c) => s.push(c),
                    }
                }
                Tok::Str(s)
            }
            'π' | 'τ' => {
                cs.bump();
                Tok::Ident(c.to_string())
            }
            c if c.is_ascii_digit() || c == '.' => lex_number(&mut cs, line, col)?,
            c if c.is_ascii_alphabetic() || c == '_' => {
                let mut s = String::new();
                while matches!(cs.peek(), Some(c) if c.is_ascii_alphanumeric() || c == '_') {
                    s.push(cs.bump().unwrap());
                }
                Tok::Ident(s)
            }
            c => {
                return Err(err(line, col, format!("unexpected character '{c}'")));
            }
        };
        out.push(Token { tok, line, col });
    }
}

/// Lex a numeric literal: `[0-9]* ('.' [0-9]*)? ([eE] [+-]? [0-9]+)?`
/// with at least one mantissa digit.
fn lex_number(cs: &mut Chars, line: usize, col: usize) -> Result<Tok, QasmError> {
    let mut text = String::new();
    let mut has_digits = false;
    while matches!(cs.peek(), Some(c) if c.is_ascii_digit()) {
        text.push(cs.bump().unwrap());
        has_digits = true;
    }
    let mut is_int = true;
    if cs.peek() == Some('.') {
        is_int = false;
        text.push(cs.bump().unwrap());
        while matches!(cs.peek(), Some(c) if c.is_ascii_digit()) {
            text.push(cs.bump().unwrap());
            has_digits = true;
        }
    }
    if !has_digits {
        return Err(err(
            line,
            col,
            format!("malformed number '{text}': no digits"),
        ));
    }
    if matches!(cs.peek(), Some('e' | 'E')) {
        is_int = false;
        text.push(cs.bump().unwrap());
        if matches!(cs.peek(), Some('+' | '-')) {
            text.push(cs.bump().unwrap());
        }
        let mut exp_digits = false;
        while matches!(cs.peek(), Some(c) if c.is_ascii_digit()) {
            text.push(cs.bump().unwrap());
            exp_digits = true;
        }
        if !exp_digits {
            return Err(err(
                line,
                col,
                format!("malformed number '{text}': exponent has no digits"),
            ));
        }
    }
    let value = text
        .parse::<f64>()
        .map_err(|_| err(line, col, format!("malformed number '{text}'")))?;
    Ok(Tok::Number {
        text,
        value,
        is_int,
    })
}

// ---------------------------------------------------------------------------
// Parameter expressions
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Func {
    Sin,
    Cos,
    Tan,
    Exp,
    Ln,
    Sqrt,
}

impl Func {
    fn from_name(name: &str) -> Option<Func> {
        Some(match name {
            "sin" => Func::Sin,
            "cos" => Func::Cos,
            "tan" => Func::Tan,
            "exp" => Func::Exp,
            "ln" => Func::Ln,
            "sqrt" => Func::Sqrt,
            _ => return None,
        })
    }
}

/// Parameter-expression AST. Identifiers are resolved to formal-parameter
/// indices at parse time, so evaluation cannot fail.
#[derive(Debug, Clone, PartialEq)]
enum Expr {
    Num(f64),
    Pi,
    Tau,
    Param(usize),
    Neg(Box<Expr>),
    Bin(BinOp, Box<Expr>, Box<Expr>),
    Fun(Func, Box<Expr>),
}

impl Expr {
    fn eval(&self, params: &[f64]) -> f64 {
        match self {
            Expr::Num(v) => *v,
            Expr::Pi => std::f64::consts::PI,
            Expr::Tau => std::f64::consts::TAU,
            Expr::Param(i) => params[*i],
            Expr::Neg(e) => -e.eval(params),
            Expr::Bin(op, a, b) => {
                let (a, b) = (a.eval(params), b.eval(params));
                match op {
                    BinOp::Add => a + b,
                    BinOp::Sub => a - b,
                    BinOp::Mul => a * b,
                    BinOp::Div => a / b,
                    BinOp::Pow => a.powf(b),
                }
            }
            Expr::Fun(f, e) => {
                let v = e.eval(params);
                match f {
                    Func::Sin => v.sin(),
                    Func::Cos => v.cos(),
                    Func::Tan => v.tan(),
                    Func::Exp => v.exp(),
                    Func::Ln => v.ln(),
                    Func::Sqrt => v.sqrt(),
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// User-defined gates (macros)
// ---------------------------------------------------------------------------

/// One statement inside a `gate` body. Qubit arguments are indices into the
/// definition's formal qubit-argument list.
#[derive(Debug, Clone)]
enum MacroStmt {
    Call {
        /// Resolved callee name: a native gate or a previously defined macro
        /// (`U`/`CX`/`phase`/`cphase` are resolved at definition time).
        name: String,
        params: Vec<Expr>,
        args: Vec<usize>,
    },
    Barrier {
        args: Vec<usize>,
    },
}

#[derive(Debug, Clone)]
struct GateMacro {
    n_params: usize,
    n_qargs: usize,
    body: Vec<MacroStmt>,
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

/// A resolved qubit/clbit argument: a whole register or a single bit.
#[derive(Debug, Clone, Copy)]
enum ArgVal {
    /// (register index into `prog.qregs`/`prog.cregs`, size, global offset)
    Whole(usize, u32, u32),
    /// Global bit index.
    Bit(u32),
}

struct Parser {
    toks: Vec<Token>,
    pos: usize,
    prog: Program,
    /// Register name → index into `prog.qregs`.
    qregs: HashMap<String, usize>,
    /// Register name → index into `prog.cregs`.
    cregs: HashMap<String, usize>,
    macros: HashMap<String, GateMacro>,
}

impl Parser {
    fn new(toks: Vec<Token>) -> Self {
        Parser {
            toks,
            pos: 0,
            prog: Program::default(),
            qregs: HashMap::new(),
            cregs: HashMap::new(),
            macros: HashMap::new(),
        }
    }

    // -- token helpers ------------------------------------------------------

    fn peek(&self) -> &Token {
        &self.toks[self.pos.min(self.toks.len() - 1)]
    }

    /// The token `n` positions ahead of the cursor (saturating at Eof).
    fn peek_at(&self, n: usize) -> &Tok {
        &self.toks[(self.pos + n).min(self.toks.len() - 1)].tok
    }

    fn advance(&mut self) -> Token {
        let t = self.toks[self.pos.min(self.toks.len() - 1)].clone();
        if self.pos < self.toks.len() - 1 {
            self.pos += 1;
        }
        t
    }

    fn err_here(&self, msg: impl Into<String>) -> QasmError {
        let t = self.peek();
        err(t.line, t.col, msg)
    }

    fn expect(&mut self, want: &Tok, what: &str) -> Result<Token, QasmError> {
        if &self.peek().tok == want {
            Ok(self.advance())
        } else {
            Err(self.err_here(format!(
                "expected {} {what}, found {}",
                want.describe(),
                self.peek().tok.describe()
            )))
        }
    }

    fn eat(&mut self, want: &Tok) -> bool {
        if &self.peek().tok == want {
            self.advance();
            true
        } else {
            false
        }
    }

    /// Expect an identifier usable as a fresh name (register, gate, formal
    /// parameter, …); rejects reserved words.
    fn expect_name(&mut self, what: &str) -> Result<(String, usize, usize), QasmError> {
        let t = self.peek().clone();
        match &t.tok {
            Tok::Ident(s) if RESERVED.contains(&s.as_str()) => Err(err(
                t.line,
                t.col,
                format!("'{s}' is a reserved word and cannot be used as {what}"),
            )),
            Tok::Ident(s) => {
                let s = s.clone();
                self.advance();
                Ok((s, t.line, t.col))
            }
            other => Err(err(
                t.line,
                t.col,
                format!(
                    "expected an identifier ({what}), found {}",
                    other.describe()
                ),
            )),
        }
    }

    /// Expect an unsigned integer literal, parsed with overflow checking.
    fn expect_uint(&mut self, max: u64, what: &str) -> Result<(u64, usize, usize), QasmError> {
        let t = self.peek().clone();
        match &t.tok {
            Tok::Number { text, is_int, .. } if *is_int => {
                let v = text.parse::<u64>().map_err(|_| {
                    err(
                        t.line,
                        t.col,
                        format!("integer '{text}' is too large for {what}"),
                    )
                })?;
                if v > max {
                    return Err(err(
                        t.line,
                        t.col,
                        format!("integer '{text}' is too large for {what} (max {max})"),
                    ));
                }
                self.advance();
                Ok((v, t.line, t.col))
            }
            other => Err(err(
                t.line,
                t.col,
                format!(
                    "expected an unsigned integer ({what}), found {}",
                    other.describe()
                ),
            )),
        }
    }

    // -- program ------------------------------------------------------------

    fn parse_program(mut self) -> Result<Program, QasmError> {
        self.parse_header()?;
        while self.peek().tok != Tok::Eof {
            self.parse_statement()?;
        }
        Ok(self.prog)
    }

    fn parse_header(&mut self) -> Result<(), QasmError> {
        let t = self.peek().clone();
        match &t.tok {
            Tok::Ident(s) if s == "OPENQASM" => {
                self.advance();
            }
            other => {
                return Err(err(
                    t.line,
                    t.col,
                    format!(
                        "expected 'OPENQASM 3;' as the first statement, found {}",
                        other.describe()
                    ),
                ));
            }
        }
        let t = self.peek().clone();
        match &t.tok {
            Tok::Number { text, .. } if text == "3" || text == "3.0" => {
                self.advance();
            }
            Tok::Number { text, .. } => {
                return Err(err(
                    t.line,
                    t.col,
                    format!(
                        "unsupported OpenQASM version '{text}' for the OpenQASM 3 front end \
                         (write 'OPENQASM 3;' or 'OPENQASM 3.0;'; version 2.0 is handled by \
                         the OpenQASM 2 front end)"
                    ),
                ));
            }
            other => {
                return Err(err(
                    t.line,
                    t.col,
                    format!(
                        "expected version number '3' or '3.0' after 'OPENQASM', found {}",
                        other.describe()
                    ),
                ));
            }
        }
        self.expect(&Tok::Semi, "after the OPENQASM header")?;
        Ok(())
    }

    fn parse_statement(&mut self) -> Result<(), QasmError> {
        let t = self.peek().clone();
        if t.tok == Tok::At {
            return Err(unsupported(t.line, t.col, "annotations ('@name')"));
        }
        let Tok::Ident(word) = &t.tok else {
            return Err(err(
                t.line,
                t.col,
                format!("expected a statement, found {}", t.tok.describe()),
            ));
        };
        if let Some(feature) = unsupported_feature(word) {
            return Err(unsupported(t.line, t.col, feature));
        }
        match word.as_str() {
            "OPENQASM" => Err(err(
                t.line,
                t.col,
                "duplicate 'OPENQASM' header (it must appear exactly once, first)",
            )),
            "include" => self.parse_include(),
            "qubit" | "bit" => self.parse_reg_decl(),
            "qreg" => Err(err(
                t.line,
                t.col,
                "'qreg' is OpenQASM 2 syntax; in OpenQASM 3 declare quantum registers with \
                 'qubit[n] name;'",
            )),
            "creg" => Err(err(
                t.line,
                t.col,
                "'creg' is OpenQASM 2 syntax; in OpenQASM 3 declare classical registers with \
                 'bit[n] name;'",
            )),
            "gate" => self.parse_gate_def(),
            "opaque" => Err(err(
                t.line,
                t.col,
                "'opaque' gate declarations are unsupported (they have no simulation \
                 semantics); define the gate body with 'gate' instead",
            )),
            _ => {
                let instrs = self.parse_qop(false)?;
                self.prog.instrs.extend(instrs);
                Ok(())
            }
        }
    }

    fn parse_include(&mut self) -> Result<(), QasmError> {
        self.advance(); // 'include'
        let t = self.peek().clone();
        match &t.tok {
            Tok::Str(name) if name == "stdgates.inc" => {
                self.advance();
            }
            Tok::Str(name) => {
                return Err(err(
                    t.line,
                    t.col,
                    format!(
                        "cannot include \"{name}\": only \"stdgates.inc\" is supported \
                         (and it is satisfied internally — the native gate set is \
                         always available)"
                    ),
                ));
            }
            other => {
                return Err(err(
                    t.line,
                    t.col,
                    format!(
                        "expected a quoted filename after 'include', found {}",
                        other.describe()
                    ),
                ));
            }
        }
        self.expect(&Tok::Semi, "after the include")?;
        Ok(())
    }

    fn parse_reg_decl(&mut self) -> Result<(), QasmError> {
        let kw = self.advance(); // 'qubit' | 'bit'
        let is_q = matches!(&kw.tok, Tok::Ident(s) if s == "qubit");
        let (size, sline, scol) = if self.eat(&Tok::LBracket) {
            let (size, sline, scol) = self.expect_uint(u32::MAX as u64, "a register size")?;
            if size == 0 {
                return Err(err(sline, scol, "register size must be at least 1"));
            }
            self.expect(&Tok::RBracket, "after the register size")?;
            (size as u32, sline, scol)
        } else {
            (1, kw.line, kw.col)
        };
        let (name, nline, ncol) = self.expect_name("a register name")?;
        if self.qregs.contains_key(&name) || self.cregs.contains_key(&name) {
            return Err(err(
                nline,
                ncol,
                format!("register '{name}' is already declared"),
            ));
        }
        self.expect(&Tok::Semi, "after the register declaration")?;
        if is_q {
            let total = self.prog.n_qubits() as u64 + size as u64;
            if total > MAX_TOTAL_QUBITS as u64 {
                return Err(err(
                    sline,
                    scol,
                    format!(
                        "declaring 'qubit[{size}] {name}' would bring the total to {total} \
                         qubits; the maximum is {MAX_TOTAL_QUBITS}"
                    ),
                ));
            }
            self.qregs.insert(name.clone(), self.prog.qregs.len());
            self.prog.qregs.push(Reg { name, size });
        } else {
            self.cregs.insert(name.clone(), self.prog.cregs.len());
            self.prog.cregs.push(Reg { name, size });
        }
        Ok(())
    }

    // -- quantum operations ---------------------------------------------------

    /// Parse one quantum operation (gate call, measure — either form —,
    /// reset, barrier or if) including the trailing ';', returning the
    /// broadcast-expanded instructions. `in_if` restricts to the ops allowed
    /// under `if`.
    fn parse_qop(&mut self, in_if: bool) -> Result<Vec<Instr>, QasmError> {
        let t = self.peek().clone();
        if t.tok == Tok::At {
            return Err(unsupported(t.line, t.col, "annotations ('@name')"));
        }
        let Tok::Ident(word) = &t.tok else {
            return Err(err(
                t.line,
                t.col,
                format!("expected a quantum operation, found {}", t.tok.describe()),
            ));
        };
        if let Some(feature) = unsupported_feature(word) {
            return Err(unsupported(t.line, t.col, feature));
        }
        match word.as_str() {
            "measure" => self.parse_measure_arrow(),
            "reset" => self.parse_reset(),
            "barrier" if in_if => Err(err(
                t.line,
                t.col,
                "'barrier' is not allowed under 'if' \
                 (only gate calls, measure and reset are)",
            )),
            "barrier" => self.parse_barrier(),
            "if" if in_if => Err(err(t.line, t.col, "'if' statements cannot be nested")),
            "if" => self.parse_if(),
            "include" | "qubit" | "bit" | "qreg" | "creg" | "gate" | "opaque" | "OPENQASM" => {
                Err(err(
                    t.line,
                    t.col,
                    format!(
                        "'{word}' is not a quantum operation \
                         (only gate calls, measure and reset can appear under 'if')"
                    ),
                ))
            }
            _ if self.looks_like_assignment() => self.parse_measure_assign(),
            _ => self.parse_gate_call(),
        }
    }

    /// Lookahead: `ident =` or `ident [ uint ] =` starts a measurement
    /// assignment rather than a gate call.
    fn looks_like_assignment(&self) -> bool {
        match self.peek_at(1) {
            Tok::Eq => true,
            Tok::LBracket => {
                matches!(self.peek_at(2), Tok::Number { .. })
                    && *self.peek_at(3) == Tok::RBracket
                    && *self.peek_at(4) == Tok::Eq
            }
            _ => false,
        }
    }

    fn parse_gate_call(&mut self) -> Result<Vec<Instr>, QasmError> {
        let name_tok = self.advance();
        let (name, line, col) = match &name_tok.tok {
            Tok::Ident(s) => (s.clone(), name_tok.line, name_tok.col),
            _ => unreachable!("dispatched on Ident"),
        };
        // Resolve the callee: OpenQASM 3 builtins/aliases, then user macros,
        // then natives.
        let (resolved, n_params, arity, is_macro) = self.resolve_gate(&name, line, col)?;

        // Optional parameter list, evaluated immediately (top level has no
        // formal parameters in scope).
        let params = if self.peek().tok == Tok::LParen {
            let exprs = self.parse_param_list(&[])?;
            exprs.iter().map(|e| e.eval(&[])).collect::<Vec<f64>>()
        } else {
            Vec::new()
        };
        if params.len() != n_params {
            return Err(err(
                line,
                col,
                format!(
                    "gate '{name}' expects {n_params} parameter(s), got {}",
                    params.len()
                ),
            ));
        }

        // Qubit arguments.
        let mut args = Vec::new();
        loop {
            args.push(self.parse_q_arg()?);
            if !self.eat(&Tok::Comma) {
                break;
            }
        }
        self.expect(&Tok::Semi, "after the gate call")?;
        if args.len() != arity {
            return Err(err(
                line,
                col,
                format!(
                    "gate '{name}' expects {arity} qubit argument(s), got {}",
                    args.len()
                ),
            ));
        }

        // Broadcast and emit.
        let n = self.broadcast_len(&args, line, col)?;
        let mut out = Vec::new();
        for i in 0..n {
            let qubits: Vec<u32> = args
                .iter()
                .map(|a| match a {
                    ArgVal::Whole(_, _, off) => off + i,
                    ArgVal::Bit(q) => *q,
                })
                .collect();
            self.check_distinct(&qubits, &name, line, col)?;
            if is_macro {
                self.expand_macro(&resolved, &params, &qubits, 0, line, col, &mut out)?;
            } else {
                out.push(Instr::Gate(GateInstr {
                    name: resolved.clone(),
                    params: params.clone(),
                    qubits,
                }));
            }
        }
        Ok(out)
    }

    /// Resolve a gate name to `(resolved_name, n_params, arity, is_macro)`.
    ///
    /// OpenQASM 3 aliases handled here: the builtin `U` → `u3`, and the
    /// `stdgates.inc` compatibility names `CX` → `cx`, `phase` → `p`,
    /// `cphase` → `cp`.
    fn resolve_gate(
        &self,
        name: &str,
        line: usize,
        col: usize,
    ) -> Result<(String, usize, usize, bool), QasmError> {
        match name {
            "U" => return Ok(("u3".into(), 3, 1, false)),
            "CX" => return Ok(("cx".into(), 0, 2, false)),
            "phase" => return Ok(("p".into(), 1, 1, false)),
            "cphase" => return Ok(("cp".into(), 1, 2, false)),
            _ => {}
        }
        if let Some(m) = self.macros.get(name) {
            return Ok((name.into(), m.n_params, m.n_qargs, true));
        }
        if let Some(def) = gates::lookup(name) {
            return Ok((
                name.into(),
                def.n_params as usize,
                def.arity as usize,
                false,
            ));
        }
        let mut msg = format!("unknown gate '{name}'");
        let lower = name.to_lowercase();
        if lower != name && (gates::lookup(&lower).is_some() || self.macros.contains_key(&lower)) {
            msg.push_str(&format!(
                " (gate names are case-sensitive; did you mean '{lower}'?)"
            ));
        }
        Err(err(line, col, msg))
    }

    /// Parse `( expr, expr, ... )` with the given formal-parameter scope.
    fn parse_param_list(&mut self, scope: &[String]) -> Result<Vec<Expr>, QasmError> {
        self.expect(&Tok::LParen, "before the gate parameters")?;
        let mut exprs = Vec::new();
        if self.peek().tok != Tok::RParen {
            loop {
                exprs.push(self.parse_expr(scope)?);
                if !self.eat(&Tok::Comma) {
                    break;
                }
            }
        }
        self.expect(&Tok::RParen, "after the gate parameters")?;
        Ok(exprs)
    }

    /// Parse a quantum argument: `reg` or `reg[i]`, resolved against qregs.
    fn parse_q_arg(&mut self) -> Result<ArgVal, QasmError> {
        self.parse_reg_arg(true)
    }

    /// Parse a classical argument: `reg` or `reg[i]`, resolved against cregs.
    fn parse_c_arg(&mut self) -> Result<ArgVal, QasmError> {
        self.parse_reg_arg(false)
    }

    fn parse_reg_arg(&mut self, quantum: bool) -> Result<ArgVal, QasmError> {
        let t = self.peek().clone();
        let Tok::Ident(name) = &t.tok else {
            let kind = if quantum { "qubit" } else { "classical bit" };
            return Err(err(
                t.line,
                t.col,
                format!(
                    "expected a {kind} argument (a register name, optionally indexed), \
                     found {}",
                    t.tok.describe()
                ),
            ));
        };
        let name = name.clone();
        self.advance();
        let (idx, size, offset) = if quantum {
            match self.qregs.get(&name) {
                Some(&i) => (i, self.prog.qregs[i].size, self.prog.qreg_offset(i)),
                None => {
                    let msg = if self.cregs.contains_key(&name) {
                        format!(
                            "'{name}' is a classical register; a quantum register is expected here"
                        )
                    } else {
                        format!("unknown quantum register '{name}'")
                    };
                    return Err(err(t.line, t.col, msg));
                }
            }
        } else {
            match self.cregs.get(&name) {
                Some(&i) => (i, self.prog.cregs[i].size, self.prog.creg_offset(i)),
                None => {
                    let msg = if self.qregs.contains_key(&name) {
                        format!(
                            "'{name}' is a quantum register; a classical register is expected here"
                        )
                    } else {
                        format!("unknown classical register '{name}'")
                    };
                    return Err(err(t.line, t.col, msg));
                }
            }
        };
        if self.eat(&Tok::LBracket) {
            let (i, iline, icol) = self.expect_uint(u32::MAX as u64, "a register index")?;
            if self.peek().tok == Tok::Colon {
                return Err(self.err_here(format!(
                    "range indexing on register '{name}' is not supported in zeno's \
                     OpenQASM 3 subset (see docs/QASM3.md); index a single bit"
                )));
            }
            self.expect(&Tok::RBracket, "after the register index")?;
            if i >= size as u64 {
                return Err(err(
                    iline,
                    icol,
                    format!("index {i} out of range for register '{name}' of size {size}"),
                ));
            }
            Ok(ArgVal::Bit(offset + i as u32))
        } else {
            Ok(ArgVal::Whole(idx, size, offset))
        }
    }

    /// Broadcast length: all whole-register args must have equal size; if
    /// none, the length is 1.
    fn broadcast_len(&self, args: &[ArgVal], line: usize, col: usize) -> Result<u32, QasmError> {
        let mut n: Option<(u32, usize)> = None;
        for a in args {
            if let ArgVal::Whole(idx, size, _) = a {
                match n {
                    None => n = Some((*size, *idx)),
                    Some((s, first)) if s != *size => {
                        return Err(err(
                            line,
                            col,
                            format!(
                                "broadcast size mismatch: register '{}' has size {s} but \
                                 register '{}' has size {size}",
                                self.prog.qregs[first].name, self.prog.qregs[*idx].name
                            ),
                        ));
                    }
                    Some(_) => {}
                }
            }
        }
        Ok(n.map_or(1, |(s, _)| s))
    }

    /// Error if any qubit appears twice in one expanded gate application.
    fn check_distinct(
        &self,
        qubits: &[u32],
        gate: &str,
        line: usize,
        col: usize,
    ) -> Result<(), QasmError> {
        let mut seen = HashSet::new();
        for &q in qubits {
            if !seen.insert(q) {
                return Err(err(
                    line,
                    col,
                    format!(
                        "gate '{gate}' applied to duplicate qubit {} after broadcasting",
                        self.qubit_name(q)
                    ),
                ));
            }
        }
        Ok(())
    }

    /// Pretty name (`q[3]`) for a global qubit index, for error messages.
    fn qubit_name(&self, q: u32) -> String {
        let mut off = 0u32;
        for r in &self.prog.qregs {
            if q < off + r.size {
                return format!("'{}[{}]'", r.name, q - off);
            }
            off += r.size;
        }
        format!("#{q}")
    }

    /// `measure q -> c;` — the OpenQASM 2-compatible arrow form.
    fn parse_measure_arrow(&mut self) -> Result<Vec<Instr>, QasmError> {
        let kw = self.advance(); // 'measure'
        let src = self.parse_q_arg()?;
        self.expect(&Tok::Arrow, "between the qubit and the classical target")?;
        let dst = self.parse_c_arg()?;
        self.expect(&Tok::Semi, "after the measure statement")?;
        self.lower_measure(src, dst, kw.line, kw.col)
    }

    /// `c = measure q;` / `c[i] = measure q[j];` — the OpenQASM 3
    /// assignment form.
    fn parse_measure_assign(&mut self) -> Result<Vec<Instr>, QasmError> {
        let start = self.peek().clone();
        let dst = self.parse_c_arg()?;
        self.expect(&Tok::Eq, "in the measurement assignment")?;
        let t = self.peek().clone();
        match &t.tok {
            Tok::Ident(s) if s == "measure" => {
                self.advance();
            }
            other => {
                return Err(err(
                    t.line,
                    t.col,
                    format!(
                        "expected 'measure' after '=' (in this subset an assignment can \
                         only store a measurement result), found {}",
                        other.describe()
                    ),
                ));
            }
        }
        let src = self.parse_q_arg()?;
        self.expect(&Tok::Semi, "after the measurement assignment")?;
        self.lower_measure(src, dst, start.line, start.col)
    }

    /// Shared lowering for both measure forms: normalize size-1 whole
    /// registers, then emit per-bit [`Instr::Measure`]s with size checks.
    fn lower_measure(
        &self,
        src: ArgVal,
        dst: ArgVal,
        line: usize,
        col: usize,
    ) -> Result<Vec<Instr>, QasmError> {
        let norm = |a: ArgVal| match a {
            ArgVal::Whole(_, 1, off) => ArgVal::Bit(off),
            other => other,
        };
        match (norm(src), norm(dst)) {
            (ArgVal::Bit(q), ArgVal::Bit(c)) => Ok(vec![Instr::Measure { qubit: q, clbit: c }]),
            (ArgVal::Whole(qi, qs, qoff), ArgVal::Whole(ci, cs, coff)) => {
                if qs != cs {
                    return Err(err(
                        line,
                        col,
                        format!(
                            "measure size mismatch: quantum register '{}' has size {qs} but \
                             classical register '{}' has size {cs}",
                            self.prog.qregs[qi].name, self.prog.cregs[ci].name
                        ),
                    ));
                }
                Ok((0..qs)
                    .map(|i| Instr::Measure {
                        qubit: qoff + i,
                        clbit: coff + i,
                    })
                    .collect())
            }
            (ArgVal::Whole(qi, qs, _), ArgVal::Bit(_)) => Err(err(
                line,
                col,
                format!(
                    "cannot broadcast measure: '{}' is a register of size {qs} but the \
                     target is a single classical bit",
                    self.prog.qregs[qi].name
                ),
            )),
            (ArgVal::Bit(_), ArgVal::Whole(ci, cs, _)) => Err(err(
                line,
                col,
                format!(
                    "cannot broadcast measure: the source is a single qubit but '{}' is a \
                     classical register of size {cs}",
                    self.prog.cregs[ci].name
                ),
            )),
        }
    }

    fn parse_reset(&mut self) -> Result<Vec<Instr>, QasmError> {
        self.advance(); // 'reset'
        let arg = self.parse_q_arg()?;
        self.expect(&Tok::Semi, "after the reset statement")?;
        Ok(match arg {
            ArgVal::Bit(q) => vec![Instr::Reset { qubit: q }],
            ArgVal::Whole(_, size, off) => {
                (0..size).map(|i| Instr::Reset { qubit: off + i }).collect()
            }
        })
    }

    fn parse_barrier(&mut self) -> Result<Vec<Instr>, QasmError> {
        self.advance(); // 'barrier'

        // OpenQASM 3: a bare `barrier;` applies to all qubits declared so far.
        if self.eat(&Tok::Semi) {
            return Ok(vec![Instr::Barrier((0..self.prog.n_qubits()).collect())]);
        }
        let mut qubits = Vec::new();
        let mut seen = HashSet::new();
        loop {
            match self.parse_q_arg()? {
                ArgVal::Bit(q) => {
                    if seen.insert(q) {
                        qubits.push(q);
                    }
                }
                ArgVal::Whole(_, size, off) => {
                    for q in off..off + size {
                        if seen.insert(q) {
                            qubits.push(q);
                        }
                    }
                }
            }
            if !self.eat(&Tok::Comma) {
                break;
            }
        }
        self.expect(&Tok::Semi, "after the barrier")?;
        Ok(vec![Instr::Barrier(qubits)])
    }

    fn parse_if(&mut self) -> Result<Vec<Instr>, QasmError> {
        self.advance(); // 'if'
        self.expect(&Tok::LParen, "after 'if'")?;
        let t = self.peek().clone();
        let Tok::Ident(name) = &t.tok else {
            return Err(err(
                t.line,
                t.col,
                format!(
                    "expected a classical register name in the if condition, found {}",
                    t.tok.describe()
                ),
            ));
        };
        let name = name.clone();
        let Some(&creg) = self.cregs.get(&name) else {
            let msg = if self.qregs.contains_key(&name) {
                format!(
                    "'{name}' is a quantum register; an if condition tests a classical register"
                )
            } else {
                format!("unknown classical register '{name}' in if condition")
            };
            return Err(err(t.line, t.col, msg));
        };
        self.advance();
        if self.peek().tok == Tok::LBracket {
            return Err(self.err_here(format!(
                "an if condition must test a whole classical register \
                 ('if ({name} == n)'), not an indexed bit"
            )));
        }
        if self.peek().tok == Tok::Eq {
            return Err(self.err_here(
                "found a single '='; the comparison operator in an if condition is '=='",
            ));
        }
        self.expect(&Tok::EqEq, "in the if condition")?;
        let (value, _, _) = self.expect_uint(u64::MAX, "the if comparison value")?;
        self.expect(&Tok::RParen, "after the if condition")?;

        // Body: a single op, or a `{ op; op; ... }` block lowered to one
        // `Instr::If` per contained op.
        let ops = if self.eat(&Tok::LBrace) {
            let mut ops = Vec::new();
            while self.peek().tok != Tok::RBrace {
                if self.peek().tok == Tok::Eof {
                    return Err(self.err_here("unclosed 'if' block: expected '}'"));
                }
                ops.extend(self.parse_qop(true)?);
            }
            self.advance(); // '}'
            ops
        } else {
            self.parse_qop(true)?
        };
        Ok(ops
            .into_iter()
            .map(|op| Instr::If {
                creg,
                value,
                op: Box::new(op),
            })
            .collect())
    }

    // -- gate definitions ---------------------------------------------------

    fn parse_gate_def(&mut self) -> Result<(), QasmError> {
        self.advance(); // 'gate'
        let (name, nline, ncol) = self.expect_name("a gate name")?;
        if gates::lookup(&name).is_some() {
            return Err(err(
                nline,
                ncol,
                format!("gate '{name}' is already defined (it is a native gate)"),
            ));
        }
        if matches!(name.as_str(), "phase" | "cphase") {
            return Err(err(
                nline,
                ncol,
                format!("gate '{name}' is already defined (it is a built-in stdgates alias)"),
            ));
        }
        if self.macros.contains_key(&name) {
            return Err(err(
                nline,
                ncol,
                format!("gate '{name}' is already defined"),
            ));
        }

        // Formal parameters.
        let mut params: Vec<String> = Vec::new();
        if self.eat(&Tok::LParen) {
            if self.peek().tok != Tok::RParen {
                loop {
                    let (p, pline, pcol) = self.expect_name("a gate parameter name")?;
                    if Func::from_name(&p).is_some() {
                        return Err(err(
                            pline,
                            pcol,
                            format!("'{p}' is a built-in function and cannot be a parameter name"),
                        ));
                    }
                    if params.contains(&p) {
                        return Err(err(pline, pcol, format!("duplicate parameter name '{p}'")));
                    }
                    params.push(p);
                    if !self.eat(&Tok::Comma) {
                        break;
                    }
                }
            }
            self.expect(&Tok::RParen, "after the gate parameters")?;
        }

        // Formal qubit arguments.
        let mut qargs: Vec<String> = Vec::new();
        loop {
            let (q, qline, qcol) = self.expect_name("a gate qubit argument name")?;
            if qargs.contains(&q) || params.contains(&q) {
                return Err(err(qline, qcol, format!("duplicate argument name '{q}'")));
            }
            qargs.push(q);
            if !self.eat(&Tok::Comma) {
                break;
            }
        }
        self.expect(&Tok::LBrace, "before the gate body")?;

        // Body.
        let mut body = Vec::new();
        while self.peek().tok != Tok::RBrace {
            body.push(self.parse_gate_body_stmt(&name, &params, &qargs)?);
        }
        self.expect(&Tok::RBrace, "after the gate body")?;

        self.macros.insert(
            name,
            GateMacro {
                n_params: params.len(),
                n_qargs: qargs.len(),
                body,
            },
        );
        Ok(())
    }

    fn parse_gate_body_stmt(
        &mut self,
        gate_name: &str,
        params: &[String],
        qargs: &[String],
    ) -> Result<MacroStmt, QasmError> {
        let t = self.peek().clone();
        let Tok::Ident(word) = &t.tok else {
            return Err(err(
                t.line,
                t.col,
                format!(
                    "expected a gate call, 'barrier' or '}}' in the gate body, found {}",
                    t.tok.describe()
                ),
            ));
        };
        if let Some(feature) = unsupported_feature(word) {
            return Err(unsupported(t.line, t.col, feature));
        }
        match word.as_str() {
            "barrier" => {
                self.advance();
                let mut args = Vec::new();
                loop {
                    args.push(self.parse_formal_qarg(gate_name, qargs)?);
                    if !self.eat(&Tok::Comma) {
                        break;
                    }
                }
                self.expect(&Tok::Semi, "after the barrier")?;
                args.dedup();
                Ok(MacroStmt::Barrier { args })
            }
            "measure" | "reset" | "if" | "qubit" | "bit" | "qreg" | "creg" | "gate" | "opaque"
            | "include" => Err(err(
                t.line,
                t.col,
                format!(
                    "'{word}' is not allowed inside a gate body \
                     (only gate calls and 'barrier' are)"
                ),
            )),
            callee => {
                let callee = callee.to_string();
                self.advance();
                if callee == gate_name {
                    return Err(err(
                        t.line,
                        t.col,
                        format!(
                            "gate '{callee}' is not defined yet at this point \
                             (recursive gate definitions are not allowed)"
                        ),
                    ));
                }
                let (resolved, n_params, arity, _) = self.resolve_gate(&callee, t.line, t.col)?;
                let exprs = if self.peek().tok == Tok::LParen {
                    self.parse_param_list(params)?
                } else {
                    Vec::new()
                };
                if exprs.len() != n_params {
                    return Err(err(
                        t.line,
                        t.col,
                        format!(
                            "gate '{callee}' expects {n_params} parameter(s), got {}",
                            exprs.len()
                        ),
                    ));
                }
                let mut args = Vec::new();
                loop {
                    args.push(self.parse_formal_qarg(gate_name, qargs)?);
                    if !self.eat(&Tok::Comma) {
                        break;
                    }
                }
                self.expect(&Tok::Semi, "after the gate call")?;
                if args.len() != arity {
                    return Err(err(
                        t.line,
                        t.col,
                        format!(
                            "gate '{callee}' expects {arity} qubit argument(s), got {}",
                            args.len()
                        ),
                    ));
                }
                let mut seen = HashSet::new();
                if let Some(&dup) = args.iter().find(|&&a| !seen.insert(a)) {
                    return Err(err(
                        t.line,
                        t.col,
                        format!(
                            "gate '{callee}' applied to duplicate qubit argument '{}'",
                            qargs[dup]
                        ),
                    ));
                }
                Ok(MacroStmt::Call {
                    name: resolved,
                    params: exprs,
                    args,
                })
            }
        }
    }

    /// Parse a qubit argument inside a gate body: a formal argument name,
    /// never indexed.
    fn parse_formal_qarg(&mut self, gate_name: &str, qargs: &[String]) -> Result<usize, QasmError> {
        let t = self.peek().clone();
        let Tok::Ident(name) = &t.tok else {
            return Err(err(
                t.line,
                t.col,
                format!(
                    "expected a qubit argument of gate '{gate_name}', found {}",
                    t.tok.describe()
                ),
            ));
        };
        let name = name.clone();
        self.advance();
        if self.peek().tok == Tok::LBracket {
            return Err(self.err_here(format!(
                "cannot index '{name}' inside a gate body (gate arguments are single qubits)"
            )));
        }
        qargs.iter().position(|q| q == &name).ok_or_else(|| {
            err(
                t.line,
                t.col,
                format!("'{name}' is not declared as a qubit argument of gate '{gate_name}'"),
            )
        })
    }

    /// Inline-expand a user gate at a call site.
    #[allow(clippy::too_many_arguments)]
    fn expand_macro(
        &self,
        name: &str,
        params: &[f64],
        qubits: &[u32],
        depth: usize,
        line: usize,
        col: usize,
        out: &mut Vec<Instr>,
    ) -> Result<(), QasmError> {
        if depth >= MAX_EXPANSION_DEPTH {
            return Err(err(
                line,
                col,
                format!(
                    "expansion of gate '{name}' exceeds the nesting depth limit \
                     of {MAX_EXPANSION_DEPTH}"
                ),
            ));
        }
        let m = &self.macros[name];
        for stmt in &m.body {
            match stmt {
                MacroStmt::Call {
                    name: callee,
                    params: exprs,
                    args,
                } => {
                    let vals: Vec<f64> = exprs.iter().map(|e| e.eval(params)).collect();
                    let qs: Vec<u32> = args.iter().map(|&a| qubits[a]).collect();
                    if self.macros.contains_key(callee) {
                        self.expand_macro(callee, &vals, &qs, depth + 1, line, col, out)?;
                    } else {
                        out.push(Instr::Gate(GateInstr {
                            name: callee.clone(),
                            params: vals,
                            qubits: qs,
                        }));
                    }
                }
                MacroStmt::Barrier { args } => {
                    out.push(Instr::Barrier(args.iter().map(|&a| qubits[a]).collect()));
                }
            }
        }
        Ok(())
    }

    // -- expressions ----------------------------------------------------------

    /// expr := term { ('+'|'-') term }
    fn parse_expr(&mut self, scope: &[String]) -> Result<Expr, QasmError> {
        let mut lhs = self.parse_term(scope)?;
        loop {
            let op = match self.peek().tok {
                Tok::Plus => BinOp::Add,
                Tok::Minus => BinOp::Sub,
                _ => break,
            };
            self.advance();
            let rhs = self.parse_term(scope)?;
            lhs = Expr::Bin(op, Box::new(lhs), Box::new(rhs));
        }
        Ok(lhs)
    }

    /// term := unary { ('*'|'/') unary }
    fn parse_term(&mut self, scope: &[String]) -> Result<Expr, QasmError> {
        let mut lhs = self.parse_unary(scope)?;
        loop {
            let op = match self.peek().tok {
                Tok::Star => BinOp::Mul,
                Tok::Slash => BinOp::Div,
                _ => break,
            };
            self.advance();
            let rhs = self.parse_unary(scope)?;
            lhs = Expr::Bin(op, Box::new(lhs), Box::new(rhs));
        }
        Ok(lhs)
    }

    /// unary := '-' unary | power
    fn parse_unary(&mut self, scope: &[String]) -> Result<Expr, QasmError> {
        if self.eat(&Tok::Minus) {
            Ok(Expr::Neg(Box::new(self.parse_unary(scope)?)))
        } else {
            self.parse_power(scope)
        }
    }

    /// power := atom [ '^' unary ]   (right-associative, binds tightest)
    fn parse_power(&mut self, scope: &[String]) -> Result<Expr, QasmError> {
        let base = self.parse_atom(scope)?;
        if self.eat(&Tok::Caret) {
            let exp = self.parse_unary(scope)?;
            Ok(Expr::Bin(BinOp::Pow, Box::new(base), Box::new(exp)))
        } else {
            Ok(base)
        }
    }

    /// atom := number | 'pi'/'π' | 'tau'/'τ' | param | func '(' expr ')'
    ///       | '(' expr ')'
    fn parse_atom(&mut self, scope: &[String]) -> Result<Expr, QasmError> {
        let t = self.peek().clone();
        match &t.tok {
            Tok::Number { value, .. } => {
                let v = *value;
                self.advance();
                Ok(Expr::Num(v))
            }
            Tok::LParen => {
                self.advance();
                let e = self.parse_expr(scope)?;
                self.expect(&Tok::RParen, "to close the parenthesized expression")?;
                Ok(e)
            }
            Tok::Ident(name) if name == "pi" || name == "π" => {
                self.advance();
                Ok(Expr::Pi)
            }
            Tok::Ident(name) if name == "tau" || name == "τ" => {
                self.advance();
                Ok(Expr::Tau)
            }
            Tok::Ident(name) => {
                if let Some(f) = Func::from_name(name) {
                    self.advance();
                    self.expect(&Tok::LParen, "after the function name")?;
                    let e = self.parse_expr(scope)?;
                    self.expect(&Tok::RParen, "after the function argument")?;
                    return Ok(Expr::Fun(f, Box::new(e)));
                }
                if let Some(i) = scope.iter().position(|p| p == name) {
                    self.advance();
                    return Ok(Expr::Param(i));
                }
                let msg = if scope.is_empty() {
                    format!(
                        "unknown identifier '{name}' in expression \
                         (no gate parameters are in scope here)"
                    )
                } else {
                    format!(
                        "unknown identifier '{name}' in expression \
                         (gate parameters in scope: {})",
                        scope.join(", ")
                    )
                };
                Err(err(t.line, t.col, msg))
            }
            other => Err(err(
                t.line,
                t.col,
                format!(
                    "expected a number, 'pi'/'π', 'tau'/'τ', a parameter or '(' in \
                     expression, found {}",
                    other.describe()
                ),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// OpenQASM 3 bell, arrow-form measurement.
    const BELL_ARROW: &str = "OPENQASM 3;\nqubit[2] q;\nbit[2] c;\nh q[0];\ncx q[0],q[1];\n\
                              measure q -> c;\n";
    /// The same bell, assignment-form measurement.
    const BELL_ASSIGN: &str = "OPENQASM 3.0;\nqubit[2] q;\nbit[2] c;\nh q[0];\ncx q[0],q[1];\n\
                               c = measure q;\n";
    /// The same bell in OpenQASM 2.0, for the dispatch test.
    const BELL_QASM2: &str = "OPENQASM 2.0;\nqreg q[2];\ncreg c[2];\nh q[0];\ncx q[0],q[1];\n\
                              measure q -> c;\n";

    fn parse(src: &str) -> Program {
        parse_str(src).unwrap_or_else(|e| panic!("parse failed: {e}"))
    }

    fn gate(name: &str, params: &[f64], qubits: &[u32]) -> Instr {
        Instr::Gate(GateInstr {
            name: name.into(),
            params: params.to_vec(),
            qubits: qubits.to_vec(),
        })
    }

    fn bell_program() -> Program {
        Program {
            qregs: vec![Reg {
                name: "q".into(),
                size: 2,
            }],
            cregs: vec![Reg {
                name: "c".into(),
                size: 2,
            }],
            instrs: vec![
                gate("h", &[], &[0]),
                gate("cx", &[], &[0, 1]),
                Instr::Measure { qubit: 0, clbit: 0 },
                Instr::Measure { qubit: 1, clbit: 1 },
            ],
        }
    }

    fn assert_err(src: &str, line: usize, col: usize, substr: &str) {
        let e = parse_str(src).expect_err("expected a parse error");
        assert!(
            e.msg.contains(substr),
            "error message {:?} does not contain {:?}",
            e.msg,
            substr
        );
        assert_eq!((e.line, e.col), (line, col), "wrong position for: {e}");
    }

    // -- positive: structure, both measure forms, dispatch ----------------------

    #[test]
    fn bell_arrow_exact_program() {
        assert_eq!(parse(BELL_ARROW), bell_program());
    }

    #[test]
    fn bell_assign_exact_program() {
        assert_eq!(parse(BELL_ASSIGN), bell_program());
    }

    #[test]
    fn dispatch_routes_2_to_old_parser_and_3_here() {
        // The shared fixture must produce byte-identical Programs through the
        // single entry point, whichever front end the header selects.
        let p2 = crate::qasm::parse_str(BELL_QASM2).expect("2.0 fixture");
        let p3 = crate::qasm::parse_str(BELL_ARROW).expect("3 fixture");
        assert_eq!(p2, bell_program());
        assert_eq!(p3, bell_program());
        assert_eq!(p2, p3);
        // 'qreg' exists only in the 2.0 grammar: accepting it under a 2.0
        // header proves the old parser ran; rejecting it under a 3 header
        // (with the migration hint) proves this parser ran.
        let e = crate::qasm::parse_str("OPENQASM 3;\nqreg q[1];\n").unwrap_err();
        assert!(e.msg.contains("OpenQASM 2 syntax"), "msg: {}", e.msg);
        assert_eq!((e.line, e.col), (2, 1));
    }

    #[test]
    fn header_bare_3_and_3_dot_0_accepted() {
        assert_eq!(parse("OPENQASM 3;\nqubit[1] q;\n").n_qubits(), 1);
        assert_eq!(parse("OPENQASM 3.0;\nqubit[1] q;\n").n_qubits(), 1);
    }

    #[test]
    fn sizeless_declarations_are_single_bits() {
        let p = parse("OPENQASM 3;\nqubit q;\nbit c;\nh q;\nc = measure q;\n");
        assert_eq!(
            p,
            Program {
                qregs: vec![Reg {
                    name: "q".into(),
                    size: 1
                }],
                cregs: vec![Reg {
                    name: "c".into(),
                    size: 1
                }],
                instrs: vec![gate("h", &[], &[0]), Instr::Measure { qubit: 0, clbit: 0 }],
            }
        );
    }

    #[test]
    fn include_stdgates_is_internal() {
        let p = parse("OPENQASM 3;\ninclude \"stdgates.inc\";\nqubit[1] q;\nh q[0];\n");
        assert_eq!(p.instrs, vec![gate("h", &[], &[0])]);
    }

    #[test]
    fn stdgates_aliases_resolve_to_natives() {
        let src = "OPENQASM 3;\nqubit[2] q;\nphase(0.5) q[0];\ncphase(0.25) q[0],q[1];\n\
                   U(0.1,0.2,0.3) q[0];\nCX q[0],q[1];\np(0.75) q[1];\n";
        let p = parse(src);
        assert_eq!(
            p.instrs,
            vec![
                gate("p", &[0.5], &[0]),
                gate("cp", &[0.25], &[0, 1]),
                gate("u3", &[0.1, 0.2, 0.3], &[0]),
                gate("cx", &[], &[0, 1]),
                gate("p", &[0.75], &[1]),
            ]
        );
    }

    #[test]
    fn block_and_line_comments() {
        let src = "/* leading\n   block */\nOPENQASM 3; // trailing\nqubit[1] q;\n\
                   /* mid */ h q[0];\n";
        let p = parse(src);
        assert_eq!(p.instrs, vec![gate("h", &[], &[0])]);
        // ... and through the dispatching entry point (the sniffer must see
        // through the block comment).
        assert_eq!(crate::qasm::parse_str(src).unwrap().instrs.len(), 1);
    }

    #[test]
    fn declarations_are_order_independent_of_each_other() {
        // A declaration may appear after other statements, as long as it
        // precedes its own first use.
        let p = parse("OPENQASM 3;\nqubit[1] q;\nh q[0];\nbit[1] c;\nc[0] = measure q[0];\n");
        assert_eq!(
            p.instrs,
            vec![gate("h", &[], &[0]), Instr::Measure { qubit: 0, clbit: 0 }]
        );
    }

    // -- positive: broadcasting, measurement forms ------------------------------

    #[test]
    fn broadcast_h_and_cx() {
        let p = parse("OPENQASM 3;\nqubit[3] q;\nh q;\n");
        assert_eq!(
            p.instrs,
            vec![
                gate("h", &[], &[0]),
                gate("h", &[], &[1]),
                gate("h", &[], &[2])
            ]
        );
        let p = parse("OPENQASM 3;\nqubit[2] a;\nqubit[2] b;\ncx a,b[1];\n");
        assert_eq!(
            p.instrs,
            vec![gate("cx", &[], &[0, 3]), gate("cx", &[], &[1, 3])]
        );
    }

    #[test]
    fn measure_assign_broadcasts_with_offsets() {
        // 'a' shifts the global offsets of 'q' to make sure they are used.
        let p = parse("OPENQASM 3;\nqubit[1] a;\nqubit[2] q;\nbit[2] c;\nc = measure q;\n");
        assert_eq!(
            p.instrs,
            vec![
                Instr::Measure { qubit: 1, clbit: 0 },
                Instr::Measure { qubit: 2, clbit: 1 },
            ]
        );
    }

    #[test]
    fn measure_assign_indexed() {
        let p = parse("OPENQASM 3;\nqubit[2] q;\nbit[2] c;\nc[1] = measure q[0];\n");
        assert_eq!(p.instrs, vec![Instr::Measure { qubit: 0, clbit: 1 }]);
    }

    #[test]
    fn measure_arrow_indexed() {
        let p = parse("OPENQASM 3;\nqubit[2] q;\nbit[2] c;\nmeasure q[1] -> c[0];\n");
        assert_eq!(p.instrs, vec![Instr::Measure { qubit: 1, clbit: 0 }]);
    }

    #[test]
    fn reset_broadcasts() {
        let p = parse("OPENQASM 3;\nqubit[3] q;\nreset q;\nreset q[1];\n");
        assert_eq!(
            p.instrs,
            vec![
                Instr::Reset { qubit: 0 },
                Instr::Reset { qubit: 1 },
                Instr::Reset { qubit: 2 },
                Instr::Reset { qubit: 1 },
            ]
        );
    }

    #[test]
    fn barrier_with_args_dedupes() {
        let p = parse("OPENQASM 3;\nqubit[2] a;\nqubit[2] b;\nbarrier a[1], b, a;\n");
        assert_eq!(p.instrs, vec![Instr::Barrier(vec![1, 2, 3, 0])]);
    }

    #[test]
    fn barrier_no_args_is_all_qubits() {
        let p = parse("OPENQASM 3;\nqubit[2] a;\nqubit[1] b;\nbarrier;\n");
        assert_eq!(p.instrs, vec![Instr::Barrier(vec![0, 1, 2])]);
    }

    // -- positive: if (single statement and block form) -------------------------

    #[test]
    fn if_single_statement_lowers() {
        let src = "OPENQASM 3;\nqubit[1] q;\nbit[2] c0;\nbit[3] c1;\nif (c1 == 5) x q[0];\n";
        let p = parse(src);
        assert_eq!(
            p.instrs,
            vec![Instr::If {
                creg: 1, // index into Program.cregs — NOT the bit offset (2)
                value: 5,
                op: Box::new(gate("x", &[], &[0])),
            }]
        );
    }

    #[test]
    fn if_block_lowers_one_if_per_op() {
        let src = "OPENQASM 3;\nqubit[2] q;\nbit[2] c;\n\
                   if (c == 3) {\n  x q[0];\n  c[0] = measure q[0];\n  reset q[1];\n}\n";
        let p = parse(src);
        let wrap = |op: Instr| Instr::If {
            creg: 0,
            value: 3,
            op: Box::new(op),
        };
        assert_eq!(
            p.instrs,
            vec![
                wrap(gate("x", &[], &[0])),
                wrap(Instr::Measure { qubit: 0, clbit: 0 }),
                wrap(Instr::Reset { qubit: 1 }),
            ]
        );
    }

    #[test]
    fn if_block_broadcasts_inside() {
        let src = "OPENQASM 3;\nqubit[2] q;\nbit[2] c;\nif (c == 1) { h q; }\n";
        let p = parse(src);
        assert_eq!(p.instrs.len(), 2);
        assert!(matches!(&p.instrs[1], Instr::If { op, .. }
            if **op == gate("h", &[], &[1])));
    }

    #[test]
    fn if_empty_block_is_noop() {
        let p = parse("OPENQASM 3;\nqubit[1] q;\nbit[1] c;\nif (c == 1) { }\n");
        assert!(p.instrs.is_empty());
    }

    // -- positive: gate definitions ---------------------------------------------

    #[test]
    fn user_gate_with_params() {
        let src = "OPENQASM 3;\nqubit[1] q;\n\
                   gate rot(t,s) a { rz(t/2) a; ry(s*2) a; rz(-t) a; }\n\
                   rot(pi,0.25) q[0];\n";
        let p = parse(src);
        let pi = std::f64::consts::PI;
        assert_eq!(
            p.instrs,
            vec![
                gate("rz", &[pi / 2.0], &[0]),
                gate("ry", &[0.5], &[0]),
                gate("rz", &[-pi], &[0]),
            ]
        );
    }

    #[test]
    fn user_gate_nesting() {
        let src = "OPENQASM 3;\nqubit[3] q;\n\
                   gate maj a,b,c { cx c,b; cx c,a; ccx a,b,c; }\n\
                   gate twice a,b,c { maj a,b,c; maj c,b,a; }\n\
                   twice q[0],q[1],q[2];\n";
        let p = parse(src);
        assert_eq!(
            p.instrs,
            vec![
                gate("cx", &[], &[2, 1]),
                gate("cx", &[], &[2, 0]),
                gate("ccx", &[], &[0, 1, 2]),
                gate("cx", &[], &[0, 1]),
                gate("cx", &[], &[0, 2]),
                gate("ccx", &[], &[2, 1, 0]),
            ]
        );
    }

    #[test]
    fn user_gate_broadcasts_and_uses_aliases() {
        let src = "OPENQASM 3;\nqubit[2] q;\ngate flip a { x a; phase(pi) a; }\nflip q;\n";
        let p = parse(src);
        let pi = std::f64::consts::PI;
        assert_eq!(
            p.instrs,
            vec![
                gate("x", &[], &[0]),
                gate("p", &[pi], &[0]),
                gate("x", &[], &[1]),
                gate("p", &[pi], &[1]),
            ]
        );
    }

    // -- positive: expressions (π, tau/τ, parity with the 2.0 grammar) ----------

    fn param_of(expr: &str) -> f64 {
        let src = format!("OPENQASM 3;\nqubit[1] q;\nu1({expr}) q[0];\n");
        let p = parse(&src);
        match &p.instrs[0] {
            Instr::Gate(g) => g.params[0],
            other => panic!("expected a gate, got {other:?}"),
        }
    }

    #[test]
    fn expr_pi_ascii_and_unicode() {
        assert_eq!(param_of("pi/2"), std::f64::consts::FRAC_PI_2);
        assert_eq!(param_of("π/2"), std::f64::consts::FRAC_PI_2);
        assert_eq!(param_of("-π"), -std::f64::consts::PI);
    }

    #[test]
    fn expr_tau_ascii_and_unicode() {
        assert_eq!(param_of("tau"), std::f64::consts::TAU);
        assert_eq!(param_of("τ"), std::f64::consts::TAU);
        assert_eq!(param_of("τ/4"), std::f64::consts::FRAC_PI_2);
        assert_eq!(param_of("tau/2"), std::f64::consts::PI);
    }

    #[test]
    fn expr_precedence_functions_scientific() {
        assert_eq!(param_of("1+2*3"), 7.0);
        assert_eq!(param_of("2*3^2"), 18.0); // ^ binds tighter than *
        assert_eq!(param_of("2^3^2"), 512.0); // ^ is right-associative
        assert_eq!(param_of("1.5e2"), 150.0);
        assert_eq!(param_of(".5"), 0.5);
        let v = param_of("ln(exp(1))+sqrt(4)-cos(0)+tan(0)+sin(0)");
        assert!((v - 2.0).abs() < 1e-12);
    }

    // -- end-to-end ---------------------------------------------------------------

    #[test]
    fn e2e_ghz_counts() {
        let src = "OPENQASM 3.0;\ninclude \"stdgates.inc\";\nqubit[3] q;\nbit[3] c;\n\
                   h q[0];\ncx q[0], q[1];\ncx q[1], q[2];\nc = measure q;\n";
        let p = crate::qasm::parse_str(src).expect("dispatch + parse");
        let opts = crate::RunOptions {
            shots: 256,
            seed: Some(1),
            ..Default::default()
        };
        let r = crate::run_program(&p, &opts).expect("run");
        let keys: Vec<&str> = r.counts.0.keys().map(|s| s.as_str()).collect();
        assert_eq!(keys, ["000", "111"]);
    }

    // -- negative: every unsupported OpenQASM 3 feature is named ----------------

    #[test]
    fn err_for_loop() {
        assert_err(
            "OPENQASM 3;\nqubit[1] q;\nfor uint i in [0:3] { x q; }\n",
            3,
            1,
            "for loops are not supported in zeno's OpenQASM 3 subset",
        );
    }

    #[test]
    fn err_while_loop() {
        assert_err(
            "OPENQASM 3;\nqubit[1] q;\nbit[1] c;\nwhile (c == 0) { x q; }\n",
            4,
            1,
            "while loops are not supported in zeno's OpenQASM 3 subset",
        );
    }

    #[test]
    fn err_def_subroutine() {
        assert_err(
            "OPENQASM 3;\ndef flip(qubit q) -> bit { }\n",
            2,
            1,
            "def subroutines are not supported in zeno's OpenQASM 3 subset",
        );
    }

    #[test]
    fn err_ctrl_modifier() {
        assert_err(
            "OPENQASM 3;\nqubit[2] q;\nctrl @ x q[0], q[1];\n",
            3,
            1,
            "gate modifiers ('ctrl@'/'negctrl@'/'inv@'/'pow@') are not supported",
        );
    }

    #[test]
    fn err_negctrl_modifier() {
        assert_err(
            "OPENQASM 3;\nqubit[2] q;\nnegctrl @ x q[0], q[1];\n",
            3,
            1,
            "gate modifiers",
        );
    }

    #[test]
    fn err_inv_modifier() {
        assert_err(
            "OPENQASM 3;\nqubit[1] q;\ninv @ s q[0];\n",
            3,
            1,
            "gate modifiers",
        );
    }

    #[test]
    fn err_pow_modifier() {
        assert_err(
            "OPENQASM 3;\nqubit[1] q;\npow(2) @ x q[0];\n",
            3,
            1,
            "gate modifiers",
        );
    }

    #[test]
    fn err_modifier_under_if() {
        assert_err(
            "OPENQASM 3;\nqubit[1] q;\nbit[1] c;\nif (c == 1) inv @ s q[0];\n",
            4,
            13,
            "gate modifiers",
        );
    }

    #[test]
    fn err_delay() {
        assert_err(
            "OPENQASM 3;\nqubit[1] q;\ndelay[100ns] q;\n",
            3,
            1,
            "'delay' instructions are not supported in zeno's OpenQASM 3 subset",
        );
    }

    #[test]
    fn err_duration_and_stretch() {
        assert_err(
            "OPENQASM 3;\nduration t = 100ns;\n",
            2,
            1,
            "'duration'/'stretch' timing types are not supported",
        );
        assert_err(
            "OPENQASM 3;\nstretch s;\n",
            2,
            1,
            "'duration'/'stretch' timing types are not supported",
        );
    }

    #[test]
    fn err_typed_declarations() {
        for decl in ["float[64] f;", "int[32] i;", "angle[32] a;"] {
            let src = format!("OPENQASM 3;\n{decl}\n");
            assert_err(
                &src,
                2,
                1,
                "typed classical declarations ('float'/'int'/'uint'/'angle'/'bool'/'complex') \
                 are not supported",
            );
        }
    }

    #[test]
    fn err_const_declaration() {
        assert_err(
            "OPENQASM 3;\nconst float x = 1.0;\n",
            2,
            1,
            "'const' declarations are not supported in zeno's OpenQASM 3 subset",
        );
    }

    #[test]
    fn err_input_output() {
        assert_err(
            "OPENQASM 3;\ninput float theta;\n",
            2,
            1,
            "'input'/'output' parameters are not supported",
        );
        assert_err(
            "OPENQASM 3;\noutput bit result;\n",
            2,
            1,
            "'input'/'output' parameters are not supported",
        );
    }

    #[test]
    fn err_array() {
        assert_err(
            "OPENQASM 3;\narray[int[8], 4] a;\n",
            2,
            1,
            "arrays beyond 1-D qubit/bit registers are not supported",
        );
    }

    #[test]
    fn err_switch() {
        assert_err(
            "OPENQASM 3;\nbit[2] c;\nswitch (c) { case 1 { } }\n",
            3,
            1,
            "'switch' statements are not supported in zeno's OpenQASM 3 subset",
        );
    }

    #[test]
    fn err_extern() {
        assert_err(
            "OPENQASM 3;\nextern f(float) -> float;\n",
            2,
            1,
            "'extern' declarations are not supported in zeno's OpenQASM 3 subset",
        );
    }

    #[test]
    fn err_pragma_both_spellings() {
        assert_err(
            "OPENQASM 3;\npragma once\n",
            2,
            1,
            "pragma directives are not supported in zeno's OpenQASM 3 subset",
        );
        assert_err(
            "OPENQASM 3;\n#pragma once\n",
            2,
            1,
            "pragma directives are not supported in zeno's OpenQASM 3 subset",
        );
    }

    #[test]
    fn err_annotation() {
        assert_err(
            "OPENQASM 3;\n@reversible\nqubit[1] q;\n",
            2,
            1,
            "annotations ('@name') are not supported in zeno's OpenQASM 3 subset",
        );
    }

    #[test]
    fn err_else_clause() {
        assert_err(
            "OPENQASM 3;\nqubit[1] q;\nbit[1] c;\nif (c == 1) x q;\nelse y q;\n",
            5,
            1,
            "'else' clauses are not supported in zeno's OpenQASM 3 subset",
        );
    }

    #[test]
    fn err_range_index() {
        assert_err(
            "OPENQASM 3;\nqubit[4] q;\nh q[0:2];\n",
            3,
            6,
            "range indexing on register 'q' is not supported",
        );
    }

    // -- negative: migration and malformed input ---------------------------------

    #[test]
    fn err_qreg_creg_migration() {
        assert_err(
            "OPENQASM 3;\nqreg q[2];\n",
            2,
            1,
            "'qreg' is OpenQASM 2 syntax",
        );
        assert_err(
            "OPENQASM 3;\ncreg c[2];\n",
            2,
            1,
            "'creg' is OpenQASM 2 syntax",
        );
    }

    #[test]
    fn err_include_qelib1() {
        assert_err(
            "OPENQASM 3;\ninclude \"qelib1.inc\";\n",
            2,
            9,
            "only \"stdgates.inc\" is supported",
        );
    }

    #[test]
    fn err_wrong_version_reaches_this_front_end() {
        // Direct call (bypassing dispatch): the 3.x front end names what it
        // accepts.
        assert_err(
            "OPENQASM 2.0;\nqubit q;\n",
            1,
            10,
            "unsupported OpenQASM version '2.0'",
        );
    }

    #[test]
    fn err_measure_assign_size_mismatch_names_both() {
        let e = parse_str("OPENQASM 3;\nqubit[3] q;\nbit[5] c;\nc = measure q;\n")
            .expect_err("expected a size mismatch");
        assert!(e.msg.contains("'q' has size 3"), "msg: {}", e.msg);
        assert!(e.msg.contains("'c' has size 5"), "msg: {}", e.msg);
        assert_eq!((e.line, e.col), (4, 1));
    }

    #[test]
    fn err_measure_arrow_size_mismatch() {
        assert_err(
            "OPENQASM 3;\nqubit[3] q;\nbit[2] c;\nmeasure q -> c;\n",
            4,
            1,
            "measure size mismatch",
        );
    }

    #[test]
    fn err_assign_rhs_not_measure() {
        assert_err(
            "OPENQASM 3;\nqubit[1] q;\nbit[1] c;\nc = h q;\n",
            4,
            5,
            "expected 'measure' after '='",
        );
    }

    #[test]
    fn err_single_eq_in_if_condition() {
        assert_err(
            "OPENQASM 3;\nqubit[1] q;\nbit[1] c;\nif (c = 1) x q[0];\n",
            4,
            7,
            "single '='",
        );
    }

    #[test]
    fn err_barrier_under_if() {
        assert_err(
            "OPENQASM 3;\nqubit[1] q;\nbit[1] c;\nif (c == 1) { barrier q; }\n",
            4,
            15,
            "'barrier' is not allowed under 'if'",
        );
    }

    #[test]
    fn err_nested_if() {
        assert_err(
            "OPENQASM 3;\nqubit[1] q;\nbit[1] c;\nif (c == 1) { if (c == 1) x q[0]; }\n",
            4,
            15,
            "'if' statements cannot be nested",
        );
    }

    #[test]
    fn err_unclosed_if_block() {
        assert_err(
            "OPENQASM 3;\nqubit[1] q;\nbit[1] c;\nif (c == 1) { x q[0];\n",
            5,
            1,
            "unclosed 'if' block",
        );
    }

    #[test]
    fn err_declaration_under_if() {
        assert_err(
            "OPENQASM 3;\nqubit[1] q;\nbit[1] c;\nif (c == 1) { qubit r; }\n",
            4,
            15,
            "not a quantum operation",
        );
    }

    #[test]
    fn err_unknown_gate_case_hint() {
        assert_err(
            "OPENQASM 3;\nqubit[1] q;\nH q[0];\n",
            3,
            1,
            "did you mean 'h'",
        );
    }

    #[test]
    fn err_use_before_declare() {
        assert_err(
            "OPENQASM 3;\nh q[0];\nqubit[1] q;\n",
            2,
            3,
            "unknown quantum register 'q'",
        );
    }

    #[test]
    fn err_index_out_of_range() {
        assert_err(
            "OPENQASM 3;\nqubit[2] q;\nh q[5];\n",
            3,
            5,
            "index 5 out of range for register 'q' of size 2",
        );
    }

    #[test]
    fn err_duplicate_register() {
        assert_err(
            "OPENQASM 3;\nqubit[1] q;\nbit[1] q;\n",
            3,
            8,
            "already declared",
        );
    }

    #[test]
    fn err_missing_semicolon() {
        assert_err(
            "OPENQASM 3;\nqubit[1] q;\nh q[0]\nx q[0];\n",
            4,
            1,
            "expected ';'",
        );
    }

    #[test]
    fn err_zero_size_register() {
        assert_err("OPENQASM 3;\nqubit[0] q;\n", 2, 7, "at least 1");
    }

    #[test]
    fn err_too_many_qubits() {
        assert_err(
            "OPENQASM 3;\nqubit[40] a;\nqubit[9] b;\n",
            3,
            7,
            "maximum is 48",
        );
    }

    #[test]
    fn err_reserved_word_as_register_name() {
        assert_err(
            "OPENQASM 3;\nqubit[2] for;\n",
            2,
            10,
            "'for' is a reserved word",
        );
    }

    #[test]
    fn err_broadcast_size_mismatch() {
        assert_err(
            "OPENQASM 3;\nqubit[2] a;\nqubit[3] b;\ncx a,b;\n",
            4,
            1,
            "size mismatch",
        );
    }

    #[test]
    fn err_duplicate_qubit_after_broadcast() {
        assert_err(
            "OPENQASM 3;\nqubit[2] q;\ncx q[1],q;\n",
            3,
            1,
            "duplicate qubit 'q[1]'",
        );
    }

    #[test]
    fn err_measure_inside_gate_body() {
        assert_err(
            "OPENQASM 3;\ngate f a { measure a -> a; }\n",
            2,
            12,
            "not allowed inside a gate body",
        );
    }

    #[test]
    fn err_for_inside_gate_body() {
        assert_err(
            "OPENQASM 3;\ngate f a { for uint i in [0:1] { x a; } }\n",
            2,
            12,
            "for loops are not supported",
        );
    }

    #[test]
    fn err_recursive_gate_definition() {
        assert_err(
            "OPENQASM 3;\ngate f a { f a; }\n",
            2,
            12,
            "recursive gate definitions are not allowed",
        );
    }

    #[test]
    fn err_redefine_native_gate_or_alias() {
        assert_err("OPENQASM 3;\ngate h a { x a; }\n", 2, 6, "native gate");
        assert_err(
            "OPENQASM 3;\ngate phase(t) a { rz(t) a; }\n",
            2,
            6,
            "built-in stdgates alias",
        );
    }

    #[test]
    fn err_expansion_depth_limit() {
        let mut src = String::from("OPENQASM 3;\nqubit[1] q;\ngate g0 a { x a; }\n");
        for i in 1..=200 {
            src.push_str(&format!("gate g{i} a {{ g{} a; }}\n", i - 1));
        }
        src.push_str("g200 q[0];\n");
        let e = parse_str(&src).expect_err("expected depth error");
        assert!(
            e.msg.contains("nesting depth limit of 128"),
            "msg: {}",
            e.msg
        );
        assert_eq!((e.line, e.col), (204, 1));
    }

    #[test]
    fn deep_nesting_within_limit() {
        let mut src = String::from("OPENQASM 3;\nqubit[1] q;\ngate g0 a { x a; }\n");
        for i in 1..=100 {
            src.push_str(&format!("gate g{i} a {{ g{} a; }}\n", i - 1));
        }
        src.push_str("g100 q[0];\n");
        let p = parse(&src);
        assert_eq!(p.instrs, vec![gate("x", &[], &[0])]);
    }

    #[test]
    fn err_missing_header() {
        assert_err("qubit q;\n", 1, 1, "expected 'OPENQASM 3;'");
    }

    #[test]
    fn err_unterminated_block_comment() {
        assert_err("OPENQASM 3;\n/* forever\nqubit q;\n", 2, 1, "unterminated");
    }
}
