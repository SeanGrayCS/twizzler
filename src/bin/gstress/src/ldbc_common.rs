//! Shared scaffolding for the LDBC complex read arms: per-query parameter
//! loading, the digest format, latency accumulation and the per-query budget.
//!
//! Shared rather than copied because the two arms must agree byte for byte on
//! the digest.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::{Duration, Instant};

pub(crate) const DIR: &str = "/initrd";

// ---------------------------------------------------------------------------
// Digest
// ---------------------------------------------------------------------------

/// Sanitise a text value for a digest field.
///
/// Any of `,;|@~` becomes `_`. Organisation names contain commas, which would
/// otherwise collide with the field separator and make a `CDETAIL` line
/// unparseable. Byte-wise, so this agrees with the Python oracle on non-ASCII
/// input without either side needing to agree about characters.
pub(crate) fn san(s: &str) -> String {
    let bytes: Vec<u8> = s
        .bytes()
        .map(|b| match b {
            b',' | b';' | b'|' | b'@' | b'~' => b'_',
            other => other,
        })
        .collect();
    // Still valid UTF-8: every replaced byte is ASCII, and an ASCII byte never
    // occurs inside a multi-byte UTF-8 sequence. The `unwrap_or_else` is a
    // belt-and-braces path that cannot be taken.
    String::from_utf8(bytes).unwrap_or_else(|_| s.to_string())
}

const FNV64_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV64_PRIME: u64 = 0x0000_0100_0000_01b3;

pub(crate) fn fnv1a64(s: &str) -> String {
    let mut h = FNV64_OFFSET;
    for b in s.as_bytes() {
        h ^= u64::from(*b);
        h = h.wrapping_mul(FNV64_PRIME);
    }
    format!("{h:016x}")
}

/// Cross-language pin for the digest primitives.
///
/// The three-way comparison is only meaningful if Rust and Python produce the
/// same bytes, and a drift there would show up as every query disagreeing —
/// which reads like an engine bug. The expected values come from the host-side
/// Python oracle; re-run it against this after touching either `san` or
/// `fnv1a64`.
///
/// Runs in-band at startup because `gstress` is not built as a test binary, so
/// there is no unit-test slot for it and a check that never runs is not a
/// check.
pub(crate) fn self_check(tag: &str) -> bool {
    let hashes: &[(&str, &str)] = &[
        ("", "cbf29ce484222325"),
        ("a", "af63dc4c8601ec8c"),
        ("IC5", "4131cc19c9282130"),
        ("933|1266161530447", "e4ba230b9ea52a68"),
        // Non-ASCII, to pin that hashing is over UTF-8 bytes.
        (
            "École_Nationale_Supérieure_d'Électronique,_d'Électrotechnique",
            "5cfee03ad560a987",
        ),
        ("a,b;c|d@e~f", "90291e8249442453"),
    ];
    let sans: &[(&str, &str)] = &[
        ("a,b;c|d@e~f", "a_b_c_d_e_f"),
        // The substitution must not mangle multi-byte sequences.
        (
            "École_Nationale_Supérieure_d'Électronique,_d'Électrotechnique",
            "École_Nationale_Supérieure_d'Électronique__d'Électrotechnique",
        ),
    ];
    let mut ok = true;
    for (input, want) in hashes {
        let got = fnv1a64(input);
        if got != *want {
            println!("{tag} SELFCHECK FAIL fnv1a64({input:?}) = {got}, expected {want}");
            ok = false;
        }
    }
    for (input, want) in sans {
        let got = san(input);
        if got != *want {
            println!("{tag} SELFCHECK FAIL san({input:?}) = {got:?}, expected {want:?}");
            ok = false;
        }
    }
    println!(
        "{tag} SELFCHECK digest primitives {} — the three-way comparison is {}",
        if ok { "PASS" } else { "FAIL" },
        if ok {
            "meaningful"
        } else {
            "MEANINGLESS; fix this before reading any hash below"
        }
    );
    ok
}

/// Accumulates one query's canonical results, one per parameter, and prints the
/// summary line — plus, on request, the per-parameter detail lines.
///
/// Two levels on purpose: detail for every query at once is more lines than
/// the guest's stdout survives. The summary always fits; detail is asked for
/// one query at a time when a hash disagrees.
pub(crate) struct Digest {
    tag: &'static str,
    q: usize,
    entries: Vec<(String, String)>,
    counts: Vec<usize>,
    capped: bool,
}

impl Digest {
    pub(crate) fn new(tag: &'static str, q: usize) -> Self {
        Digest {
            tag,
            q,
            entries: Vec::new(),
            counts: Vec::new(),
            capped: false,
        }
    }

    pub(crate) fn record(&mut self, key: &str, rows: &[String]) {
        self.counts.push(rows.len());
        self.entries.push((key.to_string(), rows.join(";")));
    }

    /// Mark that a bound inside the query truncated the answer (IC14's path
    /// cap). A silent truncation would read as "few such paths exist".
    pub(crate) fn mark_capped(&mut self) {
        self.capped = true;
    }

    pub(crate) fn report(&self, detail: bool, host_s: f64) {
        let joined: Vec<String> = self
            .entries
            .iter()
            .map(|(k, c)| format!("{k}={c}"))
            .collect();
        let rows: usize = self.counts.iter().sum();
        let counts: Vec<String> = self.counts.iter().map(|c| c.to_string()).collect();
        println!(
            "{} CDIGEST IC{} params={} rows={} counts={} hash={} host={:.2}s{}{}",
            self.tag,
            self.q,
            self.entries.len(),
            rows,
            counts.join(","),
            fnv1a64(&joined.join("\n")),
            host_s,
            if self.capped { " CAPPED" } else { "" },
            // A query empty for every parameter still does the work, so it
            // remains a valid latency probe — but three implementations
            // agreeing on "empty" agree on nothing, so it must not be counted
            // as confirming equivalence.
            if rows == 0 {
                " WEAK-empty-for-every-parameter"
            } else {
                ""
            },
        );
        if detail {
            for (k, c) in &self.entries {
                println!("{} CDETAIL IC{} {} {}", self.tag, self.q, k, c);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------

/// One query's own substitution-parameter file, header included.
///
/// Not `ldbc_query::load_person_params`. That pools every file whose header
/// starts with `personId` into one id list, which is right for the short reads
/// — they take a person and nothing else — and wrong here: IC3 needs
/// `startDate|durationDays|countryXName|countryYName` from its file, and
/// IC13/IC14's header is `person1Id|person2Id`, which the pooling loader skips
/// entirely.
pub(crate) struct Params {
    pub(crate) head: Vec<String>,
    pub(crate) rows: Vec<Vec<String>>,
}

impl Params {
    /// A column of one row by header name. Returns `""` for an absent column,
    /// which cannot happen once `load` has checked the header — the queries
    /// name their own columns.
    pub(crate) fn get<'a>(&self, row: &'a [String], col: &str) -> &'a str {
        match self.head.iter().position(|h| h == col) {
            Some(i) => row.get(i).map(String::as_str).unwrap_or(""),
            None => "",
        }
    }

    pub(crate) fn num(&self, row: &[String], col: &str) -> i64 {
        self.get(row, col).parse().unwrap_or(0)
    }

    /// The parameter row as it appears in the file — the digest's key.
    pub(crate) fn key(&self, row: &[String]) -> String {
        row.join("|")
    }
}

pub(crate) fn load_params(q: usize) -> Option<Params> {
    let path = format!("{DIR}/interactive_{q}_param.txt");
    let f = match File::open(&path) {
        Ok(f) => f,
        Err(e) => {
            println!("GSTRESS LDBCC FAILED: cannot open {path}: {e:?} — re-run scripts/flatten_ldbc.py with --params");
            return None;
        }
    };
    let mut r = BufReader::new(f);
    let mut head_line = String::new();
    if r.read_line(&mut head_line).is_err() || head_line.trim().is_empty() {
        println!("GSTRESS LDBCC FAILED: {path} has no header");
        return None;
    }
    let head: Vec<String> = head_line
        .trim_end()
        .split('|')
        .map(str::to_string)
        .collect();
    let mut rows = Vec::new();
    for line in r.lines().map_while(|l| l.ok()) {
        let line = line.trim_end();
        if line.is_empty() {
            continue;
        }
        let vals: Vec<String> = line.split('|').map(str::to_string).collect();
        if vals.len() != head.len() {
            // Loud, not skipped: a short row here would silently change which
            // parameters the run used while the output still looked healthy.
            println!(
                "GSTRESS LDBCC FAILED: {path}: {} values for {} columns",
                vals.len(),
                head.len()
            );
            return None;
        }
        rows.push(vals);
    }
    if rows.is_empty() {
        println!("GSTRESS LDBCC FAILED: {path} has no parameter rows");
        return None;
    }
    Some(Params { head, rows })
}

// ---------------------------------------------------------------------------
// Latency — same shape as the short-read arms', so the two tables can be read
// side by side.
// ---------------------------------------------------------------------------

pub(crate) struct Lat {
    pub(crate) name: String,
    pub(crate) us: Vec<u128>,
    cold: Vec<u128>,
    warm: Vec<u128>,
    pub(crate) results: usize,
    /// Set when the query stopped on its wall-clock budget rather than
    /// completing `iters`.
    pub(crate) budget_limited: bool,
}

impl Lat {
    pub(crate) fn new(name: String) -> Self {
        Lat {
            name,
            us: Vec::new(),
            cold: Vec::new(),
            warm: Vec::new(),
            results: 0,
            budget_limited: false,
        }
    }

    pub(crate) fn push(&mut self, us: u128, first_touch: bool) {
        self.us.push(us);
        if first_touch {
            self.cold.push(us);
        } else {
            self.warm.push(us);
        }
    }

    fn pct(v: &mut Vec<u128>, p: f64) -> u128 {
        if v.is_empty() {
            return 0;
        }
        v.sort_unstable();
        v[(((v.len() - 1) as f64) * p) as usize]
    }

    pub(crate) fn report(&mut self, tag: &str) {
        if self.us.is_empty() {
            println!("{tag} {:<5} no samples", self.name);
            return;
        }
        self.us.sort_unstable();
        let n = self.us.len();
        let at = |p: f64| self.us[((n as f64 - 1.0) * p) as usize];
        let mean: u128 = self.us.iter().sum::<u128>() / n as u128;
        println!(
            "{tag} {:<5} n={:<5} mean {:>9}us  p50 {:>9}us  p95 {:>9}us  p99 {:>9}us  \
             max {:>9}us  rows/q {:.1}{}",
            self.name,
            n,
            mean,
            at(0.50),
            at(0.95),
            at(0.99),
            self.us[n - 1],
            self.results as f64 / n as f64,
            if self.budget_limited {
                "  BUDGET-LIMITED"
            } else {
                ""
            },
        );
    }

    pub(crate) fn report_split(&mut self, tag: &str) {
        let (cn, wn) = (self.cold.len(), self.warm.len());
        let (c50, c99) = (Self::pct(&mut self.cold, 0.50), Self::pct(&mut self.cold, 0.99));
        let (w50, w99) = (Self::pct(&mut self.warm, 0.50), Self::pct(&mut self.warm, 0.99));
        println!(
            "{tag} {:<5} cold n={cn} p50 {c50}us p99 {c99}us | warm n={wn} p50 {w50}us \
             p99 {w99}us | cold/warm p50 {:.1}x | warm p99/p50 {:.1}x",
            self.name,
            if w50 > 0 { c50 as f64 / w50 as f64 } else { 0.0 },
            if w50 > 0 { w99 as f64 / w50 as f64 } else { 0.0 },
        );
    }

    /// What this query's median would be if name resolution were charged per
    /// query instead of once per run. Printed beside the excluded figure,
    /// never instead of it.
    pub(crate) fn report_with_resolve(&mut self, tag: &str, resolve_us: u128) {
        if self.us.is_empty() || resolve_us == 0 {
            return;
        }
        self.us.sort_unstable();
        let p50 = self.us[((self.us.len() as f64 - 1.0) * 0.5) as usize];
        println!(
            "{tag} {:<5} p50 excluding name resolution {p50}us; charging it per query \
             would give {}us ({:.1}x)",
            self.name,
            p50 + resolve_us,
            (p50 + resolve_us) as f64 / p50.max(1) as f64,
        );
    }
}

// ---------------------------------------------------------------------------
// Budget
// ---------------------------------------------------------------------------

/// A per-query wall-clock bound. Default off.
///
/// A query that cannot finish then costs one query, not the run.
#[derive(Clone, Copy)]
pub(crate) struct Budget {
    limit: Option<Duration>,
    start: Instant,
}

impl Budget {
    pub(crate) fn new(secs: u64) -> Self {
        Budget {
            limit: if secs == 0 {
                None
            } else {
                Some(Duration::from_secs(secs))
            },
            start: Instant::now(),
        }
    }
    pub(crate) fn restart(&mut self) {
        self.start = Instant::now();
    }
    pub(crate) fn spent(&self) -> bool {
        match self.limit {
            Some(l) => self.start.elapsed() >= l,
            None => false,
        }
    }
    pub(crate) fn describe(&self) -> String {
        match self.limit {
            Some(l) => format!("{}s per query", l.as_secs()),
            None => "unbounded".to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// Argument parsing, shared so the two arms cannot drift
// ---------------------------------------------------------------------------

pub(crate) struct Args {
    pub(crate) iters: usize,
    pub(crate) queries: Vec<usize>,
    pub(crate) detail: Vec<usize>,
    pub(crate) budget_s: u64,
    pub(crate) digest_only: bool,
}

/// `gstress ldbc-complex[-indradb] [iters] [q:1,5,10] [detail:5] [budget:600] [digest]`
pub(crate) fn parse_args(skip: usize) -> Args {
    let rest: Vec<String> = std::env::args().skip(skip).collect();
    let mut a = Args {
        iters: 150,
        queries: (1..=14).collect(),
        detail: Vec::new(),
        budget_s: 0,
        digest_only: false,
    };
    for s in &rest {
        if let Some(v) = s.strip_prefix("q:") {
            a.queries = parse_list(v);
        } else if let Some(v) = s.strip_prefix("detail:") {
            a.detail = parse_list(v);
        } else if let Some(v) = s.strip_prefix("budget:") {
            a.budget_s = v.parse().unwrap_or(0);
        } else if s == "digest" {
            a.digest_only = true;
        } else if let Ok(n) = s.parse::<usize>() {
            a.iters = n;
        }
    }
    // `digest` means "one pass over the parameter list": the list cycles, so
    // iterations beyond it are exact repeats and add no information to an
    // equivalence check, while costing stdout lines the guest does not have.
    if a.digest_only {
        a.iters = 0; // resolved per query, to that query's parameter count
    }
    a
}

fn parse_list(v: &str) -> Vec<usize> {
    let mut out = Vec::new();
    for part in v.split(',') {
        let part = part.trim();
        if let Some((lo, hi)) = part.split_once('-') {
            if let (Ok(l), Ok(h)) = (lo.parse::<usize>(), hi.parse::<usize>()) {
                out.extend(l..=h);
            }
        } else if let Ok(n) = part.parse::<usize>() {
            out.push(n);
        }
    }
    out
}
