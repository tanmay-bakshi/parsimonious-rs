use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use parsimonious::Expression;

fn build_repeated_literal() -> Expression {
    let piece = Expression::literal("abcXYZ123");
    Expression::one_or_more(piece).with_name("repeated_literal")
}

fn build_log_line_file() -> Expression {
    let timestamp = Expression::regex(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z")
        .expect("timestamp regex must compile");
    let level = Expression::one_of(vec![
        Expression::literal("INFO"),
        Expression::literal("WARN"),
        Expression::literal("ERROR"),
    ]);
    let module = Expression::regex(r"[a-z_][a-z_0-9]*").expect("module regex must compile");
    let message = Expression::regex(r"[^\n]*").expect("message regex must compile");

    let line = Expression::sequence(vec![
        timestamp,
        Expression::literal(" "),
        level,
        Expression::literal(" "),
        module,
        Expression::literal(": "),
        message,
        Expression::literal("\n"),
    ]);

    Expression::one_or_more(line).with_name("log_file")
}

fn bench_parse_end(c: &mut Criterion) {
    let mut group = c.benchmark_group("parse_end");

    let repeated_expr = build_repeated_literal();
    let repeated_input = "abcXYZ123".repeat(4000);
    group.bench_with_input(
        BenchmarkId::new("repeated_literal", repeated_input.len()),
        &repeated_input,
        |b, input| {
            b.iter(|| {
                let end = repeated_expr
                    .parse_end(input)
                    .expect("parse_end should succeed");
                assert_eq!(end, input.len());
            });
        },
    );

    let log_expr = build_log_line_file();
    let log_line = "2025-09-24T12:34:56Z INFO parser_core: matched rule packet_header in 182us\n";
    let log_input = log_line.repeat(2500);
    group.bench_with_input(
        BenchmarkId::new("log_lines", log_input.len()),
        &log_input,
        |b, input| {
            b.iter(|| {
                let end = log_expr.parse_end(input).expect("parse_end should succeed");
                assert_eq!(end, input.len());
            });
        },
    );

    group.finish();
}

criterion_group!(benches, bench_parse_end);
criterion_main!(benches);
