mod generate_compiling_code;

#[test]
fn tests() {
    let t = trybuild::TestCases::new();
    t.pass("tests/generate_compiling_code.rs");
    t.compile_fail("tests/wrong_return_type_in_process.rs");
}
