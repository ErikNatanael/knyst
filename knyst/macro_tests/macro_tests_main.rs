#[test]
#[cfg_attr(miri, ignore)]
fn tests() {
    let t = trybuild::TestCases::new();
    t.pass("macro_tests/generate_compiling_code.rs");
    t.compile_fail("macro_tests/wrong_return_type_in_process.rs");
}
