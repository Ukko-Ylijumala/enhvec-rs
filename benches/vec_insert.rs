extern crate criterion;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::time::Duration;

const APPEND_VEC_SIZE: usize = 32;
const TEST_VEC_SIZES: [u32; 6] = [128, 512, 1024, 4096, 8192, 16384];

#[derive(Clone)]
struct TestStruct {
    _a: String,
    _b: String,
    _c: [usize; APPEND_VEC_SIZE],
}

impl TestStruct {
    fn new(num: usize) -> Self {
        TestStruct {
            _a: "Lorem ipsum dolor sit amet".to_string(),
            _b: "consectetur adipiscing elit".to_string(),
            _c: [num; APPEND_VEC_SIZE],
        }
    }
}

fn make_test_struct_vec(size: usize) -> Vec<TestStruct> {
    (0..size).map(|_| TestStruct::new(size)).collect()
}

fn bench_ins_u32(c: &mut Criterion) {
    let mut group = c.benchmark_group("ins_head_u32");
    let test_vec: Vec<u32> = (0..APPEND_VEC_SIZE as u32).collect();

    for &size in &TEST_VEC_SIZES {
        group.bench_with_input(
            BenchmarkId::new("rotate_right", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    let mut main_vec: Vec<u32> = (0..size).collect();
                    main_vec.extend(test_vec.iter());
                    main_vec.rotate_right(APPEND_VEC_SIZE);
                    main_vec
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("new_alloc", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    let main_vec: Vec<u32> = (0..size).collect();
                    let mut new_vec: Vec<u32> = Vec::with_capacity(size as usize + APPEND_VEC_SIZE);
                    new_vec.extend_from_slice(&test_vec);
                    new_vec.extend_from_slice(&main_vec);
                    drop(main_vec);
                    new_vec
                });
            },
        );
    }
    group.finish();
}

fn bench_ins_struct(c: &mut Criterion) {
    let mut group = c.benchmark_group("ins_head_struct");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(100);
    let test_vec: Vec<TestStruct> = make_test_struct_vec(APPEND_VEC_SIZE);

    for &size in &TEST_VEC_SIZES {
        group.bench_with_input(
            BenchmarkId::new("rotate_right", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    let mut main_vec: Vec<TestStruct> = make_test_struct_vec(size as usize);
                    main_vec.extend(test_vec.clone());
                    main_vec.rotate_right(APPEND_VEC_SIZE);
                    main_vec
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("new_alloc", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    let main_vec: Vec<TestStruct> = make_test_struct_vec(size as usize);
                    let mut new_vec: Vec<TestStruct> =
                        Vec::with_capacity(size as usize + APPEND_VEC_SIZE);
                    new_vec.extend_from_slice(&test_vec);
                    new_vec.extend_from_slice(&main_vec);
                    drop(main_vec);
                    new_vec
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_ins_u32, bench_ins_struct);
criterion_main!(benches);
