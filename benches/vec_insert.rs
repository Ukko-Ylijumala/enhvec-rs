extern crate criterion;
use criterion::{
    criterion_group, criterion_main, measurement::WallTime, BatchSize, BenchmarkGroup, BenchmarkId,
    Criterion,
};
use std::time::Duration;

const APPEND_VEC_SIZE: usize = 32;
const TEST_VEC_SIZES: [usize; 6] = [128, 512, 1024, 4096, 8192, 16384];

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

fn make_usize_vec(size: usize) -> Vec<usize> {
    (0..size).collect()
}

fn make_test_struct_vec(size: usize) -> Vec<TestStruct> {
    (0..size).map(|_| TestStruct::new(size)).collect()
}

/**
Benchmark the ways of folding a head of `APPEND_VEC_SIZE` elements into the
front of a main Vec, like `EnhVecInner::compact()` does. Elements are moved,
never cloned, and building the Vecs is not part of the measurement.
*/
fn bench_ins_head<T>(group: &mut BenchmarkGroup<WallTime>, make_vec: fn(usize) -> Vec<T>) {
    for &size in &TEST_VEC_SIZES {
        let setup = || (make_vec(size), make_vec(APPEND_VEC_SIZE));

        group.bench_function(BenchmarkId::new("rotate_right", size), |b| {
            b.iter_batched(
                setup,
                |(mut main_vec, mut head_vec)| {
                    main_vec.extend(head_vec.drain(..).rev());
                    main_vec.rotate_right(APPEND_VEC_SIZE);
                    (main_vec, head_vec)
                },
                BatchSize::LargeInput,
            );
        });

        group.bench_function(BenchmarkId::new("new_alloc", size), |b| {
            b.iter_batched(
                setup,
                |(mut main_vec, mut head_vec)| {
                    let mut new_vec: Vec<T> = Vec::with_capacity(main_vec.len() + head_vec.len());
                    new_vec.extend(head_vec.drain(..).rev());
                    new_vec.append(&mut main_vec);
                    (new_vec, head_vec)
                },
                BatchSize::LargeInput,
            );
        });

        group.bench_function(BenchmarkId::new("splice", size), |b| {
            b.iter_batched(
                setup,
                |(mut main_vec, mut head_vec)| {
                    main_vec.splice(0..0, head_vec.drain(..).rev());
                    (main_vec, head_vec)
                },
                BatchSize::LargeInput,
            );
        });

        // the main Vec usually has some spare capacity, which splice() can use
        let setup_spare = || {
            let (mut main_vec, head_vec) = setup();
            main_vec.reserve(APPEND_VEC_SIZE);
            (main_vec, head_vec)
        };
        group.bench_function(BenchmarkId::new("splice_spare_capacity", size), |b| {
            b.iter_batched(
                setup_spare,
                |(mut main_vec, mut head_vec)| {
                    main_vec.splice(0..0, head_vec.drain(..).rev());
                    (main_vec, head_vec)
                },
                BatchSize::LargeInput,
            );
        });
    }
}

fn bench_ins_usize(c: &mut Criterion) {
    let mut group = c.benchmark_group("ins_head_usize");
    bench_ins_head(&mut group, make_usize_vec);
    group.finish();
}

fn bench_ins_struct(c: &mut Criterion) {
    let mut group = c.benchmark_group("ins_head_struct");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(100);
    bench_ins_head(&mut group, make_test_struct_vec);
    group.finish();
}

criterion_group!(benches, bench_ins_usize, bench_ins_struct);
criterion_main!(benches);
