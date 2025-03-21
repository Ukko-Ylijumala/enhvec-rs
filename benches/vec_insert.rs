extern crate criterion;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

const APPEND_VEC_SIZE: usize = 32;

fn bench_rotate_right(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert_head");
    let test_vec: Vec<u32> = (0..APPEND_VEC_SIZE as u32).collect();

    for &size in &[64, 256, 512, 1024, 4096, 8192, 16384, 32768] {
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
        });

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
        });
    }
    group.finish();
}

criterion_group!(benches, bench_rotate_right);
criterion_main!(benches);
